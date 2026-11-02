"""RAG core: two chunking strategies, E5 embeddings, two ChromaDB collections, grounded generation (Tasks 3-4)."""
import re
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings

import llm

ROOT = Path(__file__).parent
KB_DIR = ROOT / "kb"
MODEL_NAME = "intfloat/e5-small-v2"
FIXED_SIZE, FIXED_OVERLAP = 200, 50  # characters
COLLECTIONS = {"fixed": "ola_kb_fixed", "sentence": "ola_kb_sentence"}
DEFAULT_STRATEGY = "sentence"  # chosen in Task 5, see README
TOP_K = 3
THRESHOLD = 0.83  # Task 4: in-scope top-1 min 0.8499 vs out-of-scope max 0.8067, midpoint of the gap


def load_docs():
    """{doc_id: text}, with the markdown heading turned into a plain first line."""
    return {p.stem: p.read_text(encoding="utf-8").replace("# ", "").strip() for p in sorted(KB_DIR.glob("*.md"))}


def fixed_chunks(text, size=FIXED_SIZE, overlap=FIXED_OVERLAP):
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step)]


def sentence_chunks(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


CHUNKERS = {"fixed": fixed_chunks, "sentence": sentence_chunks}


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    try:  # zero network at runtime: use the locally cached model
        return SentenceTransformer(MODEL_NAME, local_files_only=True)
    except OSError:  # first run only: one-time download
        return SentenceTransformer(MODEL_NAME)


def embed(texts, kind):
    # E5 requires "query: " / "passage: " prefixes
    return _model().encode([f"{kind}: {t}" for t in texts], normalize_embeddings=True).tolist()


@lru_cache(maxsize=1)
def _client():
    # anonymized_telemetry=False: Chroma must not phone home (zero-network requirement)
    return chromadb.PersistentClient(path=str(ROOT / "chroma_db"), settings=Settings(anonymized_telemetry=False))


def collection(strategy=DEFAULT_STRATEGY):
    return _client().get_or_create_collection(COLLECTIONS[strategy], metadata={"hnsw:space": "cosine"})


def index_document(doc_id, text, strategy):
    chunks = CHUNKERS[strategy](text)
    collection(strategy).upsert(
        ids=[f"{doc_id}::{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embed(chunks, "passage"),
        metadatas=[{"doc_id": doc_id} for _ in chunks],
    )
    return len(chunks)


def add_document(doc_id, text):
    """Index a new doc into both collections; cached answers may now be stale, so drop them."""
    CACHE.clear()
    return {s: index_document(doc_id, text, s) for s in CHUNKERS}


def build_index():
    """Index every KB doc into BOTH collections. Returns {strategy: chunk_count}."""
    docs = load_docs()
    return {s: sum(index_document(d, t, s) for d, t in docs.items()) for s in CHUNKERS}


def retrieve(query, strategy=DEFAULT_STRATEGY, k=TOP_K):
    res = collection(strategy).query(query_embeddings=embed([query], "query"), n_results=k)
    return [
        {"text": t, "doc_id": m["doc_id"], "score": round(1 - d, 4)}  # cosine distance -> similarity
        for t, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


IDK = "I don't know. That isn't covered by Ola's support policies I have access to."


CACHE = {}  # Task 16: normalized query -> grounded_answer result (in-process)


def normalize(query):
    return " ".join(re.sub(r"[^\w\s-]", " ", query.lower()).split())


def grounded_answer(query, strategy=DEFAULT_STRATEGY, k=TOP_K, threshold=None):
    """Cached front of _grounded_answer: a repeated query skips retrieval and the LLM call."""
    key = (normalize(query), strategy, k, threshold)
    if key not in CACHE:
        CACHE[key] = _grounded_answer(query, strategy, k, threshold)
    return CACHE[key]


def _grounded_answer(query, strategy, k, threshold):
    """Answer using ONLY retrieved context; fall back to IDK when top-1 similarity is below threshold."""
    threshold = THRESHOLD if threshold is None else threshold
    hits = retrieve(query, strategy, k)
    top = hits[0]["score"] if hits else 0.0
    if top < threshold:
        return {"answer": IDK, "grounded": False, "top_score": top, "sources": [], "context": hits}
    answer = llm.generate_grounded(query, [h["text"] for h in hits])
    # similar topic, but nothing retrieved actually answers the query (mock: no overlap; real LLM: says so)
    if not answer.strip() or answer.lower().startswith("i don't know"):
        return {"answer": IDK, "grounded": False, "top_score": top, "sources": [], "context": hits}
    return {
        "answer": answer,
        "grounded": True,
        "top_score": top,
        "sources": sorted({h["doc_id"] for h in hits}),
        "context": hits,
    }
