"""Task demos. `python demos.py` runs every demo and writes each one's stdout to transcripts/NN_name.txt."""
import contextlib
import io
import os
import sys
from pathlib import Path

os.environ.setdefault("MOCK_LLM", "1")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import dataset
import evaluate
import rag


def t01_dataset():
    dataset.report()


def t03_chunk_index():
    counts = rag.build_index()
    for s, n in counts.items():
        print(f"Collection {rag.COLLECTIONS[s]} ({s} chunking): {n} chunks upserted, "
              f"{rag.collection(s).count()} in collection")
    q = "How long does Ola take to refund a wrong fare?"
    for s in rag.CHUNKERS:
        print(f"\nSample query on '{s}': {q}")
        for h in rag.retrieve(q, s):
            print(f"  {h['score']:.4f}  [{h['doc_id']}]  {h['text']!r}")


def t04_grounded():
    print("--- Calibration: top-1 cosine similarity, both collections ---")
    rows = [(q, "IN") for q, _ in evaluate.RAG_QUERIES] + [(q, "OUT") for q in evaluate.OUT_OF_SCOPE]
    for q, scope in rows:
        s = {st: rag.retrieve(q, st, 1)[0]["score"] for st in rag.CHUNKERS}
        print(f"{scope:<4} fixed={s['fixed']:.4f} sentence={s['sentence']:.4f}  {q}")
    print(f"\nChosen threshold: {rag.THRESHOLD} (collection: {rag.DEFAULT_STRATEGY})")
    print("\n--- Grounded generation: 5 in-scope + 1 out-of-scope ---")
    for q in [q for q, _ in evaluate.RAG_QUERIES] + evaluate.OUT_OF_SCOPE[:1]:
        r = rag.grounded_answer(q)
        print(f"\nQ: {q}\n   top-1={r['top_score']:.4f} grounded={r['grounded']} sources={r['sources']}\n   A: {r['answer']}")


def t05_chunking_eval():
    res = {s: evaluate.precision_recall(s) for s in rag.CHUNKERS}
    print("\nSummary: " + ", ".join(f"{s}: mean P={p:.2f}, mean R={r:.2f}" for s, (p, r) in res.items()))
    (fp, fr), (sp, sr) = res["fixed"], res["sentence"]
    print(f"\nRecommendation: deploy sentence chunking. Both reach mean recall {sr:.2f} (fixed {fr:.2f}), but sentence "
          f"chunks give higher mean precision ({sp:.2f} vs {fp:.2f}): each chunk is one self-contained policy rule, "
          "so fewer unrelated documents ride along in the top-3, and answers never start mid-sentence.")


DEMOS = {"01_dataset": t01_dataset, "03_chunk_index": t03_chunk_index,
         "04_grounded_generation": t04_grounded, "05_chunking_eval": t05_chunking_eval}

if __name__ == "__main__":
    out = Path("transcripts"); out.mkdir(exist_ok=True)
    only = sys.argv[1:]
    for name, fn in DEMOS.items():
        if only and not any(name.startswith(o) for o in only):
            continue
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print(f"# Transcript {name}  (MOCK_LLM={os.environ['MOCK_LLM']})\n")
            fn()
        (out / f"{name}.txt").write_text(buf.getvalue(), encoding="utf-8")
        print(buf.getvalue())
