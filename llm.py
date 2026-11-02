"""Single LLM switch. MOCK_LLM=1 (default, graded) is deterministic, needs no keys or network.
MOCK_LLM=0 uses Google Gemini via GEMINI_API_KEY."""
import os
import re

MOCK = os.getenv("MOCK_LLM", "1") != "0"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CALLS = {"generate_grounded": 0}  # call counter, used as cache evidence (Task 16)

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("a an the is are was were be to of in on for and or what how do does i my me can when if it with by at".split())


def _words(text):
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def generate_grounded(query, context_chunks):
    """Answer `query` using ONLY `context_chunks`."""
    CALLS["generate_grounded"] += 1
    if MOCK:
        # Extractive: up to 2 context sentences with the most query-word overlap (and at least half the best
        # overlap, so off-topic neighbours are dropped), kept in context order. Full sentences only: this skips
        # document titles and fixed-chunk fragments without an end mark.
        q = _words(query)
        sents = list(dict.fromkeys(s for c in context_chunks for s in _sentences(c) if s[-1] in ".!?"))
        overlap = {s: len(q & _words(s)) for s in sents}
        top = max(overlap.values(), default=0)
        best = sorted((s for s in sents if overlap[s] and overlap[s] * 2 >= top), key=lambda s: -overlap[s])[:2]
        return " ".join(s for s in sents if s in best) or sents[0]
    from google import genai  # ponytail: lazy import, only needed for the optional real-LLM path
    prompt = ("Answer the question using ONLY the context below. If the context does not contain the answer, "
              "say \"I don't know\".\n\nContext:\n" + "\n".join(context_chunks) + f"\n\nQuestion: {query}")
    return genai.Client().models.generate_content(model=GEMINI_MODEL, contents=prompt).text.strip()
