# Ola Domain Support Agent (CrewAI + Autogen + FastAPI)

**Track completed: Ola (Business Operations / Customer Support).**

A domain support agent for Ola that answers support-policy questions from a hand-written
knowledge base and checks a specific support ticket's status. Built with
CrewAI for orchestration, an Autogen review stage, and a FastAPI deployment.
It also has guardrails, session memory, evaluation and governance controls.

> **Runs fully under `MOCK_LLM` with zero API keys and zero network access.**
> Every transcript in [transcripts/](transcripts/) was produced in `MOCK_LLM` mode.
> CrewAI telemetry is disabled: `CREWAI_DISABLE_TELEMETRY=true` and `OTEL_SDK_DISABLED=true`
> are set in code before CrewAI is imported, and are also listed in `.env.example`.

---

## Dataset design choices (Part 1, Task 1). Used to reproduce the dataset

| Choice | Value |
|---|---|
| Seed | `TBD` |
| Record count | `TBD` (≥40) |
| Category weights | `TBD` |
| Status weights | `TBD` |
| `escalated` probability | `TBD` (result must land in 10–30%) |
| `resolution_time_hours` range | `TBD`, with reasoning: `TBD` |
| `days_since_created` | integer, 0–30 |

Generated counts (per category, per status, % escalated): `TBD`. These are pasted from `python dataset.py`.

## Quick start

```bash
# Python 3.10 (CrewAI does not support 3.14)
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python dataset.py                 # Task 1: dataset + stats
python run_demos.py               # Tasks 1-16: regenerates every transcript
uvicorn app:app --reload          # Task 11: API on http://127.0.0.1:8000
```

`MOCK_LLM=1` is the default. An optional real LLM (Google Gemini) can be enabled by copying
`.env.example` to `.env` and setting `MOCK_LLM=0` and `GEMINI_API_KEY=...`. No acceptance criterion depends on it.

Embeddings use the free local **E5** model (`intfloat/e5-small-v2`) via SentenceTransformers,
with E5's required `query: ` / `passage: ` prefixes.

## Results summary. Filled in after implementation

### RAG threshold calibration (Task 4)
| Query | In/Out scope | Top-1 cosine |
|---|---|---|
| `TBD` | | |

Chosen "I don't know" threshold: `TBD`. It sits between the in-scope and out-of-scope clusters observed above.

### Chunking comparison (Task 5)
Per-query precision/recall for both collections: see `transcripts/05_chunking_eval.txt`.
Recommendation: `TBD`

### Escalation score (Task 6)
Formula: `TBD`. Threshold: `TBD`, justified by: `TBD`

### Evaluation (Task 13)
Averages over 15 queries: Accuracy `TBD`, Grounding `TBD`, Completeness `TBD`, Safety `TBD`.

### Governance (Task 15)
- **Least autonomy:** `TBD`
- **Risk classification:** `TBD`
- **Runtime budget cap:** `TBD`

## PII scope
Phone numbers (fixed format) are masked on input, on the model path and in the logs.
Rider/driver names, addresses and payment details are free text with no fixed format, so
they are out of scope for masking. Only fabricated examples are used.

## Repository map
See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the build order.
