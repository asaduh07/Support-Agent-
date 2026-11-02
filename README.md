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
| Seed | `42` (`random.Random(42)`) |
| Record count | `50` |
| Category weights | Billing 0.30, Technical Issue 0.25, Account Access 0.15, Product Defect 0.10, General Inquiry 0.20 |
| Status weights | Open 0.20, In Progress 0.20, Escalated 0.10, Resolved 0.30, Closed 0.20 |
| `escalated` rule | `True` if status is `Escalated`, else `True` with probability 0.12 |
| `resolution_time_hours` range | uniform 1.0–72.0, rounded to 0.1. Most rider/driver tickets close within a shift, but billing and defect investigations can take up to 3 days |
| `days_since_created` | integer, 0–30 |

Generated with the first draw on seed 42; no seed or weight changes were needed. Output of `python dataset.py`:
- Per category: Billing 18, Technical Issue 10, Account Access 7, Product Defect 3, General Inquiry 12
- Per status: Open 12, In Progress 7, Escalated 5, Resolved 13, Closed 13
- `escalated=True`: 12/50 = **24.0%**

## Quick start

```bash
# Python 3.10 (CrewAI does not support 3.14)
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python dataset.py                 # Task 1: dataset + stats
python demos.py                   # regenerates every transcript (or: python demos.py 04 05)
uvicorn app:app --reload          # Task 11: API on http://127.0.0.1:8000
```

`MOCK_LLM=1` is the default. An optional real LLM (Google Gemini) can be enabled by copying
`.env.example` to `.env` and setting `MOCK_LLM=0` and `GEMINI_API_KEY=...`. No acceptance criterion depends on it.

Embeddings use the free local **E5** model (`intfloat/e5-small-v2`) via SentenceTransformers,
with E5's required `query: ` / `passage: ` prefixes.

## Results summary. Filled in after implementation

### RAG threshold calibration (Task 4)
Measured with E5 (`intfloat/e5-small-v2`); full output is in `transcripts/04_grounded_generation.txt`.

| Query | Scope | Top-1 fixed | Top-1 sentence |
|---|---|---|---|
| How quickly must a P1 ticket be resolved? | IN | 0.9092 | 0.9048 |
| When will my refund reach my bank account? | IN | 0.8508 | 0.8754 |
| Who approves a refund above 2,000 rupees? | IN | 0.8521 | 0.8499 |
| What happens if I complain about the same issue again within two weeks? | IN | 0.8775 | 0.8931 |
| How long are support tickets kept before they are deleted? | IN | 0.9142 | 0.9125 |
| What is the capital of France? | OUT | 0.7668 | 0.7609 |
| Can you give me a good recipe for chicken biryani? | OUT | 0.7857 | 0.7830 |
| Who won the cricket world cup in 2011? | OUT | 0.7376 | 0.7426 |
| What is the weather in Mumbai today? | OUT | 0.7718 | 0.7875 |
| How do I reset my Netflix password? | OUT | 0.8054 | 0.8067 |
| Tell me a joke | OUT | 0.7906 | 0.7929 |

The in-scope cluster spans **0.8499–0.9142** and the out-of-scope cluster spans **0.7376–0.8067**. E5 scores even unrelated text around 0.75–0.80, so tutorial presets such as 0.5, 0.6 or 0.7 would let every query through.
**Chosen threshold: 0.83**, roughly the midpoint of the 0.8067–0.8499 gap. The closest out-of-scope probe ("reset my Netflix password", which sounds like a support question) sits 0.023 below it.

### Chunking comparison (Task 5)
Per-query precision/recall arithmetic for both collections is in `transcripts/05_chunking_eval.txt` (top-3, document level, deduplicated).

| Collection | Mean precision | Mean recall |
|---|---|---|
| `ola_kb_fixed` (200 chars, 50 overlap) | 0.70 | 1.00 |
| `ola_kb_sentence` | **0.80** | 1.00 |

**Recommendation: deploy sentence chunking.** Both strategies reach a mean recall of 1.00, but sentence chunks give higher mean precision (0.80 vs 0.70). Each chunk is one self-contained policy rule, so fewer unrelated documents come along in the top-3 (for example, the refund query pulls in 3 documents with fixed chunking against 2 with sentence chunking). Answers also never start mid-sentence. The crew uses `ola_kb_sentence`.

### Escalation score (Task 6)
`escalation_score = 0.5 * escalated + 0.5 * min(days_since_created, 30) / 30`, which is always in [0, 1]. Older tickets score higher, because they are closer to breaching their SLA.

**Recommend escalation when the score is ≥ 0.75.** That is the escalated flag (0.5) plus an age at the dataset's **median `days_since_created` of 15 days** (0.25). So it flags escalated tickets older than half of all tickets: **8/50 = 16%**. A non-escalated ticket maxes out at 0.5, so age alone never triggers escalation. See `transcripts/06_ticket_tool.txt`.

### Evaluation (Task 13)
15 queries: one per required KB topic (12), plus 1 out-of-scope query, 1 prompt injection and 1 PII + ticket lookup. Each runs through the full pipeline and is scored by an LLM-as-judge (`llm.JUDGE_PROMPT`). Under `MOCK_LLM`, the judge applies that prompt's rubric deterministically:
- **Accuracy:** share of answer sentences that state a reference fact.
- **Grounding:** share of answer sentences whose content words are at least 60% present in the retrieved context or ticket record.
- **Completeness:** share of reference facts present in the answer.
- **Safety:** no raw phone number, and expected refusals actually refused.

Per-query scores are in `transcripts/13_evaluation.txt`.

| Accuracy | Grounding | Completeness | Safety |
|---|---|---|---|
| **0.733** | **1.000** | **0.933** | **1.000** |

How to read these numbers:
- **Grounding is 1.0 by construction.** The mock generator is extractive, so it can only copy retrieved sentences.
- **Accuracy is lowered by second sentences that are on-topic but not the reference fact.**
- **One real miss:** the outage-frequency query got the "within 30 minutes" sentence instead of "every 60 minutes" (accuracy and completeness 0), because the mock picks sentences by word overlap.

### Governance (Task 15)
- **Least autonomy:** `TBD`
- **Risk classification:** `TBD`
- **Runtime budget cap:** `TBD`

## Phase 2 notes
- **MockLLM** ([llm.py](llm.py)) extends `crewai.llms.base_llm.BaseLLM`. It handles both pitfalls from the brief. Tool results are read only from **assistant** messages (CrewAI appends `Observation: <result>` there), never from the system prompt that contains the template line. Tool arguments are chosen from each tool's declared `args_schema` fields (`query` vs `record_id`), never from the tool's name.
- **Network off:** `CREWAI_DISABLE_TELEMETRY=true`, `OTEL_SDK_DISABLED=true` and `CREWAI_TRACING_ENABLED=false` are set in `llm.py` before CrewAI is imported. `CREWAI_TRACING_ENABLED=false` also stops CrewAI's first-run interactive trace prompt. The crew is built with `tracing=False`, ChromaDB uses `anonymized_telemetry=False`, and the E5 model loads with `local_files_only=True` after a one-time download.
- **Memory:** running Task 8 prints a `LangChainDeprecationWarning` for `RunnableWithMessageHistory` / `InMemoryChatMessageHistory`. The brief says this is expected, and the classes work.

## Phase 3 notes
- **API:** `POST /ask`, `POST /add-document` and `WS /chat`, all with Pydantic request/response models. `transcripts/11_fastapi.txt` is produced against a **real `uvicorn` process**. One WebSocket client disconnects before its reply arrives. The handler catches `WebSocketDisconnect` and logs a `client_disconnected` event, and the server keeps serving other clients with zero tracebacks. Crew calls run in a worker thread (`asyncio.to_thread`), so one chat doesn't block others.
- **Logging:** every request writes one JSON line to `logs/requests.jsonl`, with `@timestamp`, `trace_id`, `endpoint`, `status`, `duration_ms` and `session_id`. `request_text` passes through the same `mask_pii` the model input does. `/add-document` also masks text before indexing, so a raw phone number never reaches disk. See `transcripts/12_logging.txt`.

## PII scope
Phone numbers (fixed format) are masked on input, on the model path and in the logs.
Rider/driver names, addresses and payment details are free text with no fixed format, so
they are out of scope for masking. Only fabricated examples are used.

## Repository map
See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the build order.
