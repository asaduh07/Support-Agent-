# Implementation Plan

Each phase ends with a runnable demo that writes a transcript to `transcripts/`.
The order follows dependencies, so later phases use earlier ones as fixed inputs.

## Phase 0: Setup
- [x] `.venv` on Python 3.10 (CrewAI requires <3.14)
- [x] git repo on branch `main`, `.gitignore`, `.env.example` (MOCK_LLM, telemetry, GEMINI_API_KEY)
- [ ] Dependencies get installed per phase as needed and pinned into `requirements.txt` as we go
- [x] Download the embedding model once: `intfloat/e5-small-v2` (E5)
- [ ] Optional Gemini path behind `MOCK_LLM=0` (key provided by the user)

## Phase 1: Dataset & RAG core (Part 1, 30 marks)
1. `dataset.py`: seeded generator with ≥40 records and stats printout. Iterate seed/weights until
   every category ≥3, every status ≥1, and escalated is 10–30%. Record the choices in the README. **[T1]**
2. `kb/`: 12 docs, one per required topic, 2–5 sentences each. **[T2]**
3. `rag.py`: two chunkers, embeddings, two Chroma collections (`upsert`), and a sample query on both. **[T3]**
4. Calibrate: top-1 cosine for ≥3 in-scope and ≥2 out-of-scope queries, then pick the threshold.
   Grounded generation on 5 in-scope + 1 out-of-scope query. **[T4]**
5. `evaluate.py` part A: doc-level P/R per query per collection with arithmetic shown,
   then the recommendation. **[T5]**

## Phase 2: Crew, tools, memory, guardrails (Part 2, 30 marks)
6. `tools.py`: `check_support_ticket_status` + escalation formula and data-justified threshold. **[T6]**
7. `llm.py`: CrewAI `MockLLM(BaseLLM)`. Unit-check both pitfalls (template "Observation:",
   name-based dispatch). **The riskiest piece, so it gets built and verified first in this phase.**
8. `crew.py`: Retrieval / Lookup / Composer agents, `.kickoff()` demo with both tools invoked. **[T7]**
9. `CrewResponse` Pydantic schema + validation of every response. **[T9]**
10. `guardrails.py`: phone mask, injection detect, groundedness refuse. One firing demo each. **[T10]**
11. `pipeline.py`: memory with a multi-turn transcript + a fresh-session transcript. **[T8]**

## Phase 3: Deploy, log, evaluate (Part 3, 20 marks)
12. `app.py`: `POST /ask`, `POST /add-document`, `WS /chat` with `WebSocketDisconnect` handling.
    Demo through FastAPI `TestClient`, including a mid-chat disconnect and a second client still working. **[T11]**
13. JSONL logging with trace_id + duration_ms. Demo that the log has a masked phone and no raw one. **[T12]**
14. `evaluate.py` part B: 15 queries (all 12 topics + ≥2 OOS/edge), mock judge, four scores per
    query, plus averages. **[T13]**

## Phase 4: Review & governance (Part 4, 20 marks)
15. `review.py`: Autogen `RoundRobinGroupChat(max_turns=2)`, `Verdict` via `output_content_type`,
    `custom_message_types=[StructuredMessage[Verdict]]`. One approve demo + one revise demo
    (an ungrounded claim injected into the draft). **[T14]**
16. Least-autonomy guard demo, risk classification paragraph, budget-cap rejection demo. **[T15]**
17. Query cache + call-counter before/after demo. **[T16]**

## Phase 5: Wrap-up
- `demos.py` regenerates all transcripts from scratch
- Fill every `TBD` in the README with the real measured numbers
- Clean-run check: fresh venv, no network/API keys, `demos.py` passes
- Push to a public GitHub repo (only when you say so)

## Risks
| Risk | Mitigation |
|---|---|
| CrewAI `BaseLLM` interface differs by version | Pin the crewai version; read its `BaseLLM` source after install |
| Autogen API drift (0.4+ `autogen-agentchat`) | Pin the version; verify `max_turns` / `custom_message_types` exist |
| First model download needs network | One-time download at setup; runtime is offline afterwards |
| Escalated % misses 10–30% | Change the seed/weights only, never edit records |
