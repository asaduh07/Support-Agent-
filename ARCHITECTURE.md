# Architecture

## Request flow

```
client (HTTP /ask | WebSocket /chat)
  │
  ▼
app.py ── trace_id, timer, JSONL log (masked text only)                [T11, T12]
  │
  ▼
pipeline.answer(session_id, query)
  1. guardrails.check_input   → injection? reject | mask phone numbers [T10]
  2. budget.check             → est. tokens > cap? reject               [T15 runtime]
  3. memory (RunnableWithMessageHistory, per session_id)                [T8]
  4. crew.kickoff()  ── CrewAI, MockLLM(BaseLLM)                        [T7]
       ├─ Retrieval Agent  ── tool: rag_lookup   (cache → Chroma)       [T3-5, T16]
       ├─ Lookup Agent     ── tool: check_support_ticket_status         [T6, T15 app]
       └─ Composer Agent   ── output_pydantic=CrewResponse              [T9]
  5. guardrails.check_grounded → top-1 sim < threshold? "I don't know"  [T10, T4]
  6. review.run()  ── Autogen RoundRobinGroupChat(max_turns=2)          [T14]
       Policy-Compliance-Reviewer → Final-Editor (output_content_type=Verdict)
  7. return final answer (validated Pydantic)
```

## Modules (flat layout, one file per concern)

| File | Responsibility | Tasks |
|---|---|---|
| `dataset.py` | Seeded generator → `SUPPORT_TICKETS`; prints stats | 1 |
| `kb/*.md` | 12+ policy documents, one per required topic | 2 |
| `rag.py` | Fixed-size+overlap and sentence chunkers, E5 embeddings (`intfloat/e5-small-v2`, `query:`/`passage:` prefixes), two Chroma collections, retrieve, grounded generate, threshold, query cache | 3, 4, 16 |
| `mock_llm.py` | Deterministic `MockLLM(BaseLLM)` for CrewAI, mock Autogen `ChatCompletionClient`, mock judge | all LLM calls |
| `tools.py` | `rag_lookup`, `check_support_ticket_status` + `escalation_score` | 6 |
| `crew.py` | 3 agents, tasks, `CrewResponse` schema, least-autonomy guard | 7, 9, 15 |
| `guardrails.py` | Phone masking, injection detection, groundedness check | 10, 12 |
| `review.py` | Autogen 2-agent review team, `Verdict` model | 14 |
| `pipeline.py` | Glues it together: memory, budget cap, guardrails, crew, review | 8, 15 |
| `app.py` | FastAPI: `POST /ask`, `POST /add-document`, `WS /chat`; JSONL logging | 11, 12 |
| `evaluate.py` | Chunking P/R (T5) and 15-query LLM-as-judge eval (T13) | 5, 13 |
| `run_demos.py` | Runs each task demo, writes `transcripts/NN_*.txt` | all |

## Key design decisions

**LLM switch.** `MOCK_LLM=1` (the default, used for grading) uses deterministic mocks everywhere. With `MOCK_LLM=0`
+ `GEMINI_API_KEY`, the system uses Gemini instead: CrewAI via its built-in `LLM(model="gemini/...")`, and Autogen via
`autogen-ext`'s OpenAI-compatible client pointed at Gemini's endpoint. One flag is read in `mock_llm.py`.

**MockLLM (CrewAI).** Subclass `crewai.llms.base_llm.BaseLLM` and implement `call()`. It reads
only the *last* user/assistant messages, never the system prompt, so the template text
`"Observation: the result of the action"` can't be mistaken for a real tool result. Tool
arguments are chosen from the tool's declared `args_schema` field names (`query` vs `record_id`),
never from the tool's name. That way `rag_lookup` is not misread as the lookup tool.
Behaviour: first call emits `Action: <tool>\nAction Input: {...}`; once a real `Observation:`
appears in the model-generated turn, it emits `Final Answer: <observation-derived text>`.

**MockLLM (Autogen).** A custom `ChatCompletionClient` that returns deterministic text. For
the Final-Editor it returns `Verdict` JSON. The reviewer flags any draft sentence whose words
don't appear in the retrieved context, and the editor removes those sentences (revise) or approves the draft unchanged.

**Grounded generation under mock.** Extractive: the answer is built from the top retrieved
chunk sentences. If the top-1 cosine is below the calibrated threshold, it returns "I don't know".

**Escalation score.** `w·escalated + (1−w)·(days_since_created/30)`. Older tickets score higher, because they are closer to breaching the SLA. `w` and the
threshold chosen from the generated data's distribution. See README.

**Least autonomy.** `check_support_ticket_status` is only attached to the Lookup Agent. On top
of that, a guard in `crew.py` asserts at crew-build time that no other agent holds the tool, and
the tool itself refuses calls whose caller isn't the Lookup Agent. Demo: wiring it to the
Composer raises.

**Budget cap.** Token estimate is `len(text)/4` before kickoff. Requests over the cap are rejected with
an explicit error before any LLM or tool call.

**Cache.** A `dict` keyed by normalized query (lowercase, collapsed whitespace, stripped
punctuation) in front of grounded generation. A call counter proves the hit.

**Memory.** `InMemoryChatMessageHistory` per `session_id` + `RunnableWithMessageHistory`
wrapping the pipeline. It lives in-process only, as the brief allows.

**Logging.** Stdlib `logging` with a JSON formatter writes to `logs/requests.jsonl`. Text is masked before it is logged.

## Risk classification
**Medium**, since this is a customer-support ticket system. Final justification goes in the README.
