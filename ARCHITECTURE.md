# Architecture

## Request flow

```
client (HTTP /ask | WebSocket /chat)
  │
  ▼
app.py ── trace_id, timer, JSONL log (masked text only)                [T11, T12]
  │
  ▼
pipeline.answer(query, session_id)
  1. guardrails.mask_pii + detect_injection → mask phones | refuse      [T10]
  2. pipeline.check_budget    → est. tokens > cap? reject (HTTP 413)    [T15 runtime]
  3. memory (RunnableWithMessageHistory, per session_id); follow-ups
     like "should it be escalated?" inherit the session's last ticket ID [T8]
  4. crew.kickoff()  ── CrewAI, MockLLM(BaseLLM)                        [T7]
       ├─ Retrieval Agent  ── tool: rag_lookup   (cache → Chroma)       [T3-5, T16]
       ├─ Lookup Agent     ── tool: check_support_ticket_status         [T6, T15 app]
       └─ Composer Agent   ── output_pydantic=CrewResponse              [T9]
  5. guardrails.check_grounded → no KB support and no ticket? refuse    [T10, T4]
  6. review.review() ── Autogen RoundRobinGroupChat(max_turns=2)        [T14]
       Policy-Compliance-Reviewer → Final-Editor (output_content_type=Verdict)
  7. return final answer (validated Pydantic)
```

## Modules (flat layout, one file per concern)

| File | Responsibility | Tasks |
|---|---|---|
| `dataset.py` | Seeded generator → `SUPPORT_TICKETS`; prints and validates stats | 1 |
| `kb/*.md` | 12 policy documents, one per required topic | 2 |
| `rag.py` | Fixed-size+overlap and sentence chunkers, E5 embeddings (`intfloat/e5-small-v2`, `query:`/`passage:` prefixes), two Chroma collections, retrieve, grounded generation, calibrated threshold, query cache, add-document | 3, 4, 16 |
| `llm.py` | MOCK_LLM switch, extractive grounded generation, `MockLLM(BaseLLM)` for CrewAI, LLM-as-judge prompt + mock rubric, Gemini paths | all LLM calls |
| `tools.py` | `rag_lookup`, `check_support_ticket_status` + `escalation_score` | 6 |
| `crew.py` | 3 agents, tasks, `CrewResponse` schema, least-autonomy guard | 7, 9, 15 |
| `guardrails.py` | Phone masking, injection detection, groundedness check | 10, 12 |
| `review.py` | Autogen 2-agent review team, `Verdict` model, `MockReviewClient(ChatCompletionClient)` | 14 |
| `pipeline.py` | Glues it together: memory, budget cap, guardrails, crew, review | 8, 15 |
| `app.py` | FastAPI: `POST /ask`, `POST /add-document`, `WS /chat`; JSONL logging | 11, 12 |
| `evaluate.py` | Chunking P/R (T5) and 15-query LLM-as-judge eval (T13) | 5, 13 |
| `demos.py` | Runs each task demo, writes `transcripts/NN_*.txt` | all |

## Key design decisions

**LLM switch.** `MOCK_LLM=1` (the default, used for grading) uses deterministic mocks everywhere. With `MOCK_LLM=0`
+ `GEMINI_API_KEY`, the system uses Gemini instead: CrewAI via its built-in `LLM(model="gemini/...")`, Autogen via
`autogen-ext`'s OpenAI client, and grounded generation and the judge via `google-genai`. One flag is read in `llm.py`.

**MockLLM (CrewAI).** Subclass `crewai.llms.base_llm.BaseLLM` and implement `call()`. CrewAI appends the real
`Observation: <result>` to the **assistant** message, so tool results are read only from assistant messages, never
from the system prompt. That way the template text `"Observation: the result of the action"` can't be mistaken for a
real tool result. Tool arguments are chosen from the tool's declared `args_schema` field names (`query` vs
`record_id`), never from the tool's name. That way `rag_lookup` is not misread as the lookup tool.
Behaviour: the first call emits `Action: <tool>` / `Action Input: {...}`; once a real `Observation:` appears in the
model-generated turn, it emits `Final Answer: <observation>`. When CrewAI passes a `response_model` (the Composer's
`output_pydantic`), it merges the two agents' JSON outputs into a `CrewResponse`.

**MockReviewClient (Autogen).** A custom `ChatCompletionClient`. The reviewer flags any draft sentence whose content
words are less than 60% present in the retrieved context. The editor reads those flags and returns `Verdict` JSON: it
approves the draft unchanged, or removes the flagged sentences.

**Grounded generation under mock.** Extractive: it picks up to 2 retrieved full sentences with the most query-word
overlap (domain stopwords removed, naive plural stemming). If the top-1 cosine is below the calibrated threshold (0.83),
or no retrieved sentence shares a content word with the query, it returns "I don't know".

**Escalation score.** `0.5·escalated + 0.5·(min(days_since_created, 30)/30)`. Older tickets score higher, because
they are closer to breaching the SLA. The threshold of 0.75 is the escalated flag plus the median ticket age
(15 days). See README.

**Least autonomy.** `check_support_ticket_status` is only attached to the Lookup Agent. On top of that,
`crew.assert_least_autonomy()` runs inside `build_crew()` and raises `PermissionError` if any other agent holds the
tool. Demo: wiring it to the Composer raises.

**Budget cap.** The token estimate is `ceil(len(text)/4)`, checked before kickoff. Requests over 200 tokens are
rejected with `BudgetExceeded` (HTTP 413) before any LLM or tool call.

**Cache.** A `dict` keyed by normalized query (lowercase, collapsed whitespace, stripped punctuation) in front of
grounded generation. `/add-document` clears it. A call counter proves the hit.

**Memory.** `InMemoryChatMessageHistory` per `session_id` + `RunnableWithMessageHistory` wrapping the crew turn. It
lives in-process only, as the brief allows.

**Logging.** Stdlib `logging` with a JSON-Lines formatter writes to `logs/requests.jsonl`. Text is masked with the
same `mask_pii` the model input uses before it is logged or indexed.

**Network off.** Telemetry and tracing env vars are set before CrewAI is imported, the crew uses `tracing=False`,
Chroma has `anonymized_telemetry=False`, and E5 loads with `local_files_only=True`.

## Risk classification
**Medium** (customer-support tickets). The full justification is in the README.
