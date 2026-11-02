"""Single LLM switch. MOCK_LLM=1 (default, graded) is deterministic, needs no keys or network.
MOCK_LLM=0 uses Google Gemini via GEMINI_API_KEY."""
import json
import os
import re

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")  # no outbound telemetry on crew.kickoff()
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")  # no trace upload, no interactive consent prompt

from crewai.llms.base_llm import BaseLLM  # noqa: E402  (must come after the telemetry env vars)

MOCK = os.getenv("MOCK_LLM", "1") != "0"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CALLS = {"generate_grounded": 0, "crew_llm": 0}  # call counters, used as cache evidence (Task 16)
RECORD_ID = re.compile(r"\bOLA-\d{4}\b", re.I)

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("a an the is are was were be to of in on for and or what how do does i my me can when if it with by at "
             "ola ticket tickets status support customer please".split())  # last line: too generic in this domain


def _words(text):
    # ponytail: naive plural stemming (refunds -> refund); swap for a real stemmer if recall suffers
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _WORD.findall(text.lower()) if w not in _STOP}


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
        return " ".join(s for s in sents if s in best)  # "" = nothing in the context answers this
    from google import genai  # ponytail: lazy import, only needed for the optional real-LLM path
    prompt = ("Answer the question using ONLY the context below. If the context does not contain the answer, "
              "say \"I don't know\".\n\nContext:\n" + "\n".join(context_chunks) + f"\n\nQuestion: {query}")
    return genai.Client().models.generate_content(model=GEMINI_MODEL, contents=prompt).text.strip()


# ---------------------------------------------------------------- CrewAI

def _json_objects(text):
    """Every top-level JSON object embedded in free text."""
    dec, out, i = json.JSONDecoder(), [], 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = dec.raw_decode(text, i)
            out.append(obj)
            i = end
        except ValueError:
            i += 1
    return out


def compose(context, schema):
    """Mock Composer: merge the Retrieval and Lookup agents' JSON outputs into one `schema` instance."""
    objs = _json_objects(context)
    rag_out = next((o for o in objs if "grounded" in o), {"answer": "", "grounded": False, "sources": []})
    ticket = next((o for o in objs if "found" in o), None)
    parts = [rag_out["answer"]] if rag_out["grounded"] else []
    if ticket and ticket["found"]:
        parts.append(
            f"Ticket {ticket['record_id']} ({ticket['category']}) is {ticket['status']}, created "
            f"{ticket['days_since_created']} days ago, resolution time {ticket['resolution_time_hours']} hours, "
            f"escalation score {ticket['escalation_score']}: "
            + ("escalation recommended." if ticket["recommend_escalation"] else "no escalation needed."))
    elif ticket and ticket.get("record_id"):
        parts.append(f"I could not find a ticket with ID {ticket['record_id']}.")
    return schema(
        answer=" ".join(parts) or rag_out["answer"],
        grounded=bool(rag_out["grounded"] or (ticket and ticket["found"])),
        sources=rag_out.get("sources", []),
        ticket=ticket if ticket and ticket["found"] else None,
    )


class MockLLM(BaseLLM):
    """Deterministic ReAct LLM for CrewAI (BaseLLM is CrewAI's extension point for non-litellm models).

    Pitfall 1: CrewAI's system prompt contains the template line "Observation: the result of the action".
    We only read tool results from ASSISTANT messages (the model's own generated turns, to which CrewAI appends
    the real "Observation: ..."), never from the system prompt.
    Pitfall 2: tool arguments are chosen from each tool's declared args_schema fields, never from its name,
    so a tool called "rag_lookup" can't be mistaken for the ticket lookup tool.
    """

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None):
        CALLS["crew_llm"] += 1
        msgs = [{"role": "user", "content": messages}] if isinstance(messages, str) else messages
        user = "\n".join(m["content"] for m in msgs if m["role"] == "user")
        own = [m["content"] for m in msgs if m["role"] == "assistant"]

        if response_model is not None:  # Composer: structured output
            context = user.split("context you're working with:", 1)[-1]
            return "Thought: I now know the final answer\nFinal Answer: " + compose(context, response_model).model_dump_json()

        if own and "Observation:" in own[-1]:  # a real tool result, in the model's own turn
            return "Thought: I now know the final answer\nFinal Answer: " + own[-1].split("Observation:", 1)[1].strip()

        query = (re.findall(r"QUERY: (.*)", user) or [""])[-1].strip()
        for t in getattr(from_agent, "tools", None) or []:
            fields = t.args_schema.model_fields
            if "record_id" in fields:
                m = RECORD_ID.search(query)
                if not m:
                    return 'Thought: The query has no ticket ID.\nFinal Answer: {"found": false, "record_id": null}'
                args = {"record_id": m.group().upper()}
            elif "query" in fields:
                args = {"query": query}
            else:
                continue
            return f"Thought: I should use a tool.\nAction: {t.name}\nAction Input: {json.dumps(args)}"
        return "Thought: No tool applies.\nFinal Answer: {}"


def crew_llm():
    if MOCK:
        return MockLLM(model="mock-llm")
    from crewai import LLM
    return LLM(model=f"gemini/{GEMINI_MODEL}", api_key=os.getenv("GEMINI_API_KEY"), temperature=0)
