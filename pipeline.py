"""End-to-end request path: input guardrails -> budget cap -> session memory -> crew -> groundedness check."""
import math
import re

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel

import guardrails
from crew import CrewResponse, run_crew
from llm import RECORD_ID

# Runtime-layer budget (Task 15): ~4 characters per token, checked BEFORE any LLM or tool call.
MAX_INPUT_TOKENS = 200
# A ticket follow-up: "should it be escalated?", "is it still open?", "what about this ticket?"
FOLLOW_UP = re.compile(r"\b(?:it|its)\b.*\b(?:escalat\w*|status|old|open|resolved|closed)\b"
                       r"|\b(?:this|that|the|my) ticket\b", re.I)


class BudgetExceeded(Exception):
    pass


class Result(BaseModel):
    session_id: str
    masked_query: str
    answer: str
    response: CrewResponse | None = None
    guardrail: str | None = None  # which guardrail fired, if any
    resolved_query: str | None = None  # query after memory-based reference resolution


def estimate_tokens(text):
    return math.ceil(len(text) / 4)


def check_budget(text):
    n = estimate_tokens(text)
    if n > MAX_INPUT_TOKENS:
        raise BudgetExceeded(f"Request needs ~{n} tokens, over the per-request cap of {MAX_INPUT_TOKENS}. Rejected "
                             "before any LLM or tool call.")
    return n


# ---------------------------------------------------------------- session memory (Task 8)
_SESSIONS = {}


def session_history(session_id):
    return _SESSIONS.setdefault(session_id, InMemoryChatMessageHistory())


def resolve_reference(query, history):
    """A follow-up like "should it be escalated?" inherits the last ticket ID mentioned in this session."""
    if RECORD_ID.search(query) or not FOLLOW_UP.search(query):
        return query
    for msg in reversed(history):
        if m := RECORD_ID.search(msg.content):
            return f"{query} (ticket {m.group().upper()})"
    return query


CLARIFY = "Which ticket do you mean? Please share the ticket ID (format OLA-0000)."


def _turn(inp):
    query = resolve_reference(inp["question"], inp["history"])
    if FOLLOW_UP.search(query) and not RECORD_ID.search(query):  # a follow-up, but no ticket in this session
        return {"answer": CLARIFY, "response": None, "fired": False, "resolved_query": query}
    resp, fired = guardrails.check_grounded(run_crew(query))
    return {"answer": resp.answer, "response": resp, "fired": fired, "resolved_query": query}


chat = RunnableWithMessageHistory(RunnableLambda(_turn), session_history, input_messages_key="question",
                                  history_messages_key="history", output_messages_key="answer")


def answer(query, session_id="default"):
    masked, _ = guardrails.mask_pii(query)  # the model, memory and logs only ever see the masked text
    if hit := guardrails.detect_injection(masked):
        return Result(session_id=session_id, masked_query=masked, answer=guardrails.INJECTION_REFUSAL,
                      guardrail=f"prompt_injection: {hit!r}")
    check_budget(masked)
    out = chat.invoke({"question": masked}, config={"configurable": {"session_id": session_id}})
    return Result(session_id=session_id, masked_query=masked, answer=out["answer"], response=out["response"],
                  guardrail="groundedness" if out["fired"] else None, resolved_query=out["resolved_query"])
