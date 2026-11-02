"""Agent tools: RAG lookup and ticket status with a designed escalation score (Task 6)."""
import json

from crewai.tools import tool

import dataset
import rag

# escalation_score = W * escalated + (1 - W) * min(days_since_created, 30) / 30   -> always in [0, 1]
# Older tickets score higher: they are closer to (or past) their SLA.
ESCALATION_WEIGHT = 0.5
# 0.75 = escalated flag (0.5) + an age of >= 15 days (0.25). 15 days is the median days_since_created of the
# generated dataset, so this flags escalated tickets that are older than half of all tickets (8/50 = 16%).
# A non-escalated ticket maxes out at 0.5, so recency alone never triggers escalation.
ESCALATION_THRESHOLD = 0.75
LOOKUP_TOOL_NAME = "check_support_ticket_status"

TOOL_CALLS = {"rag_lookup": 0, LOOKUP_TOOL_NAME: 0}  # invocation evidence for demos
_BY_ID = {t["record_id"]: t for t in dataset.SUPPORT_TICKETS}


def escalation_score(escalated, days_since_created):
    w = ESCALATION_WEIGHT
    return round(w * bool(escalated) + (1 - w) * min(max(days_since_created, 0), 30) / 30, 3)


def check_support_ticket_status(record_id: str) -> dict:
    rid = record_id.strip().upper()
    t = _BY_ID.get(rid)
    if t is None:
        return {"record_id": rid, "found": False}
    score = escalation_score(t["escalated"], t["days_since_created"])
    return {
        "record_id": rid, "found": True, "status": t["status"], "category": t["category"],
        "resolution_time_hours": t["resolution_time_hours"], "days_since_created": t["days_since_created"],
        "escalated": t["escalated"], "escalation_score": score, "recommend_escalation": score >= ESCALATION_THRESHOLD,
    }


@tool("rag_lookup")
def rag_lookup_tool(query: str) -> str:
    """Search Ola's support-policy knowledge base and return a grounded answer with its sources as JSON."""
    TOOL_CALLS["rag_lookup"] += 1
    r = rag.grounded_answer(query)
    return json.dumps({k: r[k] for k in ("answer", "grounded", "top_score", "sources")})


@tool(LOOKUP_TOOL_NAME)
def ticket_status_tool(record_id: str) -> str:
    """Look up an Ola support ticket by its ID (format OLA-0000) and return its status and escalation score as JSON."""
    TOOL_CALLS[LOOKUP_TOOL_NAME] += 1
    return json.dumps(check_support_ticket_status(record_id))


if __name__ == "__main__":
    assert escalation_score(True, 30) == 1.0 and escalation_score(False, 0) == 0.0
    assert escalation_score(True, 15) == 0.75 and escalation_score(False, 30) == 0.5
    assert check_support_ticket_status("ola-0001")["found"] and not check_support_ticket_status("OLA-9999")["found"]
    print("tools self-check ok")
