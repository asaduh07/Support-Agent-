"""FastAPI deployment (Task 11) with JSON-Lines structured request logging (Task 12).
Run: uvicorn app:app"""
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

import guardrails
import pipeline
import rag
from crew import TicketInfo
from review import Verdict

LOG_PATH = Path(__file__).parent / "logs" / "requests.jsonl"
LOG_PATH.parent.mkdir(exist_ok=True)


class _JsonLines(logging.Formatter):
    def format(self, record):
        return json.dumps(record.payload, ensure_ascii=False)


log = logging.getLogger("ola.requests")
log.setLevel(logging.INFO)
log.propagate = False
if not log.handlers:
    _h = logging.FileHandler(LOG_PATH, encoding="utf-8")
    _h.setFormatter(_JsonLines())
    log.addHandler(_h)


def log_request(trace_id, endpoint, text, started, status, **extra):
    """One ELK-style JSON line per request. The text is PII-masked exactly as the model sees it."""
    log.info("request", extra={"payload": {
        "@timestamp": datetime.now(timezone.utc).isoformat(), "level": "INFO", "service": "ola-support-agent",
        "trace_id": trace_id, "endpoint": endpoint, "request_text": guardrails.mask_pii(text)[0],
        "status": status, "duration_ms": round((time.perf_counter() - started) * 1000, 1), **extra,
    }})


app = FastAPI(title="Ola Support Agent")


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(default="default", max_length=64)


class AskResponse(BaseModel):
    trace_id: str
    answer: str
    grounded: bool
    sources: list[str]
    ticket: TicketInfo | None
    guardrail: str | None
    review: Verdict | None


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(pattern=r"^[a-z0-9_]{3,64}$")
    text: str = Field(min_length=20, max_length=5000)


class AddDocumentResponse(BaseModel):
    trace_id: str
    doc_id: str
    chunks_indexed: dict[str, int]


def _ask(query, session_id, trace_id, endpoint, started):
    try:
        r = pipeline.answer(query, session_id)
    except pipeline.BudgetExceeded as e:
        log_request(trace_id, endpoint, query, started, 413, session_id=session_id, guardrail="budget_cap")
        raise HTTPException(status_code=413, detail=str(e))
    resp = r.response
    log_request(trace_id, endpoint, query, started, 200, session_id=session_id, guardrail=r.guardrail,
                grounded=bool(resp and resp.grounded))
    return AskResponse(trace_id=trace_id, answer=r.answer, grounded=bool(resp and resp.grounded),
                       sources=resp.sources if resp else [], ticket=resp.ticket if resp else None,
                       guardrail=r.guardrail, review=r.review)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    return _ask(req.query, req.session_id, uuid.uuid4().hex, "POST /ask", time.perf_counter())


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document(req: AddDocumentRequest):
    started, trace_id = time.perf_counter(), uuid.uuid4().hex
    text = guardrails.mask_pii(req.text)[0]  # never index (persist) a raw phone number either
    counts = rag.add_document(req.doc_id, text)
    log_request(trace_id, "POST /add-document", text, started, 200, doc_id=req.doc_id)
    return AddDocumentResponse(trace_id=trace_id, doc_id=req.doc_id, chunks_indexed=counts)


@app.websocket("/chat")
async def chat(ws: WebSocket):
    """Multi-turn chat: one memory session per connection. A client vanishing mid-conversation only ends
    this handler; the server keeps serving other clients."""
    await ws.accept()
    session_id = f"ws-{uuid.uuid4().hex[:8]}"
    try:
        while True:
            query = await ws.receive_text()
            started, trace_id = time.perf_counter(), uuid.uuid4().hex
            try:
                # crew runs in a worker thread so one slow chat never blocks other clients
                out = (await asyncio.to_thread(_ask, query, session_id, trace_id, "WS /chat", started)).model_dump()
            except HTTPException as e:
                out = {"trace_id": trace_id, "error": e.detail}
            await ws.send_json(out)
    except WebSocketDisconnect:
        log_request(uuid.uuid4().hex, "WS /chat", "", time.perf_counter(), 499, session_id=session_id,
                    event="client_disconnected")
