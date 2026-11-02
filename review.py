"""Autogen review stage (Task 14): Policy-Compliance-Reviewer -> Final-Editor, RoundRobinGroupChat(max_turns=2)."""
import asyncio
import os
import re

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import ChatCompletionClient, CreateResult, ModelInfo, RequestUsage
from pydantic import BaseModel

import llm
import rag

UNSUPPORTED = "UNSUPPORTED: "


class Verdict(BaseModel):
    approved: bool
    final_answer: str
    reason: str


def _task_parts(text):
    draft, context = re.match(r"DRAFT:\n(.*)\n\nCONTEXT:\n(.*)", text, re.S).groups()
    return draft.strip(), context


class MockReviewClient(ChatCompletionClient):
    """Deterministic Autogen model client. `role` picks reviewer or editor behaviour."""

    def __init__(self, role):
        self.role = role

    async def create(self, messages, *, tools=(), tool_choice="auto", json_output=None, extra_create_args=None,
                     cancellation_token=None):
        texts = [m.content for m in messages if isinstance(getattr(m, "content", None), str)]
        draft, context = _task_parts(next(t for t in texts if t.startswith("DRAFT:")))
        if self.role == "reviewer":
            bad = [s for s in llm._sentences(draft) if not llm._supported(s, context)]
            content = "\n".join(UNSUPPORTED + s for s in bad) or "COMPLIANT: every sentence is supported by the context."
        else:  # editor: acts on the reviewer's findings
            review = next((t for t in reversed(texts) if t.startswith((UNSUPPORTED, "COMPLIANT"))), "")
            bad = [line[len(UNSUPPORTED):] for line in review.splitlines() if line.startswith(UNSUPPORTED)]
            if not bad:
                v = Verdict(approved=True, final_answer=draft, reason="Reviewer found every sentence grounded.")
            else:
                kept = " ".join(s for s in llm._sentences(draft) if s not in bad)
                v = Verdict(approved=False, final_answer=kept or rag.IDK,
                            reason=f"Removed {len(bad)} sentence(s) not supported by the retrieved context: {bad}")
            content = v.model_dump_json()
        return CreateResult(content=content, finish_reason="stop", usage=RequestUsage(prompt_tokens=0,
                            completion_tokens=0), cached=False)

    def create_stream(self, *args, **kwargs):
        raise NotImplementedError("streaming not used")

    async def close(self):
        pass

    def actual_usage(self):
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    total_usage = actual_usage

    def count_tokens(self, messages, *, tools=()):
        return 0

    def remaining_tokens(self, messages, *, tools=()):
        return 10**6

    @property
    def capabilities(self):
        return self.model_info

    @property
    def model_info(self):
        return ModelInfo(vision=False, function_calling=False, json_output=True, family="unknown",
                         structured_output=True)


def _client(role):
    if llm.MOCK:
        return MockReviewClient(role)
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    return OpenAIChatCompletionClient(
        model=llm.GEMINI_MODEL, api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",  # Gemini's OpenAI-compatible endpoint
        model_info=ModelInfo(vision=False, function_calling=True, json_output=True, family="unknown",
                             structured_output=True))


def build_team():
    reviewer = AssistantAgent(
        "Policy_Compliance_Reviewer", _client("reviewer"),
        system_message="You check a support draft against the retrieved context. List every draft sentence the "
                       f"context does not support, one per line prefixed '{UNSUPPORTED}', or reply 'COMPLIANT'.")
    editor = AssistantAgent(
        "Final_Editor", _client("editor"), output_content_type=Verdict,
        system_message="You produce the final answer. If the reviewer found no problems, approve the draft unchanged. "
                       "Otherwise remove the unsupported sentences. Never add new facts.")
    # max_turns=2: reviewer speaks, then editor. custom_message_types is required for the editor's Verdict.
    return RoundRobinGroupChat([reviewer, editor], max_turns=2, custom_message_types=[StructuredMessage[Verdict]])


async def _review(draft, context):
    result = await build_team().run(task=f"DRAFT:\n{draft}\n\nCONTEXT:\n{context}")
    verdict = next(m.content for m in reversed(result.messages) if isinstance(m, StructuredMessage))
    return Verdict.model_validate(verdict.model_dump()), result.messages


def review(draft, context):
    """Returns (Verdict, team messages)."""
    return asyncio.run(_review(draft, context))
