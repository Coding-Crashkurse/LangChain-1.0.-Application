from __future__ import annotations

from langchain.agents import AgentState


class CustomState(AgentState, total=False):
    user_name: str
    user_id: int
    skill_level: str
    thread_id: str
    last_feedback: str | None
    pending_message_id: int | None
    last_answer: str | None
    question: str | None
    needs_retry: bool
    tool_gate: dict | None


GraphState = CustomState
