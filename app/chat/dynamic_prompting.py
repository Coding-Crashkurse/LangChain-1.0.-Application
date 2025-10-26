from __future__ import annotations

from dataclasses import dataclass

from langchain.agents.middleware import ModelRequest, dynamic_prompt


@dataclass
class Context:
    user_role: str = "beginner"
    user_name: str = ""
    user_id: int | None = None
    last_feedback: str | None = None
    current_tool_log_id: int | None = None
    thread_id: str | None = None


def _prompt_for_skill(role: str, user_name: str, feedback: str | None = None) -> str:
    base = "You are a Python explainer bot for {user}.".format(user=user_name or "the user")
    if role == "expert":
        style = (
            "You are a deeply technical assistant."
            " Use precise terminology, highlight best practices, and rely on advanced examples."
        )
    elif role == "mid":
        style = "Explain precisely with code samples and brief comparisons to alternatives."
    else:
        style = (
            "Explain concepts in very simple language, use short sentences with plenty of examples,"
            " and avoid jargon unless you define it."
        )

    guidance = f" Incorporate the latest feedback: {feedback}." if feedback else ""
    return f"{base} {style}{guidance}".strip()


@dynamic_prompt
def adjust_prompt(request: ModelRequest) -> str:
    context: Context = request.runtime.context or Context()
    return _prompt_for_skill(context.user_role, context.user_name, context.last_feedback)


def preview_prompt(role: str, feedback: str | None = None) -> str:
    return _prompt_for_skill(role, "Test", feedback)


adjust_prompt.preview_prompt = preview_prompt  # type: ignore[attr-defined]
