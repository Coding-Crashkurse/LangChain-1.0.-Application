from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from flask import current_app
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from app.extensions import db
from app.models import ChatMessage, User

from .dynamic_prompting import Context, adjust_prompt
from .schemas import Weather
from .state import GraphState
from .tools import ASSISTANT_TOOLS, weather_tool

__all__ = [
    "build_langchain_history",
    "run_chat_turn",
    "resume_chat_turn",
    "weather_demo",
]


def build_langchain_history(messages: Iterable[ChatMessage]) -> list[AnyMessage]:
    """Convert stored chat messages to LangChain message objects."""

    ordered = sorted(messages, key=lambda msg: msg.created_at)
    history: list[AnyMessage] = []
    for item in ordered:
        if item.role == "assistant" and item.approved is not True:
            continue
        if item.role == "assistant":
            history.append(AIMessage(content=item.content))
        else:
            history.append(HumanMessage(content=item.content))
    return history


def run_chat_turn(
    *,
    user: User,
    thread_id: str,
    history: list[AnyMessage],
    question: str,
    last_feedback: str | None = None,
) -> dict[str, Any]:
    """Start a new LLM turn and return the interrupt payload."""

    manager = _get_chat_manager()
    state: GraphState = {
        "messages": list(history),
        "user_name": user.username,
        "user_id": user.id,
        "skill_level": user.skill_level,
        "thread_id": thread_id,
        "last_feedback": last_feedback,
        "question": question,
        "pending_message_id": None,
        "needs_retry": False,
    }
    context = Context(
        user_role=user.skill_level,
        user_name=user.username,
        user_id=user.id,
        last_feedback=last_feedback,
        thread_id=thread_id,
    )
    return manager.run(state=state, context=context, thread_id=thread_id)


def resume_chat_turn(
    *, user: User, thread_id: str, payload: dict[str, Any], last_feedback: str | None = None
) -> dict[str, Any]:
    """Resume the graph using the provided payload."""

    manager = _get_chat_manager()
    context = Context(
        user_role=user.skill_level,
        user_name=user.username,
        user_id=user.id,
        last_feedback=last_feedback,
        thread_id=thread_id,
    )
    return manager.resume(thread_id=thread_id, payload=payload, context=context)


def weather_demo(city: str) -> Weather:
    """Run the structured weather agent and return validated data."""

    agent = _get_weather_agent()
    context = Context(user_role="mid", user_name="Demo", user_id=0)
    result = agent.invoke(
        {"messages": [HumanMessage(content=f"Wie ist das Wetter in {city}?")]},
        context=context,
    )
    structured = result.get("structured_response")
    return Weather.model_validate(structured)


class LocalSkillModel(BaseChatModel):
    """Deterministic chat model used for tests and offline mode."""

    model_name: ClassVar[str] = "local-skill-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        question = self._extract_question(messages)
        role = self._detect_role(messages)
        answer = self._render_answer(question, role)
        tool_calls = self._planned_tool_calls(question)
        ai_message = AIMessage(content=answer, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    async def _agenerate(self, *args: Any, **kwargs: Any) -> ChatResult:  # pragma: no cover
        return self._generate(*args, **kwargs)

    @property
    def _llm_type(self) -> str:
        return self.model_name

    def bind_tools(self, tools, *, tool_choice: str | None = None, **kwargs: Any):
        return self

    @staticmethod
    def _extract_question(messages: list[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return str(message.content)
        return "Python"

    @staticmethod
    def _detect_role(messages: list[BaseMessage]) -> str:
        for message in messages:
            if isinstance(message, SystemMessage):
                text = message.content.lower()
                if 'deeply technical' in text or 'advanced examples' in text:
                    return 'expert'
                if 'explain precisely' in text and 'code' in text:
                    return 'mid'
        return 'beginner'

    @staticmethod
    def _render_answer(question: str, role: str) -> str:
        if role == 'expert':
            return (
                f"Technical deep-dive on '{question}': use type hints, add pytest suites,"
                " and reason about tricky edge cases."
            )
        if role == 'mid':
            return (
                f"Explanation for '{question}': include list-comprehension examples"
                " and compare alternative approaches."
            )
        return (
            f"Simple explanation for '{question}': start with tiny print examples and"
            " describe each step in plain language."
        )

    def _planned_tool_calls(self, question: str) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        lowered = question.lower()
        if 'weather' in lowered:
            city = question.split()[-1].strip('?.!,')
            calls.append(
                {
                    'id': f"tool-{uuid4().hex}",
                    'name': weather_tool.name,
                    'args': {'city': city or 'Berlin'},
                }
            )
        return calls


class WeatherLLM(BaseChatModel):
    model_name: ClassVar[str] = 'weather-structured-llm'

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        city = 'Unknown'
        for message in messages:
            if isinstance(message, HumanMessage):
                city = str(message.content).split()[-1].strip('?!.,')
        tool_call = {
            'id': f"weather-{uuid4().hex}",
            'name': weather_tool.name,
            'args': {'city': city},
        }
        ai_message = AIMessage(content='Weather data incoming.', tool_calls=[tool_call])
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    async def _agenerate(self, *args: Any, **kwargs: Any) -> ChatResult:  # pragma: no cover
        return self._generate(*args, **kwargs)

    @property
    def _llm_type(self) -> str:
        return self.model_name

    def bind_tools(self, tools, *, tool_choice: str | None = None, **kwargs: Any):
        return self

class ChatGraphManager:
    def __init__(self) -> None:
        self._graph = None
        self._agent = None
        self._checkpointer = None
        self._conn = None
        self._weather_agent = None

    def run(self, *, state: GraphState, context: Context, thread_id: str) -> dict[str, Any]:
        graph = self._ensure_graph()
        config = {"configurable": {"thread_id": thread_id}}
        stream = graph.stream(state, config=config, context=context)
        last: dict[str, Any] | None = None
        try:
            for event in stream:
                if "__interrupt__" in event:
                    interrupt_obj = event["__interrupt__"][0]
                    return {"status": "interrupt", "payload": interrupt_obj.value}
                last = event
        finally:
            stream.close()
        return {"status": "complete", "payload": last}

    def resume(
        self, *, thread_id: str, payload: dict[str, Any], context: Context
    ) -> dict[str, Any]:
        graph = self._ensure_graph()
        config = {"configurable": {"thread_id": thread_id}}
        stream = graph.stream(Command(resume=payload), config=config, context=context)
        last: dict[str, Any] | None = None
        try:
            for event in stream:
                if "__interrupt__" in event:
                    interrupt_obj = event["__interrupt__"][0]
                    return {"status": "interrupt", "payload": interrupt_obj.value}
                last = event
        finally:
            stream.close()
        return {"status": "complete", "payload": last}

    def _ensure_graph(self):
        if self._graph is not None:
            return self._graph
        agent = self._ensure_agent()
        builder = StateGraph(state_schema=GraphState, context_schema=Context)
        builder.add_node("llm_turn", lambda state, runtime: self._llm_turn(agent, state, runtime))
        builder.add_node("approval_gate", self._approval_gate)
        builder.add_edge(START, "llm_turn")
        builder.add_edge("llm_turn", "approval_gate")
        builder.add_conditional_edges(
            "approval_gate",
            lambda state: "retry" if state.get("needs_retry") else "approved",
            {"retry": "llm_turn", "approved": END},
        )
        self._graph = builder.compile(checkpointer=self._ensure_checkpointer())
        return self._graph

    def _ensure_agent(self):
        if self._agent is not None:
            return self._agent
        model = self._build_model()
        self._agent = create_agent(
            model=model,
            tools=ASSISTANT_TOOLS,
            middleware=[adjust_prompt],
            state_schema=GraphState,
            context_schema=Context,
        )
        return self._agent

    def _build_model(self) -> BaseChatModel:
        if current_app.config.get("USE_FAKE_LLM") or current_app.config.get("TESTING"):
            return LocalSkillModel()
        model_name = current_app.config.get("LANGCHAIN_DEFAULT_MODEL")
        return init_chat_model(model=model_name)

    def _ensure_checkpointer(self) -> SqliteSaver:
        if self._checkpointer is not None:
            return self._checkpointer
        path = Path(current_app.config.get("GRAPH_DB_PATH", "graph_state.db"))
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        self._conn = conn
        self._checkpointer = SqliteSaver(conn)
        return self._checkpointer

    def reset_thread(self, thread_id: str) -> None:
        if not thread_id:
            return
        checkpointer = self._ensure_checkpointer()
        if hasattr(checkpointer, "delete_thread"):
            checkpointer.delete_thread(thread_id)

    def _llm_turn(
        self, agent, state: GraphState, runtime: Runtime[Context]
    ) -> dict[str, Any]:
        context = runtime.context or Context()
        context.last_feedback = state.get("last_feedback")
        self._log_prompt_messages(state)
        response = agent.invoke(
            state,
            config={
                "configurable": {"thread_id": state["thread_id"]},
                "context": context,
            },
        )
        messages = response["messages"]
        last_ai = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
        answer = self._message_as_text(last_ai) if last_ai else ""
        self._log_model_answer(state["thread_id"], answer)
        return {
            "messages": messages,
            "last_answer": answer,
            "needs_retry": False,
            "pending_message_id": None,
        }

    def _approval_gate(self, state: GraphState, runtime: Runtime[Context]) -> dict[str, Any]:
        answer = state.get("last_answer") or ""
        pending_id = state.get("pending_message_id")
        if not pending_id:
            pending_id = self._ensure_pending_message_row(state, answer)
        else:
            self._cleanup_extra_pending(state, pending_id)
        payload = {
            "kind": "message_review",
            "message_id": pending_id,
            "answer": answer,
            "question": state.get("question"),
            "thread_id": state["thread_id"],
        }
        decision = interrupt(payload)
        approved = bool(decision.get("approved"))
        feedback = decision.get("feedback")
        target_id = decision.get("message_id") or pending_id
        chat_message = ChatMessage.query.get(target_id)
        if chat_message:
            chat_message.approved = approved
            chat_message.rejection_reason = None if approved else feedback
            db.session.commit()
        state["last_feedback"] = None
        if approved:
            return {"pending_message_id": None, "needs_retry": False}
        for idx in range(len(state["messages"]) - 1, -1, -1):
            if isinstance(state["messages"][idx], AIMessage):
                state["messages"].pop(idx)
                break
        state["last_feedback"] = feedback
        return {"pending_message_id": None, "needs_retry": True, "last_feedback": feedback}

    @staticmethod
    def _message_as_text(message: AIMessage | None) -> str:
        if not message:
            return ""
        return ChatGraphManager._plain_text_content(message)

    @staticmethod
    def _plain_text_content(message: AnyMessage) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    texts.append(str(part["text"]))
            return "\n".join(texts)
        return str(content)

    def _log_prompt_messages(self, state: GraphState) -> None:
        logger = getattr(current_app, "logger", None)
        if not logger:
            return
        serialized = [
            {
                "role": getattr(message, "type", message.__class__.__name__),
                "content": self._plain_text_content(message),
            }
            for message in state.get("messages", [])
        ]
        logger.info(
            "Thread %s -> prompt messages: %s",
            state.get("thread_id"),
            json.dumps(serialized, ensure_ascii=False),
        )

    def _log_model_answer(self, thread_id: str, answer: str) -> None:
        logger = getattr(current_app, "logger", None)
        if not logger:
            return
        preview = answer if len(answer) < 500 else f"{answer[:500]}..."
        logger.info("Thread %s <- model answer: %s", thread_id, preview)

    def _ensure_pending_message_row(self, state: GraphState, answer: str) -> int:
        """Ensure there is exactly one pending ChatMessage per thread."""

        latest_pending = (
            ChatMessage.query.filter_by(
                user_id=state["user_id"],
                thread_id=state["thread_id"],
                role="assistant",
            )
            .order_by(ChatMessage.created_at.desc())
            .first()
        )
        if latest_pending and latest_pending.approved is None:
            updated = False
            if latest_pending.content != answer:
                latest_pending.content = answer
                updated = True
            if latest_pending.rejection_reason:
                latest_pending.rejection_reason = None
                updated = True
            if updated:
                latest_pending.created_at = datetime.now(UTC)
                db.session.commit()
            state["pending_message_id"] = latest_pending.id
            return latest_pending.id

        new_message = ChatMessage(
            user_id=state["user_id"],
            role="assistant",
            content=answer,
            thread_id=state["thread_id"],
            approved=None,
        )
        db.session.add(new_message)
        db.session.commit()
        state["pending_message_id"] = new_message.id
        return new_message.id

    def _cleanup_extra_pending(self, state: GraphState, active_id: int) -> None:
        extras = (
            ChatMessage.query.filter(
                ChatMessage.user_id == state["user_id"],
                ChatMessage.thread_id == state["thread_id"],
                ChatMessage.role == "assistant",
                ChatMessage.approved.is_(None),
                ChatMessage.id != active_id,
            )
            .all()
        )
        if not extras:
            return
        for extra in extras:
            extra.approved = False
        db.session.commit()

def _get_chat_manager() -> ChatGraphManager:
    global _CHAT_MANAGER
    if _CHAT_MANAGER is None:
        _CHAT_MANAGER = ChatGraphManager()
    return _CHAT_MANAGER


_CHAT_MANAGER: ChatGraphManager | None = None
_WEATHER_AGENT = None


def _get_weather_agent():
    global _WEATHER_AGENT
    if _WEATHER_AGENT is None:
        model = WeatherLLM()
        _WEATHER_AGENT = create_agent(
            model=model,
            tools=[weather_tool],
            middleware=[adjust_prompt],
            response_format=ToolStrategy(Weather),
            context_schema=Context,
        )
    return _WEATHER_AGENT


def reset_chat_runtime() -> None:
    """Helper used in tests to close the sqlite checkpointer connection."""

    global _CHAT_MANAGER, _WEATHER_AGENT
    if _CHAT_MANAGER and _CHAT_MANAGER._conn:
        _CHAT_MANAGER._conn.close()
    _CHAT_MANAGER = None
    _WEATHER_AGENT = None


def reset_thread_state(thread_id: str) -> None:
    """Remove checkpoints for a given conversation thread."""

    if not thread_id:
        return
    manager = _get_chat_manager()
    manager.reset_thread(thread_id)
