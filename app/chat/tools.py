from __future__ import annotations

import json
from datetime import datetime
from random import Random

from langchain_core.tools import tool
from langgraph.runtime import get_runtime

from app.extensions import db
from app.models import ToolCallLog

from .dynamic_prompting import Context


def _get_runtime_context() -> Context:
    runtime = get_runtime(Context)
    context = runtime.context or Context()
    runtime.context = context
    return context


def _start_tool_log(tool_name: str, args: dict) -> ToolCallLog:
    context = _get_runtime_context()
    if context.current_tool_log_id:
        existing = ToolCallLog.query.get(context.current_tool_log_id)
        if existing:
            return existing

    log = ToolCallLog(
        user_id=context.user_id or 0,
        tool_name=tool_name,
        args_json=json.dumps(args, ensure_ascii=False),
        thread_id=context.thread_id,
    )
    db.session.add(log)
    db.session.commit()
    context.current_tool_log_id = log.id
    return log


def _finish_tool_log(log: ToolCallLog, result: dict, approved: bool | None) -> None:
    log.result_json = json.dumps(result, ensure_ascii=False)
    log.approved = approved
    db.session.commit()
    context = _get_runtime_context()
    context.current_tool_log_id = None


@tool
def read_email(subject: str) -> str:
    """Provide a short summary of an email by subject."""

    log = _start_tool_log("read_email", {"subject": subject})
    summary = f"Summary for '{subject}': routine update ({datetime.utcnow().date()})."
    _finish_tool_log(log, {"summary": summary}, approved=True)
    return summary


@tool
def search_web(query: str) -> str:
    """Return lightweight web search results."""

    log = _start_tool_log("search_web", {"query": query})
    result = f"Top search result for '{query}': Python docs recommendation."
    _finish_tool_log(log, {"result": result}, approved=True)
    return result


@tool
def analyze_data(numbers: list[float]) -> str:
    """Compute simple statistics for a list of numbers."""

    log = _start_tool_log("analyze_data", {"numbers": numbers})
    if not numbers:
        summary = "No data received."
    else:
        avg = sum(numbers) / len(numbers)
        summary = f"Average: {avg:.2f} calculated from {len(numbers)} values."
    _finish_tool_log(log, {"summary": summary}, approved=True)
    return summary


@tool
def weather_tool(city: str) -> dict:
    """Return dummy weather data for a city."""

    log = _start_tool_log("weather_tool", {"city": city})
    rnd = Random(city.lower())
    temperature = rnd.uniform(-2, 30)
    condition = rnd.choice(["sunny", "cloudy", "rain", "windy"])
    payload = {"city": city, "temperature": round(temperature, 1), "condition": condition}
    _finish_tool_log(log, payload, approved=True)
    return payload


ASSISTANT_TOOLS = [read_email, search_web, analyze_data, weather_tool]
