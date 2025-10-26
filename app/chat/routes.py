from __future__ import annotations

import time
from uuid import uuid4

from flask import (
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_login import current_user, login_required

from app.extensions import db
from app.models import ChatMessage, ToolCallLog

from . import chat_bp
from .agent_runtime import (
    build_langchain_history,
    reset_thread_state,
    resume_chat_turn,
    run_chat_turn,
    weather_demo,
)


def _thread_id() -> str:
    thread = session.get("thread_id")
    if not thread:
        thread = uuid4().hex
        session["thread_id"] = thread
    return thread


def _enforce_rate_limit() -> None:
    last_ts = session.get("last_post_ts", 0.0)
    delta = time.monotonic() - float(last_ts)
    min_interval = current_app.config.get("POST_MIN_INTERVAL_SECONDS", 2)
    if delta < min_interval:
        abort(429, description="Too many requests. Please wait a moment.")
    session["last_post_ts"] = time.monotonic()


def _has_pending_message(thread_id: str) -> bool:
    return (
        ChatMessage.query.filter_by(
            user_id=current_user.id, thread_id=thread_id, role="assistant", approved=None
        ).first()
        is not None
    )


@chat_bp.route("/", methods=["GET"])
@login_required
def index():
    thread_id = _thread_id()
    messages = (
        ChatMessage.query.filter_by(user_id=current_user.id, thread_id=thread_id)
        .order_by(ChatMessage.created_at)
        .all()
    )
    pending_tools = (
        ToolCallLog.query.filter_by(user_id=current_user.id, approved=None, thread_id=thread_id)
        .order_by(ToolCallLog.created_at)
        .all()
    )
    awaiting_message = any(msg.role == "assistant" and msg.approved is None for msg in messages)
    status_message = None
    if awaiting_message:
        status_message = "Waiting for approval of the last answer."
    elif pending_tools:
        status_message = "A tool call requires approval."
    return render_template(
        "chat/index.html",
        messages=messages,
        pending_tools=pending_tools,
        awaiting_message=awaiting_message,
        status_message=status_message,
        thread_id=thread_id,
    )


@chat_bp.route("/", methods=["POST"])
@login_required
def send_message():
    _enforce_rate_limit()
    thread_id = _thread_id()
    if _has_pending_message(thread_id):
        flash("Please review the current assistant response first.", "warning")
        return redirect(url_for("chat.index"))

    content = (request.form.get("message") or "").strip()
    if not content:
        flash("Your question cannot be empty.", "warning")
        return redirect(url_for("chat.index"))

    user_message = ChatMessage(
        user_id=current_user.id,
        role="user",
        content=content,
        thread_id=thread_id,
        approved=True,
    )
    db.session.add(user_message)
    db.session.commit()

    messages = (
        ChatMessage.query.filter_by(user_id=current_user.id, thread_id=thread_id)
        .order_by(ChatMessage.created_at)
        .all()
    )
    history = build_langchain_history(messages)
    last_feedback = session.pop("last_feedback", None)
    run_chat_turn(
        user=current_user,
        thread_id=thread_id,
        history=history,
        question=content,
        last_feedback=last_feedback,
    )
    flash("Response is being prepared. Use Approve/Reject when it arrives.", "info")
    return redirect(url_for("chat.index"))


@chat_bp.route("/reset", methods=["POST"])
@login_required
def reset_chat():
    thread_id = session.get("thread_id")
    if thread_id:
        (
            ChatMessage.query.filter_by(user_id=current_user.id, thread_id=thread_id)
            .delete(synchronize_session=False)
        )
        (
            ToolCallLog.query.filter_by(user_id=current_user.id, thread_id=thread_id)
            .delete(synchronize_session=False)
        )
        db.session.commit()
        reset_thread_state(thread_id)
    session["thread_id"] = uuid4().hex
    flash("Chat has been reset.", "info")
    return redirect(url_for("chat.index"))


@chat_bp.route("/approve", methods=["POST"])
@login_required
def approve_message():
    message_id = request.form.get("message_id")
    if not message_id:
        abort(400, description="message_id missing")
    payload = {"kind": "message_review", "approved": True, "message_id": int(message_id)}
    resume_chat_turn(user=current_user, thread_id=_thread_id(), payload=payload)
    flash("Answer approved.", "success")
    return redirect(url_for("chat.index"))


@chat_bp.route("/reject", methods=["POST"])
@login_required
def reject_message():
    message_id = request.form.get("message_id")
    if not message_id:
        abort(400, description="message_id missing")
    feedback = (request.form.get("feedback") or "").strip()
    session["last_feedback"] = feedback or None
    payload = {
        "kind": "message_review",
        "approved": False,
        "message_id": int(message_id),
        "feedback": feedback,
    }
    resume_chat_turn(user=current_user, thread_id=_thread_id(), payload=payload)
    flash("Answer will be regenerated using your feedback.", "warning")
    return redirect(url_for("chat.index"))


@chat_bp.route("/tool/approve", methods=["POST"])
@login_required
def approve_tool():
    log_id = request.form.get("log_id")
    if not log_id:
        abort(400, description="log_id missing")
    payload = {"kind": "tool_review", "approved": True, "log_id": int(log_id)}
    resume_chat_turn(user=current_user, thread_id=_thread_id(), payload=payload)
    flash("Tool call approved.", "success")
    return redirect(url_for("chat.index"))


@chat_bp.route("/tool/reject", methods=["POST"])
@login_required
def reject_tool():
    log_id = request.form.get("log_id")
    if not log_id:
        abort(400, description="log_id missing")
    feedback = (request.form.get("feedback") or "").strip()
    payload = {
        "kind": "tool_review",
        "approved": False,
        "log_id": int(log_id),
        "feedback": feedback,
    }
    resume_chat_turn(user=current_user, thread_id=_thread_id(), payload=payload)
    flash("Tool call rejected.", "info")
    return redirect(url_for("chat.index"))


@login_required
def weather_demo_api():
    city = request.args.get("city", "Berlin")
    weather = weather_demo(city)
    return jsonify(weather.model_dump())
