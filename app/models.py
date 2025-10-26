from __future__ import annotations

from datetime import datetime

from flask_login import UserMixin
from sqlalchemy import Index, func

from .extensions import db


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    skill_level = db.Column(db.String(20), nullable=False, default="beginner")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    messages = db.relationship("ChatMessage", backref="user", lazy=True)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<User {self.username}>"


class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    approved = db.Column(db.Boolean, nullable=True)
    rejection_reason = db.Column(db.Text, nullable=True)
    thread_id = db.Column(db.String(64), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    metadata_json = db.Column(db.Text, nullable=True)

    __table_args__ = (
        Index("ix_chatmessage_user", "user_id"),
        Index("ix_chatmessage_thread_created", "thread_id", "created_at"),
    )


class ToolCallLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    tool_name = db.Column(db.String(50), nullable=False)
    args_json = db.Column(db.Text, nullable=False)
    result_json = db.Column(db.Text, nullable=True)
    approved = db.Column(db.Boolean, nullable=True)
    thread_id = db.Column(db.String(64), nullable=True, index=True)
    created_at = db.Column(db.DateTime, server_default=func.now(), index=True)

    __table_args__ = (Index("ix_toolcall_user", "user_id"),)
