from __future__ import annotations

import os
from pathlib import Path


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key")
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", "sqlite:///app.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SESSION_COOKIE_HTTPONLY = True
    LANGCHAIN_DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "openai:gpt-4o-mini")
    LANGCHAIN_PREMIUM_MODEL = os.environ.get("PREMIUM_MODEL", "openai:gpt-4o")
    GRAPH_DB_PATH = os.environ.get("GRAPH_DB_PATH", str(Path("graph_state.db")))
    POST_MIN_INTERVAL_SECONDS = float(os.environ.get("POST_MIN_INTERVAL", 2))
    USE_FAKE_LLM = (
        os.environ.get(
            "USE_FAKE_LLM",
            "1" if os.environ.get("FLASK_ENV") == "development" else "0",
        )
        == "1"
    )
    WTF_CSRF_TIME_LIMIT = None
    LANGCHAIN_TRACING_V2 = os.environ.get("LANGCHAIN_TRACING_V2", "false")
