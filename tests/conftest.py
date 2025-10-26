from __future__ import annotations

import os
from pathlib import Path

import pytest

from app import create_app
from app.chat.agent_runtime import reset_chat_runtime
from app.config import Config
from app.extensions import db


class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED = False
    USE_FAKE_LLM = True
    GRAPH_DB_PATH = "test_graph_state.db"


@pytest.fixture()
def app():
    os.environ["USE_FAKE_LLM"] = "1"
    application = create_app(TestConfig)
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()
    reset_chat_runtime()
    graph_file = Path(TestConfig.GRAPH_DB_PATH)
    if graph_file.exists():
        graph_file.unlink()


@pytest.fixture()
def client(app):
    return app.test_client()
