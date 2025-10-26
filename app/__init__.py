from __future__ import annotations

from flask import Flask, redirect, url_for
from flask_login import current_user

from .auth import auth_bp
from .chat import chat_bp
from .chat.routes import weather_demo_api
from .config import Config
from .extensions import csrf, db, login_manager, migrate
from .models import User


def create_app(config_class: type[Config] = Config) -> Flask:
    """Application factory used by both CLI and WSGI entry points."""

    app = Flask(__name__)
    app.config.from_object(config_class)

    _register_extensions(app)
    _register_blueprints(app)
    _register_routes(app)

    with app.app_context():
        db.create_all()

    return app


def _register_extensions(app: Flask) -> None:
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    csrf.init_app(app)

    login_manager.login_view = "auth.login"
    login_manager.login_message_category = "info"

    @login_manager.user_loader
    def load_user(user_id: str) -> User | None:  # pragma: no cover - simple flask hook
        if user_id.isdigit():
            return User.query.get(int(user_id))
        return None

    @app.context_processor
    def inject_csrf() -> dict[str, callable]:  # pragma: no cover - template helper
        from flask_wtf.csrf import generate_csrf

        return {"csrf_token": generate_csrf}

    @app.after_request
    def set_security_headers(response):  # pragma: no cover - middleware style hook
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Cache-Control", "no-store")
        return response


def _register_blueprints(app: Flask) -> None:
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(chat_bp, url_prefix="/chat")


def _register_routes(app: Flask) -> None:
    @app.route("/")
    def index():
        if current_user.is_authenticated:
            return redirect(url_for("chat.index"))
        return redirect(url_for("auth.login"))

    app.add_url_rule("/demo/weather", view_func=weather_demo_api)
