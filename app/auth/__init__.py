﻿from flask import Blueprint

auth_bp = Blueprint("auth", __name__, template_folder="../templates/auth")

from . import routes  # noqa: E402,F401  pylint: disable=wrong-import-position
