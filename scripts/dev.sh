#!/usr/bin/env bash
set -euo pipefail

CMD=${1:-serve}

case "$CMD" in
  serve)
    uv run flask --app wsgi:app run -p 5000
    ;;
  lint)
    uv run ruff check .
    ;;
  test)
    uv run pytest -q
    ;;
  all)
    uv run ruff check .
    uv run pytest -q
    ;;
  *)
    echo "Usage: scripts/dev.sh [serve|lint|test|all]"
    exit 1
    ;;
esac
