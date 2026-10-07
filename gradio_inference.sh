#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    interpreter="$VIRTUAL_ENV/bin/python"
elif [[ -x .venv/bin/python ]]; then
    interpreter=.venv/bin/python
else
    interpreter=python
fi
exec "$interpreter" app.py "$@"
