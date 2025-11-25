#!/usr/bin/env bash
# POSIX test runner for the repository
# Usage: ./scripts/run_tests.sh
set -euo pipefail
VENV_DIR=".venv"
PYTHON="$VENV_DIR/bin/python"

if [ ! -x "$PYTHON" ]; then
  echo "Creating virtualenv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# Upgrade pip and install minimal test deps
"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install pytest pytest-mock numpy pydantic jsonlines pandas tqdm -q

# Run pytest
"$PYTHON" -m pytest -q
