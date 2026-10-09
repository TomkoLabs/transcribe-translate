#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${TT_TEST_PYTHON:-$ROOT/.venv/bin/python}"
bash -n "$ROOT/setup.sh" "$ROOT/uninstall.sh" "$ROOT/transcribe-translate" "$ROOT/scripts/check.sh"
"$PYTHON" -m compileall -q "$ROOT/src" "$ROOT/scripts"
"$PYTHON" -m pytest "$ROOT/tests" -q
if git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    git -C "$ROOT" diff --check
fi
