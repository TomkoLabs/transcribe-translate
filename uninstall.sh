#!/usr/bin/env bash
set -euo pipefail
SCRIPT="$(readlink -f -- "${BASH_SOURCE[0]}")"
ROOT="$(cd -P -- "$(dirname -- "$SCRIPT")" && pwd)"
PYTHON_ARG=""
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[i]}" == --python ]]; then
        ((i+1 < ${#ARGS[@]})) || { echo '--python requires an executable' >&2; exit 2; }
        PYTHON_ARG="${ARGS[i+1]}"
    elif [[ "${ARGS[i]}" == --python=* ]]; then
        PYTHON_ARG="${ARGS[i]#--python=}"
        [[ -n "$PYTHON_ARG" ]] || { echo '--python requires an executable' >&2; exit 2; }
    fi
done
# Resolve a base interpreter; activated environments and their pip scripts are not
# installation destinations. -I/-S ignore PYTHONHOME/PYTHONPATH and site hooks.
CANDIDATES=()
if [[ -n "$PYTHON_ARG" ]]; then
    CANDIDATES+=("$PYTHON_ARG")
else
    for name in python3.11 python3.12 python3.13 python3; do
        path="$(command -v "$name" || true)"
        [[ -z "$path" ]] || CANDIDATES+=("$path")
    done
    CANDIDATES+=(/usr/bin/python3 /usr/local/bin/python3)
fi
BASE=""
for candidate in "${CANDIDATES[@]}"; do
    if BASE="$("$candidate" -I -S -c 'import os,sys; assert (3,11) <= sys.version_info[:2] < (3,14); p=sys._base_executable; assert os.path.isfile(p); print(p)' 2>/dev/null)"; then
        break
    fi
    BASE=""
done
[[ -n "$BASE" ]] || { echo 'Install supported Python 3.11–3.13 with venv/pip, or pass --python.' >&2; exit 2; }
exec "$BASE" -I -B "$ROOT/scripts/remove.py" --root "$ROOT" "${ARGS[@]}"
