"""Isolated checkout launcher: ignore foreign Python environment/path injection."""
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
os.environ['TT_ROOT'] = str(root)
from transcribe_translate.installation import installation_lock, verify

try:
    with installation_lock(root):
        verify(root)
        from transcribe_translate.cli import main
        raise SystemExit(main())
except (ValueError, RuntimeError, OSError, ImportError) as exc:
    print(f'Installation error: {exc}', file=sys.stderr)
    raise SystemExit(2)
