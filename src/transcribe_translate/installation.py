"""Installation identity and directory-inode locks; no inference dependencies."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import site
import tomllib
from urllib.parse import unquote, urlsplit


def source_version(root: Path) -> str:
    project = tomllib.loads((root / 'pyproject.toml').read_text())['project']['version']
    tree = ast.parse((root / 'src/transcribe_translate/__init__.py').read_text())
    version = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == '__version__' for t in n.targets))
    if version != project:
        raise ValueError('Source package and pyproject versions disagree')
    return version


def editable_root(metadata: Path) -> Path | None:
    data = json.loads(metadata.read_text())
    url = urlsplit(data['url'])
    if url.scheme != 'file' or url.netloc not in ('', 'localhost') or not data.get('dir_info', {}).get('editable'):
        return None
    return Path(unquote(url.path)).resolve()


def verify(root: Path) -> dict:
    root = root.resolve()
    env = root / '.venv'
    if env.is_symlink() or Path(sys.prefix).resolve() != env or Path(sys.executable).parent != env / 'bin':
        raise ValueError('Runtime does not belong to this checkout; run its ./setup.sh')
    import transcribe_translate
    origin = Path(transcribe_translate.__file__).resolve()
    distributions = [d for d in importlib.metadata.distributions(path=site.getsitepackages())
                     if d.metadata['Name'] == 'transcribe-translate']
    if len(distributions) != 1:
        raise ValueError('Missing/ambiguous installed application metadata; run ./setup.sh')
    dist = distributions[0]
    record = next((f for f in (dist.files or []) if f.name == 'direct_url.json'), None)
    if record is None:
        raise ValueError('Missing editable installation metadata; run ./setup.sh')
    metadata = Path(dist.locate_file(record))
    expected = source_version(root)
    if (origin != root / 'src/transcribe_translate/__init__.py' or dist.version != expected
            or transcribe_translate.__version__ != expected or editable_root(metadata) != root
            or not Path(dist.locate_file('')).resolve().is_relative_to(env)):
        raise ValueError('Version/import/editable installation mismatch; run this checkout’s ./setup.sh')
    from .config import load_config
    from .models import locations
    storage = locations(load_config(overrides={'translation_backend':'local','target_language':'en'}))
    return {**storage, 'version': expected, 'root': str(root), 'python': sys.executable,
            'environment': str(env), 'import': str(origin), 'launcher': str(root / 'transcribe-translate'),
            'installed_cli': str(env / 'bin/transcribe-translate')}


@contextmanager
def installation_lock(root: Path, *, exclusive: bool = False):
    """A canonical directory lock has no disposable lock-file race on reinstall."""
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    os.set_inheritable(fd, True)  # Keep protection across the CUDA CLI reexec.
    try:
        try:
            fcntl.flock(fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Installation is in use; wait for processing/setup to finish and rerun') from exc
        yield
    finally:
        os.close(fd)
