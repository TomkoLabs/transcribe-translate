"""Checksummed ownership for disposable files; cleanup never traverses symlinks."""
from contextlib import contextmanager
import os
from pathlib import Path

from .storage import StateError, file_hash, load, save

MARKER = '.transcribe-translate-owner.json'


def no_links(path: Path, boundary: Path):
    if not path.is_absolute() or not path.is_relative_to(boundary) or path == boundary:
        raise StateError(f'Cleanup/storage path is outside its boundary: {path}')
    for p in [path, *path.parents]:
        if p == boundary:
            break
        if p.is_symlink():
            raise StateError(f'Refusing linked managed path: {p}')


def tree(path: Path) -> dict:
    result = {}
    # os.walk does not follow directory links; their entries are recorded as links.
    for parent, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            p = Path(parent)/name
            relative = p.relative_to(path).as_posix()
            if relative == MARKER:
                continue
            if p.is_symlink():
                result[relative] = {'link': os.readlink(p)}
            elif p.is_file():
                result[relative] = {'sha256': file_hash(p)}
            elif p.is_dir():
                result[relative] = {'directory': True}
            else:
                raise StateError(f'Unsupported managed artifact: {p}')
    return result


def owner(path: Path, kind: str, *, create=False):
    if path.is_symlink():
        raise StateError(f'Refusing linked managed directory: {path}')
    marker = path/MARKER
    if marker.exists() or marker.is_symlink():
        data = load(marker)
        if data.get('layout') != kind or not isinstance(data.get('files'), dict):
            raise StateError(f'Unowned managed directory: {path}')
        for name in data['files']:
            p = Path(name)
            if p.is_absolute() or '..' in p.parts or not p.parts or name == MARKER:
                raise StateError('Unsafe ownership path')
        return data
    if not create or (path.exists() and any(path.iterdir())):
        raise StateError(f'Unowned managed directory: {path}; preserve it')
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    data = {'layout': kind, 'files': {}, 'active': False}
    save(marker, data)
    return data


@contextmanager
def operation(path: Path, kind: str, *, mutable=()):
    data = owner(path, kind, create=True)
    current = tree(path)
    for name,info in current.items():
        if 'link' in info and not (path/name).resolve().is_relative_to(path.resolve()):
            raise StateError(f'External link in managed write area: {path/name}')
    if current != data['files']:
        # Intent was committed before the library wrote recoverable partial files.
        # Only its reserved paths can change during an interrupted operation.
        changed = set(current) | set(data['files'])
        changed = {p for p in changed if current.get(p) != data['files'].get(p)}
        if not data.get('active') or any(not (any(p == a or p.startswith(a+'/') for a in mutable) or ('/' not in p and p.startswith('write-') and p.endswith('.tmp'))) for p in changed):
            raise StateError(f'Edited/unrelated managed files: {path}; preserve them')
    data.update(files=current, active=True)
    save(path/MARKER, data)
    try:
        yield
    finally:
        data.update(files=tree(path), active=False)
        save(path/MARKER, data)


def remove(path: Path, boundary: Path, kind: str) -> bool:
    no_links(path, boundary)
    if not path.exists():
        return False
    data = owner(path, kind)
    current = tree(path)
    if data.get('active') or any(data['files'].get(n)!=v for n,v in current.items()):
        raise StateError(f'Edited/unrelated or interrupted managed files retained: {path}; rerun setup to finish recovery')
    for name, info in sorted(current.items(), key=lambda item: len(Path(item[0]).parts), reverse=True):
        p = path/name
        # Check every ancestor again, but unlink a recorded link itself locally.
        no_links(p.parent, boundary) if p.parent != boundary else None
        if info.get('directory'):
            p.rmdir()
        else:
            p.unlink()
    (path/MARKER).unlink()
    path.rmdir()
    return True


def runtime_stamp(path: Path):
    """Adopt only pip-recorded files and generated venv files, never personal data."""
    import csv
    known = {'pyvenv.cfg', 'installation.json', 'bin/activate', 'bin/activate.csh',
             'bin/activate.fish', 'bin/Activate.ps1', 'bin/python', 'bin/python3',
             'bin/python3.11', 'bin/python3.12', 'bin/python3.13', 'lib64'}
    for record in path.glob('lib/python*/site-packages/*.dist-info/RECORD'):
        if record.is_symlink():
            continue
        for row in csv.reader(record.read_text().splitlines()):
            candidate = Path(os.path.abspath(record.parent.parent/row[0]))
            if candidate.is_relative_to(path):
                known.add(candidate.relative_to(path).as_posix())
    contents = tree(path)
    for name in contents:
        p = Path(name)
        if p.suffix == '.pyc' and p.parent.name == '__pycache__':
            module = p.name.split('.cpython-')[0]+'.py'
            if (p.parent.parent/module).as_posix() in known:
                known.add(name)
    known.update(n for n,v in contents.items() if v.get('directory'))
    save(path/MARKER, {'layout': 'runtime-v1', 'files': {n:v for n,v in contents.items() if n in known}, 'active': False})
