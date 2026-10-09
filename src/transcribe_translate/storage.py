from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path

OWNER = "transcribe-translate/v1"


class StateError(RuntimeError):
    pass


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(path: Path) -> dict:
    before = path.stat()
    h = file_hash(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise StateError("Input changed while hashing; retry when copying/editing has finished")
    return {"sha256": h, "size": after.st_size, "mtime_ns": after.st_mtime_ns}


def atomic_bytes(path: Path, content: bytes, *, temporary: Path | None = None) -> None:
    if path.is_symlink():
        raise StateError(f"Refusing a symlink output: {path}")
    fd, name = tempfile.mkstemp(prefix="write-", suffix=".tmp", dir=temporary or path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        d = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(d)
        finally:
            os.close(d)
        if temporary is not None and temporary != path.parent:
            sync_directory(temporary)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_text(path: Path, text: str, *, temporary: Path | None = None) -> None:
    atomic_bytes(path, text.encode("utf-8"), temporary=temporary)


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(path: Path, payload: dict) -> None:
    atomic_text(path, json.dumps({"owner": OWNER, "sha256": digest(payload), "payload": payload}, ensure_ascii=False, indent=2) + "\n")


def load(path: Path) -> dict:
    if path.is_symlink():
        raise StateError(f"Refusing a symlink checkpoint: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        payload = raw["payload"]
        if raw["owner"] != OWNER or raw["sha256"] != digest(payload) or not isinstance(payload, dict):
            raise ValueError("invalid owner or checksum")
        return payload
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise StateError(f"Invalid or edited checkpoint: {path}; preserve it and choose a fresh output folder") from exc


def private_dir(path: Path) -> None:
    if path.is_symlink():
        raise StateError(f"Refusing a symlink state directory: {path}")
    path.mkdir(mode=0o700, exist_ok=True)


def initialize_directory(path: Path, name: str, payload: dict) -> None:
    """Publish the ownership record and directory together, including after a crash."""
    staging = Path(tempfile.mkdtemp(prefix="initialize-", dir=path.parent))
    try:
        save(staging / name, payload)
        os.rename(staging, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if staging.exists():
            (staging / name).unlink(missing_ok=True)
            staging.rmdir()


def state_directory(output: Path, work: Path | None = None) -> Path:
    if output.is_symlink():
        raise StateError(f"Refusing a symlink output folder: {output}")
    output.mkdir(parents=True, exist_ok=True)
    work = work if work is not None else output.parent / "Work"
    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    if work.is_symlink():
        raise StateError(f"Refusing a symlink Work folder: {work}")
    label = output.name.lstrip(".").encode("utf-8")[:70].decode("utf-8", errors="ignore") or "output"
    path = work / f"{label}--{digest(str(output.resolve()))[:16]}"
    marker = path / "owner.json"
    ownership = {"layout": "work-state-v1", "output_path": str(output.resolve())}
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not marker.is_file() or load(marker) != ownership:
            raise StateError(f"Unowned state folder: {path}; choose another output folder")
    else:
        initialize_directory(path, "owner.json", ownership)
    return path


def job_name(source: Path) -> str:
    label = "".join(c if c.isalnum() or c in " ._-" else "_" for c in source.name)[:70].strip(" .") or "video"
    # Existing valid legacy names are unchanged; a path-hash suffix must also fit.
    label = label.encode("utf-8")[:237].decode("utf-8", errors="ignore")
    identity = str(source.resolve())
    return f"{label}--{digest(identity)[:16]}"


def job_directory(output: Path, source: Path, work: Path | None = None) -> Path:
    identity = str(source.resolve())
    inputs = state_directory(output, work) / "inputs"
    private_dir(inputs)
    path = inputs / job_name(source)
    marker = path / "owner.json"
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not marker.is_file() or load(marker) != {"input_path": identity}:
            raise StateError(f"Unowned output collision: {path}; choose another output folder")
    else:
        initialize_directory(path, "owner.json", {"input_path": identity})
    return path


def write_artifact(path: Path, content: str, expected: str) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or file_hash(path) != expected:
            raise StateError(f"Transcript was edited or is unrelated: {path}; preserve it and choose a fresh output folder")
        return
    if hashlib.sha256(content.encode()).hexdigest() != expected:
        raise StateError("Internal transcript checksum mismatch")
    atomic_text(path, content)


@contextmanager
def batch_lock(output: Path, work: Path | None = None):
    """Directory flock serializes one output even with different Work overrides."""
    output.mkdir(parents=True, exist_ok=True)
    fds = []
    legacy = output / ".transcribe-translate.lock"
    legacy_fd = None
    try:
        fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        fds.append(fd)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StateError("Another batch is using this output folder; wait for it to finish") from exc
        paths = [state_directory(output, work) / "batch.lock"]
        if legacy.exists() or legacy.is_symlink():
            info = legacy.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size or info.st_uid != os.getuid():
                raise StateError(f"Unowned or modified legacy lock: {legacy}")
            paths.insert(0, legacy)
        for path in paths:
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            fds.append(fd)
            if path == legacy:
                legacy_fd = fd
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StateError("Another batch is using this output folder; wait for it to finish") from exc
        yield
        if legacy_fd is not None and legacy.exists():
            if legacy.stat().st_ino != os.fstat(legacy_fd).st_ino or legacy.stat().st_size:
                raise StateError("Legacy lock changed during migration")
            legacy.unlink()
            sync_directory(output)
    finally:
        for fd in reversed(fds):
            os.close(fd)
