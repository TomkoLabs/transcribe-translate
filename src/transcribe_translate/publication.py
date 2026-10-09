"""Owned, recoverable source/target TXT copies, with tracking outside the output."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

from .storage import StateError, atomic_text, digest, file_hash, load, private_dir, save, state_directory


def valid_hash(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def name_key(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


def readable_stem(stem: str, suffix: str = "") -> str:
    # Reserve room for every supported canonical code, including three-letter codes.
    budget = 247 - len(suffix.encode("utf-8"))
    return (stem.encode("utf-8")[:budget].decode("utf-8", errors="ignore") or "video") + suffix


def reserve_stem(stem: str, identity: str, used: set[str]) -> str:
    candidate = readable_stem(stem)
    if name_key(candidate) in used:
        candidate = readable_stem(stem, f"--{digest(identity)[:16]}")
        if name_key(candidate) in used:
            candidate = readable_stem(stem, f"--{digest(identity)}")
        if name_key(candidate) in used:
            raise StateError("Cannot disambiguate transcript filenames")
    used.add(name_key(candidate))
    return candidate


def validate_registry(data: dict) -> dict:
    if data.get("schema") != 2 or not isinstance(data.get("inputs"), dict):
        raise StateError("Invalid transcript publication registry")
    stems = set()
    for key, entry in data["inputs"].items():
        if not isinstance(entry, dict):
            raise StateError("Invalid transcript ownership")
        stem = entry.get("stem")
        if (not isinstance(entry.get("input_path"), str) or key != digest(entry["input_path"])
                or not isinstance(stem, str) or Path(stem).name != stem or "\0" in stem
                or not stem or len(stem.encode("utf-8")) > 247 or name_key(stem) in stems
                or not isinstance(entry.get("files"), dict)):
            raise StateError("Invalid transcript ownership or stem")
        stems.add(name_key(stem))
        for language, artifact in entry["files"].items():
            if (re.fullmatch(r"[a-z]{2,3}", language) is None or not isinstance(artifact, dict)
                    or "sha256" not in artifact
                    or (artifact["sha256"] is not None and not valid_hash(artifact["sha256"]))
                    or ("pending_sha256" in artifact and not valid_hash(artifact["pending_sha256"]))):
                raise StateError("Invalid transcript publication checksum")
    return data


def verify_public(output: Path, name: str, artifact: dict) -> str | None:
    path = output / name
    for other in output.iterdir():
        if other.name != name and name_key(other.name) == name_key(name):
            raise StateError(f"Unrelated transcript filename collision: {other}; choose a fresh output folder")
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise StateError(f"Unowned or symlink transcript: {path}; choose a fresh output folder")
    actual = file_hash(path)
    if actual not in {artifact.get("sha256"), artifact.get("pending_sha256")}:
        raise StateError(f"Transcript was edited or is unrelated: {path}; preserve it and choose a fresh output folder")
    return actual


def upgrade_registry(raw: dict, output: Path) -> tuple[dict, list[dict]]:
    """Validate 0.2.3 tracking and plan owned-file renames for paired stems."""
    if raw.get("schema") != 1 or not isinstance(raw.get("files"), dict):
        raise StateError("Invalid legacy publication registry")
    data = {"schema": 2, "inputs": {}}
    used, names, moves = set(), set(), []
    for old_key, entry in raw["files"].items():
        if not isinstance(entry, dict):
            raise StateError("Invalid legacy TXT ownership")
        name, language, identity = entry.get("filename"), entry.get("target_language"), entry.get("input_path")
        if (not isinstance(identity, str) or not isinstance(language, str)
                or re.fullmatch(r"[a-z]{2,3}", language) is None
                or not isinstance(name, str) or Path(name).name != name or "\0" in name
                or not name.endswith(f".{language}.txt") or len(name.encode()) > 255
                or old_key != digest({"input_path": identity, "target_language": language})
                or (entry.get("sha256") is not None and not valid_hash(entry["sha256"]))
                or ("pending_sha256" in entry and not valid_hash(entry["pending_sha256"]))
                or name_key(name) in names):
            raise StateError("Invalid legacy TXT ownership or filename")
        names.add(name_key(name))
        artifact = {k: entry[k] for k in ("sha256", "pending_sha256") if k in entry}
        artifact.setdefault("sha256", None)
        actual = verify_public(output, name, artifact)
        key = digest(identity)
        if key not in data["inputs"]:
            stem = reserve_stem(name[:-len(f".{language}.txt")], identity, used)
            data["inputs"][key] = {"input_path": identity, "stem": stem, "files": {}}
        owner = data["inputs"][key]
        owner["files"][language] = artifact
        new_name = f"{owner['stem']}.{language}.txt"
        if actual is not None:
            moves.append({"old": name, "new": new_name, "sha256": actual})
    return validate_registry(data), moves


class Publication:
    """Called under the batch lock; all languages for an input share one stem."""

    def __init__(self, output: Path, source: Path, language: str, *, state: Path | None = None):
        self.output = output
        self.state = state if state is not None else state_directory(output)
        self.registry = self.state / "publications.json"
        self.identity = str(source.resolve())
        self.key = digest(self.identity)
        self.language = language
        data = self._load()
        entry = data["inputs"].get(self.key)
        changed = entry is None
        if entry is None:
            used = {name_key(e["stem"]) for e in data["inputs"].values()}
            entry = {"input_path": self.identity, "stem": reserve_stem(source.stem, self.identity, used), "files": {}}
            data["inputs"][self.key] = entry
        for code, artifact in entry["files"].items():
            verify_public(output, f"{entry['stem']}.{code}.txt", artifact)
        changed = changed or language not in entry["files"]
        artifact = entry["files"].setdefault(language, {"sha256": None})
        self.stem = entry["stem"]
        self.path = self.output / f"{self.stem}.{language}.txt"
        verify_public(output, self.path.name, artifact)
        if changed:
            save(self.registry, data)

    def _load(self) -> dict:
        raw = load(self.registry) if self.registry.exists() or self.registry.is_symlink() else {"schema": 2, "inputs": {}}
        return validate_registry(raw)

    def publish(self, text: str, *, language: str | None = None) -> Path:
        language = language or self.language
        data = self._load()
        entry = data["inputs"].get(self.key)
        if entry is None or entry["stem"] != self.stem:
            raise StateError("Transcript ownership changed during processing")
        artifact = entry["files"].setdefault(language, {"sha256": None})
        path = self.output / f"{self.stem}.{language}.txt"
        actual = verify_public(self.output, path.name, artifact)
        expected = hashlib.sha256(text.encode()).hexdigest()
        if actual == expected and artifact["sha256"] == expected and "pending_sha256" not in artifact:
            return path
        artifact["pending_sha256"] = expected
        save(self.registry, data)
        temporary = self.state / "temporary"
        private_dir(temporary)
        if actual != expected:
            atomic_text(path, text, temporary=temporary)
        artifact["sha256"] = expected
        artifact.pop("pending_sha256")
        save(self.registry, data)
        return path
