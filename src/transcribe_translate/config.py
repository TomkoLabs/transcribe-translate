from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from .languages import normalize_language

MODELS = frozenset("tiny tiny.en base base.en small small.en medium medium.en large-v1 large-v2 large-v3 large large-v3-turbo turbo distil-large-v2 distil-medium.en distil-small.en distil-large-v3 distil-large-v3.5".split())


def installation_root() -> Path:
    if os.environ.get("TT_ROOT"):
        return Path(os.environ["TT_ROOT"]).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file() and (checkout / "setup.sh").is_file():
        return checkout
    return Path(sys.prefix).resolve() / "share" / "transcribe-translate"


@dataclass(frozen=True)
class Config:
    root: Path
    source: Path
    output: Path
    work: Path | None = None
    models: Path | None = None
    shared_hub: Path | None = None
    source_language: str = "auto"
    target_language: str = "en"
    asr_model: str = "large-v3"
    asr_device: str = "auto"
    asr_compute_type: str = "auto"
    requested_device: str = "auto"
    requested_compute_type: str = "auto"
    translation_backend: str = "local"
    base_url: str = ""
    api_key: str = field(default="", repr=False)
    model: str = ""
    timeout_seconds: float = 180
    chunk_chars: int = 1200
    context_chars: int = 160
    attempts: int = 3

    @property
    def work_root(self) -> Path:
        return self.work if self.work is not None else self.root / "Work"

    @property
    def models_root(self) -> Path:
        return self.models if self.models is not None else self.root / "Models"

    @property
    def installation_cache(self) -> Path:
        return self.root / "Work" / "Installation"

    def validate(self, *, endpoint: bool = True) -> None:
        if self.translation_backend not in {"local", "api"}:
            raise ValueError("translation.backend must be local or api")
        if self.source == self.output:
            raise ValueError("Source and output folders must be different")
        if (self.work_root == self.source or self.work_root.is_relative_to(self.output)
                or self.output.is_relative_to(self.work_root)):
            raise ValueError("Work must be separate from the source and output folders")
        if self.asr_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("ASR device must be auto, cpu or cuda; there is no remote execution backend")
        if self.asr_compute_type not in {"auto", "int8", "float32", "float16", "int8_float16", "int8_float32"}:
            raise ValueError("Unsupported ASR compute type")
        if self.asr_device == "cpu" and self.asr_compute_type in {"float16", "int8_float16"}:
            raise ValueError("CPU requires int8, int8_float32 or float32; use int8 normally")
        if self.asr_model not in MODELS:
            model_path = Path(self.asr_model)
            if not model_path.is_dir() or not (model_path / "model.bin").is_file():
                raise ValueError("ASR model must be a faster-whisper model name or a local converted model directory")
        if self.asr_model.endswith(".en") and self.source_language not in {"en", "auto"}:
            raise ValueError("An English-only ASR model cannot transcribe the selected source language")
        if self.translation_backend == "local":
            if self.target_language != "en":
                raise ValueError("Local Whisper translation supports English (en) only; choose translation.backend = \"api\" and configure its endpoint/model for another target")
            if self.asr_model in MODELS and (self.asr_model.endswith(".en") or self.asr_model.startswith("distil-") or self.asr_model in {"turbo", "large-v3-turbo"}):
                raise ValueError("Local speech translation requires a translation-trained multilingual Whisper model; use large-v3 or choose the api backend")
            return
        if not 50 <= self.chunk_chars <= 2000 or not 0 <= self.context_chars <= 300:
            raise ValueError("chunk_chars must be 50..2000 and context_chars 0..300")
        if not 1 <= self.timeout_seconds <= 600 or not 1 <= self.attempts <= 5:
            raise ValueError("timeout_seconds must be 1..600 and attempts 1..5")
        if self.base_url:
            u = urlsplit(self.base_url)
            if u.scheme not in {"http", "https"} or not u.hostname or u.username or u.password or u.query or u.fragment:
                raise ValueError("Translation base URL must be HTTP(S), without credentials, query or fragment")
            if u.path.rstrip("/").endswith(("/chat/completions", "/models")):
                raise ValueError("Use the API base URL including its prefix (usually /v1), not a /models or /chat/completions operation URL")
        if endpoint and self.source_language != self.target_language and (not self.base_url or not self.model):
            raise ValueError("Configure translation.base_url and translation.model in config.local.toml or TT_TRANSLATION_BASE_URL / TT_TRANSLATION_MODEL")
        if any(c in self.api_key for c in "\r\n"):
            raise ValueError("API key must not contain newline characters")


def load_config(path: Path | None = None, overrides: dict | None = None) -> Config:
    root = installation_root()
    cfg_path = path or root / "config.local.toml"
    data = {}
    if cfg_path.exists():
        try:
            data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"Cannot read configuration {cfg_path}: {exc}") from exc
    permitted = {"source_folder", "output_folder", "work_folder", "models_folder", "source_language", "target_language", "asr_model", "asr_device", "asr_compute_type", "translation"}
    if set(data) - permitted:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(set(data) - permitted))}")
    translation = data.get("translation", {})
    if not isinstance(translation, dict) or set(translation) - {"backend", "base_url", "model", "api_key_env", "api_key", "timeout_seconds", "chunk_chars", "context_chars", "attempts"}:
        raise ValueError("Unknown or malformed translation configuration")
    values = dict(data)
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    backend = values.get("translation_backend", translation.get("backend", "local"))
    if not isinstance(backend, str) or backend not in {"local", "api"}:
        raise ValueError("translation.backend must be local or api")

    def folder(key: str, default: str) -> Path:
        v = values.get(key, default)
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{key} must be a nonempty path")
        p = Path(v).expanduser()
        return (p if p.is_absolute() else root / p).resolve()

    asr_model = values.get("asr_model", "large-v3")
    if not isinstance(asr_model, str) or not asr_model.strip():
        raise ValueError("asr_model must be a nonempty name or path")
    if asr_model not in MODELS:
        p = Path(asr_model).expanduser()
        asr_model = str((p if p.is_absolute() else root / p).resolve())
    # Unused API settings and environment credentials are not read or validated
    # for local processing. Old API configuration remains untouched on disk.
    api = translation if backend == "api" else {}
    key_env = api.get("api_key_env", "TT_TRANSLATION_API_KEY")
    if not isinstance(key_env, str):
        raise ValueError("api_key_env must be an environment variable name")
    settings = {}
    for key, default, kind in [("timeout_seconds", 180, (int, float)), ("chunk_chars", 1200, int), ("context_chars", 160, int), ("attempts", 3, int)]:
        v = api.get(key, default)
        if isinstance(v, bool) or not isinstance(v, kind):
            raise ValueError(f"{key} has an invalid type")
        settings[key] = v
    for key in ("base_url", "model", "api_key"):
        if key in api and not isinstance(api[key], str):
            raise ValueError(f"translation.{key} must be a string")
    shared_hub = None
    if "models_folder" not in values:
        choice = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
        if choice:
            p = Path(choice).expanduser()
            shared_hub = (p if p.is_absolute() else root/p).resolve()
            if not os.environ.get("HF_HUB_CACHE"):
                shared_hub /= "hub"
    config = Config(
        root=root, source=folder("source_folder", "Video Source"), output=folder("output_folder", "Transcribed"),
        work=folder("work_folder", "Work"), models=folder("models_folder", "Models"),
        # Explicit pre-existing HF cache choices remain external/shared exceptions.
        # models_folder takes precedence; the implicit home cache is import-only.
        shared_hub=shared_hub,
        source_language=normalize_language(values.get("source_language", "auto"), auto=True),
        target_language=normalize_language(values.get("target_language", "en")),
        asr_model=asr_model, asr_device=values.get("asr_device", "auto"),
        asr_compute_type=values.get("asr_compute_type", "auto"),
        translation_backend=backend,
        base_url=os.environ.get("TT_TRANSLATION_BASE_URL", api.get("base_url", "")).rstrip("/") if backend == "api" else "",
        model=os.environ.get("TT_TRANSLATION_MODEL", api.get("model", "")) if backend == "api" else "",
        api_key=(os.environ.get(key_env, api.get("api_key", "")) if key_env else api.get("api_key", "")) if backend == "api" else "",
        **settings,
    )
    protected = [config.source,config.output,config.work_root,config.models_root]
    if Path(config.asr_model).is_absolute():
        protected.append(Path(config.asr_model))
    if any(p.is_relative_to(config.installation_cache) or p.is_relative_to(root/".venv") for p in protected):
        raise ValueError("Models, recording/output and processing folders must be outside Work/Installation and .venv")
    return config
