from __future__ import annotations

import importlib.metadata
import math
from dataclasses import replace
from pathlib import Path

from .config import Config
from .languages import normalize_language
from .storage import file_hash

LARGE_V3_REPO = "Systran/faster-whisper-large-v3"
LARGE_V3_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
MODEL_FILES = ["model.bin", "config.json", "preprocessor_config.json", "tokenizer.json", "vocabulary.json"]


def cuda_unavailable(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return any(s in message for s in ("libcublas", "libcudnn", "cuda driver version is insufficient", "no cuda-capable device", "cuda failed with error out of memory"))


def runtime_check(config: Config, progress=print) -> Config:
    try:
        import av
        import ctranslate2
        import faster_whisper
    except (ImportError, OSError) as exc:
        raise RuntimeError("Missing or incompatible decoding/ASR dependencies; rerun setup.sh in this project") from exc
    try:
        count = ctranslate2.get_cuda_device_count() if config.asr_device in {"auto", "cuda"} else 0
    except RuntimeError as exc:
        if config.asr_device != "auto" or not cuda_unavailable(exc):
            raise
        count = 0
    if config.asr_device == "cuda" and count == 0:
        raise ValueError("CUDA selected but no visible NVIDIA device; configure cpu/int8 or install a compatible driver")
    device = "cuda" if config.asr_device == "auto" and count > 0 else ("cpu" if config.asr_device == "auto" else config.asr_device)
    try:
        supported = ctranslate2.get_supported_compute_types(device)
    except RuntimeError as exc:
        if config.asr_device != "auto" or device != "cuda" or not cuda_unavailable(exc):
            raise
        device = "cpu"
        supported = ctranslate2.get_supported_compute_types(device)
    if not supported and config.asr_device == "auto" and device == "cuda":
        device = "cpu"
        supported = ctranslate2.get_supported_compute_types(device)
    compute = config.asr_compute_type
    if compute == "auto":
        preferred = ["float16", "int8_float16", "float32"] if device == "cuda" else ["int8", "int8_float32", "float32"]
        compute = next((c for c in preferred if c in supported), "")
    if compute not in supported:
        raise ValueError(f"Compute type {compute} is unsupported on {device}; supported: {', '.join(sorted(supported))}")
    if config.asr_device == "auto" and device == "cpu":
        progress(f"No usable supported CUDA GPU selected; using CPU/{compute} with {config.asr_model}")
    return replace(config, asr_device=device, asr_compute_type=compute,
                   requested_device=config.asr_device, requested_compute_type=config.asr_compute_type)


def model_identity(config: Config) -> dict:
    if config.asr_model in {"large-v3", "large"}:
        return {"repository": LARGE_V3_REPO, "revision": LARGE_V3_REVISION}
    p = Path(config.asr_model)
    if p.is_dir():
        return {"path": str(p), "files": {f: file_hash(p / f) for f in MODEL_FILES if (p / f).is_file()}}
    # For optional named models, resolve the actual snapshot, reusing the hub cache.
    resolved = prepare_model(config)
    return {"name": config.asr_model, "snapshot": resolved.name, "model_sha256": file_hash(resolved / "model.bin")}


def settings(config: Config) -> dict:
    return {
        "implementation": "asr-v1", "model": config.asr_model, "weights": model_identity(config),
        "device": config.asr_device, "compute_type": config.asr_compute_type,
        "source_language": config.source_language, "task": "transcribe", "beam_size": 5,
        "requested_device": config.requested_device, "requested_compute_type": config.requested_compute_type,
        "vad_filter": False, "condition_on_previous_text": True, "cpu_threads": 4,
        "versions": {p: importlib.metadata.version(p) for p in ("faster-whisper", "ctranslate2", "av", "onnxruntime", "tokenizers")},
        "decoder": "first-audio-track/mono-pcm16-16000-v1",
    }


def prepare_model(config: Config, *, local_files_only: bool = False) -> Path:
    from .models import prepare
    return prepare(config, local_files_only=local_files_only)


def cuda_smoke_check(config: Config) -> dict:
    """Bounded encoder/decoder execution on synthetic features; no audio or downloads."""
    import ctranslate2
    import numpy as np
    if config.asr_device != "cuda":
        raise ValueError("CUDA smoke requires explicit CUDA; CPU fallback is not accepted")
    weights = prepare_model(config, local_files_only=True)
    try:
        model = ctranslate2.models.Whisper(str(weights), device="cuda", compute_type=config.asr_compute_type, intra_threads=4)
        if model.device != "cuda":
            raise RuntimeError("Loaded engine is not running on CUDA")
        features = ctranslate2.StorageView.from_array(np.zeros((1, model.n_mels, 3000), dtype=np.float32))
        encoded = model.encode(features)
        tasks = ("transcribe", "translate") if model.is_multilingual else ("transcribe",)
        for task in tasks:
            prompt = ["<|startoftranscript|>"]
            if model.is_multilingual:
                prompt += ["<|zh|>", f"<|{task}|>"]
            prompt += ["<|notimestamps|>"]
            result = model.generate(encoded, [prompt], beam_size=1, max_length=6)
            if len(result) != 1:
                raise RuntimeError("CUDA decoder did not return the expected batch")
        return {"device": model.device, "compute_type": model.compute_type, "model": config.asr_model, "tasks": tasks}
    except RuntimeError as exc:
        raise RuntimeError(f"CUDA smoke failed; no CPU fallback. Check pinned libraries, GPU memory and driver/PTX compatibility: {exc}") from exc


def render_source(segments: list[dict]) -> str:
    return "\n".join(("[uncertain] " if s.get("uncertain") else "") + s["text"] for s in segments) + "\n"


def validate_result(result: dict) -> None:
    try:
        normalize_language(result["language"])
        if not result["segments"]:
            raise ValueError("empty speech")
        previous = 0.0
        duration = result.get("duration_seconds")
        if duration is not None and (not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0):
            raise ValueError("invalid audio duration")
        for s in result["segments"]:
            if not isinstance(s["text"], str) or not s["text"].strip():
                raise ValueError("empty segment")
            if not all(isinstance(s[k], (int, float)) and math.isfinite(s[k]) for k in ("start", "end")):
                raise ValueError("invalid timestamp")
            if s["start"] < previous or s["end"] < s["start"]:
                raise ValueError("invalid timestamp order")
            if duration is not None and s["end"] > duration + 0.5:
                raise ValueError("segment extends beyond decoded audio")
            previous = s["start"]
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Invalid speech segments; preserve saved work and use a fresh output folder or reprocess the affected pass") from exc


class Transcriber:
    def __init__(self, config: Config, progress=print):
        self.config = config
        self.progress = progress
        self.model = None

    def _load(self):
        from faster_whisper import WhisperModel
        return WhisperModel(str(prepare_model(self.config)), device=self.config.asr_device,
                            compute_type=self.config.asr_compute_type, cpu_threads=4)

    def _fallback(self, exc: RuntimeError) -> bool:
        if self.config.requested_device != "auto" or self.config.asr_device != "cuda" or not cuda_unavailable(exc):
            return False
        compute = "int8" if self.config.requested_compute_type == "auto" else self.config.requested_compute_type
        cpu = replace(self.config, asr_device="cpu", asr_compute_type=compute)
        runtime_check(cpu)
        self.config = cpu
        self.progress(f"CUDA runtime unavailable ({type(exc).__name__}); preserving {cpu.asr_model}, falling back to CPU/{compute}")
        self.model = None
        return True

    def transcribe(self, audio: Path) -> dict:
        return self._run(audio, task="transcribe")

    def translate(self, audio: Path, source_language: str) -> dict:
        """A separate speech-to-English pass on audio, sharing the loaded model."""
        return self._run(audio, task="translate", source_language=source_language)

    def _run(self, audio: Path, *, task: str, source_language: str | None = None) -> dict:
        if self.model is None:
            self.progress(f"Loading {self.config.asr_model} on {self.config.asr_device}/{self.config.asr_compute_type}")
            try:
                self.model = self._load()
            except RuntimeError as exc:
                if not self._fallback(exc):
                    raise RuntimeError(f"ASR model load failed on {self.config.asr_device}; verify weights and CUDA 12/cuDNN 9 runtime if using CUDA: {exc}") from exc
                self.model = self._load()
        try:
            return self._decode(audio, task=task, source_language=source_language)
        except RuntimeError as exc:
            if not self._fallback(exc):
                raise
            self.model = self._load()
            return self._decode(audio, task=task, source_language=source_language)

    def _decode(self, audio: Path, *, task: str = "transcribe", source_language: str | None = None) -> dict:
        extra = {"without_timestamps": True} if task == "translate" else {}
        stream, info = self.model.transcribe(
            str(audio), language=source_language or (None if self.config.source_language == "auto" else self.config.source_language),
            task=task, beam_size=5, vad_filter=False, condition_on_previous_text=True,
            **extra,
        )
        segments = []
        self.progress(f"Source language: {info.language} (probability {info.language_probability:.3f})")
        for s in stream:
            if s.text.strip():
                segments.append({"start": s.start, "end": s.end, "text": s.text.strip(),
                                 "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
                                 "uncertain": s.avg_logprob < -1.0 or s.no_speech_prob > 0.6})
                label = "ASR" if task == "transcribe" else "Whisper translation"
                self.progress(f"  {label}: {s.end:.1f}s / {info.duration:.1f}s")
        if not segments:
            raise RuntimeError("No intelligible speech detected; check the audio track and source language")
        result = {"language": "en" if task == "translate" else normalize_language(info.language), "language_probability": info.language_probability,
                  "duration_seconds": info.duration, "segments": segments,
                  "actual_runtime": {"device": self.model.model.device, "compute_type": self.model.model.compute_type, "model": self.config.asr_model}}
        if task == "translate":
            result["source_language"] = normalize_language(info.language)
        validate_result(result)
        return result
