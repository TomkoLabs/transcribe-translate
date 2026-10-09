from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import asr, translation
from .config import Config
from .media import decoded_audio
from .migration import migrate_output
from .publication import Publication
from .storage import StateError, digest, fingerprint, initialize_directory, job_directory, load, private_dir, save, state_directory, write_artifact


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Processor:
    def __init__(self, config: Config, *, transcriber=None, translator=None, decoder=decoded_audio, asr_settings=None, progress=print):
        self.config = config
        self.progress = progress
        self.state = state_directory(config.output, config.work_root)
        migrate_output(config.output, self.state, progress)
        self.transcriber = transcriber or asr.Transcriber(config, progress)
        self.translator = (translator or translation.Translator(config, progress=progress)) if config.translation_backend == "api" else None
        self.decoder = decoder
        self.asr_settings = asr_settings if asr_settings is not None else asr.settings(config)
        self.translated_path = None
        self.source_path = None

    def close(self):
        if self.translator is not None:
            self.translator.close()

    def process(self, path: Path, *, reprocess: str | None = None) -> tuple[str, Path]:
        self.translated_path = None
        self.source_path = None
        publication = Publication(self.config.output, path, self.config.target_language, state=self.state)
        identity = fingerprint(path)
        job = job_directory(self.config.output, path, self.config.work_root)
        cache_dir = job / "asr"
        private_dir(cache_dir)
        temporary = job / "temporary"
        private_dir(temporary)
        index_file = job / "index.json"
        index = load(index_file) if index_file.exists() else {"asr": {}, "translations": {}}
        if not isinstance(index.get("asr"), dict) or not isinstance(index.get("translations"), dict):
            raise StateError("Invalid checkpoint index; preserve it and choose a fresh output folder")
        # Mtime is provenance, while content hash + size establish reusable identity.
        signature = digest({"input": {k: identity[k] for k in ("sha256", "size")}, "asr": self.asr_settings})
        if reprocess == "all" or signature not in index["asr"]:
            index["asr"][signature] = digest({"signature": signature, "generation": uuid.uuid4().hex})
            save(index_file, index)
        cache_id = index["asr"][signature]
        if not isinstance(cache_id, str) or len(cache_id) != 64 or any(c not in "0123456789abcdef" for c in cache_id):
            raise StateError("Invalid ASR checkpoint identifier")
        cache = cache_dir / f"{cache_id}.json"
        if cache.exists():
            saved = load(cache)
            if saved.get("signature") != signature:
                raise StateError("ASR cache settings mismatch")
            result = saved["result"]
            asr.validate_result(result)
            self.progress("  Reusing saved source transcription")
        else:
            self.progress("  Decoding and transcribing")
            with self.decoder(path, temporary) as audio:
                result = self.transcriber.transcribe(audio)
            asr.validate_result(result)
            if fingerprint(path) != identity:
                raise StateError("Input changed during transcription; retry after copying/editing completes")
            save(cache, {"signature": signature, "input": identity, "settings": self.asr_settings, "result": result})
        source_text = asr.render_source(result["segments"])
        actual = result.get("actual_runtime", {})
        if actual:
            self.progress(f"  Actual ASR: {actual['model']} on {actual['device']}/{actual['compute_type']}")
        source_language = result["language"]
        same_language = source_language == self.config.target_language
        if same_language:
            translation_settings = {"target_language": source_language, "mode": "source-copy-v1"}
            if self.config.translation_backend == "local":
                translation_settings["backend"] = "local"
        elif self.config.translation_backend == "local":
            translation_settings = {"implementation": "whisper-speech-to-english-v2", "backend": "local",
                                    "target_language": "en", "task": "translate", "source_language": source_language,
                                    "without_timestamps": True,
                                    "model_settings": self.asr_settings}
        else:
            # Keep legacy API signatures compatible so valid 0.1.0 work can resume.
            translation_settings = translation.settings(self.config)
        run_signature = digest({"asr_cache": cache_id, "source_text": text_hash(source_text), "translation": translation_settings})
        if reprocess in {"all", "translation"} or run_signature not in index["translations"]:
            index["translations"][run_signature] = digest({"signature": run_signature, "generation": uuid.uuid4().hex})
            save(index_file, index)
        run_id = index["translations"][run_signature]
        if not isinstance(run_id, str) or len(run_id) != 64 or any(c not in "0123456789abcdef" for c in run_id):
            raise StateError("Invalid translation checkpoint identifier")
        backend_label = "local--" if self.config.translation_backend == "local" else ""
        run = job / f"{source_language}-to-{self.config.target_language}--{backend_label}{run_id[:16]}"
        record_file = run / "record.json"
        plan = translation.chunks(source_text, self.config.chunk_chars, self.config.context_chars) if not same_language and self.config.translation_backend == "api" else []
        source_name = f"source.{source_language}.txt"
        target_name = f"translated.{self.config.target_language}.txt"
        if run.exists() or run.is_symlink():
            if run.is_symlink() or not record_file.is_file():
                raise StateError(f"Unowned result folder: {run}")
            record = load(record_file)
            if record.get("signature") != run_signature or record.get("run_id") != run_id:
                raise StateError("Translation record settings mismatch")
        else:
            record = {"schema": 1, "signature": run_signature, "run_id": run_id, "input_path": str(path),
                "input": identity, "asr_cache_id": cache_id, "asr_settings": self.asr_settings,
                "source_language": source_language, "target_language": self.config.target_language,
                "language_probability": result.get("language_probability"), "source_segments": result["segments"],
                "actual_asr_runtime": result.get("actual_runtime", {}),
                "translation_backend": self.config.translation_backend,
                "translation_settings": translation_settings, "chunks": [], "chunk_count": len(plan),
                "source_characters": len(source_text), "status": "translating", "outputs": {source_name: text_hash(source_text)}}
            initialize_directory(run, "record.json", record)
        self._check_chunks(record, plan)
        # Detect edited artifacts before making any new inference request.
        for name, expected in record["outputs"].items():
            if name not in {source_name, target_name}:
                raise StateError("Invalid output name in checkpoint")
            p = run / name
            if p.exists() or p.is_symlink():
                from .storage import file_hash
                if p.is_symlink() or file_hash(p) != expected:
                    raise StateError(f"Transcript was edited: {p}; choose a fresh output folder")
        write_artifact(run / source_name, source_text, record["outputs"][source_name])
        plain_source = "\n".join(s["text"] for s in result["segments"]) + "\n"
        self.source_path = publication.publish(plain_source, language=source_language)
        self.progress(f"  Source TXT: {self.source_path}")
        if same_language:
            target_text = source_text
            plain_text = "\n".join(s["text"] for s in result["segments"]) + "\n"
        elif self.config.translation_backend == "local":
            try:
                native = record.get("native_translation")
                if native is None:
                    if record.get("status") == "complete":
                        raise StateError("Completed local record is missing its speech translation checkpoint")
                    self.progress("  Translating original audio to English locally (separate Whisper pass)")
                    with self.decoder(path, temporary) as audio:
                        native = self.transcriber.translate(audio, source_language)
                    asr.validate_result(native)
                    if native["language"] != "en":
                        raise StateError("Local translator did not produce an English result")
                    if fingerprint(path) != identity:
                        raise StateError("Input changed during speech translation; retry after copying/editing completes")
                    record["native_translation"] = native
                    record["status"] = "translating"
                    record.pop("error", None)
                    save(record_file, record)
                else:
                    asr.validate_result(native)
                    if native["language"] != "en":
                        raise StateError("Local speech checkpoint has an incompatible output language")
                    self.progress("  Reusing saved local speech translation")
                actual = native.get("actual_runtime", {})
                if actual:
                    self.progress(f"  Actual translation: {actual['model']} on {actual['device']}/{actual['compute_type']}")
                target_text = asr.render_source(native["segments"])
                plain_text = "\n".join(s["text"] for s in native["segments"]) + "\n"
            except (Exception, KeyboardInterrupt) as exc:
                record["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                record["error"] = type(exc).__name__
                save(record_file, record)
                raise
        else:
            if not self.config.base_url or not self.config.model:
                raise ValueError("Translation endpoint/model is missing; source transcript has been saved")
            for chunk in plan[len(record["chunks"]):]:
                self.progress(f"  Translating chunk {chunk.id + 1}/{len(plan)} (source chars {chunk.start}..{chunk.end})")
                try:
                    answer = self.translator.translate(chunk, source_language) if chunk.content.strip() else chunk.content
                    if not isinstance(answer, str) or (chunk.content.strip() and not answer.strip()):
                        raise translation.TranslationError("Empty translated chunk")
                    record["chunks"].append({"id": chunk.id, "start": chunk.start, "end": chunk.end,
                        "source_sha256": text_hash(chunk.content), "translation": answer})
                    record["status"] = "translating"
                    record.pop("error", None)
                    save(record_file, record)
                except (Exception, KeyboardInterrupt) as exc:
                    record["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                    # Error category only; endpoint bodies, reasoning, and secrets never enter records.
                    record["error"] = type(exc).__name__
                    save(record_file, record)
                    raise
            self._check_chunks(record, plan)
            target_text = "\n\n".join(c["translation"] for c in record["chunks"]) + "\n"
            plain_text = target_text
        was_complete = record["status"] == "complete"
        expected_target = text_hash(target_text)
        if target_name in record["outputs"] and record["outputs"][target_name] != expected_target:
            raise StateError("Target output checksum disagrees with translation chunks")
        record["outputs"][target_name] = expected_target
        # Publish intent before TXT: a crash here can regenerate from saved chunks.
        save(record_file, record)
        write_artifact(run / target_name, target_text, expected_target)
        record["status"] = "complete"
        record.pop("error", None)
        record.setdefault("completed_at", datetime.now(timezone.utc).isoformat())
        save(record_file, record)
        self.translated_path = publication.publish(plain_text)
        self.progress(f"  Translated TXT: {self.translated_path}")
        return ("skipped" if was_complete else "completed"), run

    @staticmethod
    def _check_chunks(record: dict, plan: list[translation.Chunk]) -> None:
        saved = record.get("chunks")
        if not isinstance(saved, list) or len(saved) > len(plan) or record.get("chunk_count") != len(plan):
            raise StateError("Invalid translation checkpoint coverage")
        for c, p in zip(saved, plan):
            if not isinstance(c, dict) or (c.get("id"), c.get("start"), c.get("end"), c.get("source_sha256")) != (p.id, p.start, p.end, text_hash(p.content)) or not isinstance(c.get("translation"), str) or (not c["translation"].strip() and (p.content.strip() or c["translation"] != p.content)):
                raise StateError("Translation checkpoint has missing, duplicated or edited chunks")
        if record.get("status") == "complete" and len(saved) != len(plan):
            raise StateError("Complete record is missing translation chunks")
