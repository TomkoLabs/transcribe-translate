from __future__ import annotations

import argparse
import os
import site
import shutil
import sys
from pathlib import Path

from . import __version__, asr
from .config import installation_root, load_config
from .media import discover
from .pipeline import Processor
from .migration import migrate_output
from .translation import Translator
from .storage import batch_lock, state_directory
from .installation import installation_lock, verify
from . import models


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Transcribe videos sequentially, translate to English locally or through an API, and resume saved work")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--installation-info", action="store_true", help="Verify version, imported source and canonical project-local runtime paths; no inference")
    p.add_argument("--config", type=Path, help="TOML configuration (relative folders still use installation root)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--source", type=Path, help="Source folder (explicit relative paths use current directory)")
    group.add_argument("--input", type=Path, help="Process one file")
    p.add_argument("--output", type=Path, help="Output folder")
    p.add_argument("--work", type=Path, help="Visible recovery/processing folder (default: Work beside the launcher)")
    p.add_argument("--models", type=Path, help="Selected weights folder; relative CLI paths use caller directory")
    p.add_argument("--source-language", help="auto, Mandarin/Chinese/zh, English/en, etc.")
    p.add_argument("--target-language", help="Target language (default en)")
    p.add_argument("--translation-backend", choices=["local", "api"], help="Override translation.backend for this run")
    p.add_argument("--asr-model", help="Multilingual model name or converted model directory")
    p.add_argument("--asr-device", choices=["auto", "cpu", "cuda"])
    p.add_argument("--asr-compute-type", help="Normally int8 for CPU; float16/int8_float16 for CUDA")
    p.add_argument("--recursive", action="store_true", help="Scan source subfolders; default is top-level only")
    p.add_argument("--reprocess", nargs="?", const="all", choices=["all", "translation"], help="Create a fresh preserved generation (all or translation only)")
    p.add_argument("--prepare-model", action="store_true", help="Download/reuse only the requested ASR weights, then exit")
    p.add_argument("--check", action="store_true", help="Check configuration, dependencies, runtime and disk without inference")
    api_group = p.add_mutually_exclusive_group()
    api_group.add_argument("--api-models", action="store_true", help="List authenticated API model aliases without inference; model selection not required")
    api_group.add_argument("--api-check", action="store_true", help="Explicit API-only alias discovery and bounded authored text translation")
    p.add_argument("--cuda-smoke-check", action="store_true", help="Execute bounded CUDA encoder/decoder kernels using cached weights; no audio, API or model downloads")
    return p


def main(argv=None) -> int:
    try:
        with installation_lock(installation_root()):
            return _main(argv)
    except (OSError, RuntimeError) as exc:
        print(f"Installation error: {exc}", file=sys.stderr)
        return 2


def _main(argv=None) -> int:
    # Installed console scripts need the same pre-start CUDA library path as the
    # checkout launcher. Reexec before importing any native inference libraries.
    cuda_dirs = [str(p) for root in site.getsitepackages() for p in (Path(root) / "nvidia/cublas/lib", Path(root) / "nvidia/cudnn/lib") if p.is_dir()]
    existing = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if any(p not in existing for p in cuda_dirs):
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = ":".join(cuda_dirs + [p for p in existing if p])
        os.execve(sys.executable, [sys.executable, "-I", "-B", "-m", "transcribe_translate.cli", *(sys.argv[1:] if argv is None else argv)], env)
    args = parser().parse_args(argv)
    if args.installation_info:
        try:
            for key, value in verify(installation_root()).items():
                print(f"{key}: {value}")
            return 0
        except (OSError, ValueError) as exc:
            print(f"Installation error: {exc}", file=sys.stderr)
            return 2
    processor = None
    config = None
    try:
        overrides = {k: getattr(args, k) for k in ("source_language", "target_language", "asr_model", "asr_device", "asr_compute_type", "translation_backend")}
        if args.cuda_smoke_check:
            if args.asr_device == "cpu" or args.api_check or args.api_models or args.prepare_model or args.check:
                raise ValueError("Use --cuda-smoke-check separately, with CUDA (not CPU/other check modes)")
            overrides.update(asr_device="cuda", translation_backend="local", target_language="en")
        for key, arg in [("source_folder", args.source), ("output_folder", args.output), ("work_folder", args.work), ("models_folder", args.models)]:
            if arg is not None:
                overrides[key] = str(arg.expanduser().resolve())
        config = load_config(args.config, overrides)
        if not args.cuda_smoke_check:
            config.validate(endpoint=not (args.prepare_model or args.api_models))
        if args.api_check or args.api_models:
            flag = "--api-models" if args.api_models else "--api-check"
            if config.translation_backend != "api":
                raise ValueError(f"{flag} requires translation.backend = \"api\" or --translation-backend api")
            if not config.base_url or (args.api_check and not config.model):
                raise ValueError(f"Configure translation.base_url{' and translation.model' if args.api_check else ''} for {flag}")
            translator = Translator(config)
            try:
                if args.api_models:
                    aliases = translator.models()
                    print("Available aliases:\n" + "\n".join(aliases) if aliases else "No exposed model aliases")
                else:
                    print("Bounded authored text translation: " + translator.check())
            finally:
                translator.close()
            return 0
        models.configure_transfer(config)
        config = asr.runtime_check(config)
        if args.cuda_smoke_check:
            actual = asr.cuda_smoke_check(config)
            print(f"CUDA smoke passed: {actual['model']} on {actual['device']}/{actual['compute_type']}; encoder + {', '.join(actual['tasks'])} decoders. Runtime execution only, not speech quality")
            return 0
        disk_root = config.output
        while not disk_root.exists():
            disk_root = disk_root.parent
        free = shutil.disk_usage(disk_root).free / 1024 ** 3
        print(f"ASR: {config.asr_model}, {config.asr_device}/{config.asr_compute_type}; source: {config.source_language}; target: {config.target_language}", flush=True)
        print(f"Translation backend: {config.translation_backend}" + (f"; API model: {config.model}" if config.translation_backend == "api" else "; native Whisper speech-to-English"), flush=True)
        print(f"Disk available: {free:.1f} GiB; outputs: {config.output}", flush=True)
        if free < 1:
            raise ValueError("Less than 1 GiB free; free space for weights, decoded audio and outputs before continuing")
        if args.check:
            print(models.check(config))
            print("Model storage: " + models.locations(config)["models"] + "; installation cache: " + str(config.installation_cache))
            print("Configuration and local runtime checks passed; model inference/endpoint access were not tested")
            return 0
        if args.prepare_model:
            print(f"ASR weights ready: {asr.prepare_model(config)}")
            return 0
        if args.input:
            paths = [args.input.expanduser().resolve()]
            if not paths[0].is_file():
                raise ValueError(f"Input does not exist or is unreadable: {paths[0]}")
        else:
            paths = discover(config.source, config.output, recursive=args.recursive, work=config.work_root)
        counts = {"completed": 0, "skipped": 0, "failed": 0}
        with batch_lock(config.output, config.work_root):
            migrate_output(config.output, state_directory(config.output, config.work_root), progress=lambda s: print(s, flush=True))
            if not paths:
                print(f"No supported videos found in {config.source}. Place videos there and rerun. Discovery is {'recursive' if args.recursive else 'top-level only'}.")
                return 0
            processor = Processor(config, progress=lambda s: print(s, flush=True))
            for i, path in enumerate(paths, 1):
                print(f"[{i}/{len(paths)}] {path.name}", flush=True)
                try:
                    status, _run = processor.process(path, reprocess=args.reprocess)
                    counts[status] += 1
                    print(f"  {status.capitalize()}: {processor.translated_path}", flush=True)
                except Exception as exc:
                    counts["failed"] += 1
                    message = str(exc).replace(config.api_key, "[redacted]") if config.api_key else str(exc)
                    print(f"  FAILED {path.name}: {type(exc).__name__}: {message}", file=sys.stderr, flush=True)
        print(f"Batch finished: {counts['completed']} completed, {counts['skipped']} skipped, {counts['failed']} failed", flush=True)
        return 1 if counts["failed"] else 0
    except KeyboardInterrupt:
        print("Interrupted. Saved source and completed translation checkpoints will resume on the next run.", file=sys.stderr)
        return 130
    except Exception as exc:
        message = str(exc).replace(config.api_key, "[redacted]") if config and config.api_key else str(exc)
        print(f"Error: {type(exc).__name__}: {message}", file=sys.stderr)
        return 2
    finally:
        if processor:
            processor.close()


if __name__ == "__main__":
    raise SystemExit(main())
