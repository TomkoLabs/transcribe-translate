"""Small dependency-free installer preflight; never modifies system drivers."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

# CUDA 12 minor compatibility and cuDNN 9.10.2's pre-Blackwell matrix.
CUDA12_MIN_DRIVER = (525, 60, 13)
BLACKWELL_MIN_DRIVER = (570, 26, 0)
PRE_BLACKWELL_CAPABILITIES = {(5, 0), (5, 2), (6, 0), (6, 1), (7, 0), (7, 5),
                             (8, 0), (8, 6), (8, 9), (9, 0)}


def cuda_candidate(driver: str, capability: str) -> bool:
    version = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?", driver.strip())
    compute = re.fullmatch(r"(\d+)\.(\d+)", capability.strip())
    if not version or not compute:
        return False
    version = tuple(int(v or 0) for v in version.groups())
    compute = tuple(int(v) for v in compute.groups())
    if compute in PRE_BLACKWELL_CAPABILITIES:  # Maxwell through Hopper; kernels still need testing.
        return version >= CUDA12_MIN_DRIVER
    if compute in {(10, 0), (12, 0)}:  # Blackwell needs the newer driver path.
        return version >= BLACKWELL_MIN_DRIVER
    return False


def choose_runtime(*, cpu_only=False, require_cuda=False, find=shutil.which, run=subprocess.run, log=print):
    if cpu_only and require_cuda:
        raise ValueError("Choose --cpu or --cuda, not both")
    if cpu_only:
        log("CPU-only installation requested; CUDA packages will not be installed.")
        return "cpu"
    executable = find("nvidia-smi")
    reason = "NVIDIA driver tool nvidia-smi is unavailable"
    if executable:
        try:
            result = run([executable, "--query-gpu=driver_version,compute_cap", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=10, check=False)
            devices = list(csv.reader(result.stdout.strip().splitlines()))
            if result.returncode == 0 and devices:
                if any(len(row) == 2 and cuda_candidate(*row) for row in devices):
                    log("Eligible NVIDIA driver/GPU detected; installing pinned isolated CUDA 12/cuDNN 9 libraries. "
                        "Minor compatibility eligibility does not establish kernel execution.")
                    return "cuda"
                reason = ("No eligible GPU/driver: CUDA 12 requires Linux driver 525.60.13+ for "
                          "pre-Blackwell compute capability 5.x–9.x; Blackwell 10.0/12.0 requires 570.26+. "
                          "Unknown/unsupported capabilities are not accepted")
            else:
                reason = "nvidia-smi could not identify a usable NVIDIA driver/device"
        except (OSError, subprocess.TimeoutExpired):
            reason = "NVIDIA driver detection failed or timed out"
    message = reason + ". Install/fix the system NVIDIA driver and rerun setup for CUDA; no driver is installed by this script."
    if require_cuda:
        raise ValueError(message)
    log(message + " Installing CPU dependencies; the selected ASR model is preserved.")
    return "cpu"


def preflight(root: Path, *, cpu_only=False, require_cuda=False):
    if not (3, 11) <= sys.version_info[:2] < (3, 14):
        raise ValueError("Python 3.11–3.13 required; use --python with a supported interpreter")
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise ValueError("This installer supports Linux x86-64; other OS/architectures are not accepted for this release")
    for module in ("venv", "ensurepip"):
        if importlib.util.find_spec(module) is None:
            raise ValueError("Python venv/pip support is missing; on Debian/Ubuntu install python3-venv")
    if shutil.which('flock') is None:
        raise ValueError('Linux flock is required; on Debian/Ubuntu install util-linux')
    free = shutil.disk_usage(root).free / 1024 ** 3
    from transcribe_translate.config import load_config
    config = load_config(overrides={'translation_backend':'local','target_language':'en'})
    required = 2 if config.shared_hub or not config.models_root.is_relative_to(root) or (config.models_root.exists() and any(config.models_root.glob('Whisper-large-v3--*/model.bin'))) else 6
    print(f"Available project disk: {free:.1f} GiB; required preflight reserve: {required} GiB (plus recordings/decoded audio).", file=sys.stderr)
    if free < required:
        raise ValueError(f"Free at least {required} GiB before installing; model storage checks run before weight import/download")
    return choose_runtime(cpu_only=cpu_only, require_cuda=require_cuda, log=lambda s: print(s, file=sys.stderr))


def create_defaults(root: Path, *, cpu_only=False):
    for name in ("Video Source", "Transcribed", "Work"):
        (root / name).mkdir(mode=0o700, exist_ok=True)
    config = root / "config.local.toml"
    try:
        fd = os.open(config, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        print("Existing config.local.toml preserved byte-for-byte")
        return
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        example = (root / "config.example.toml").read_text(encoding="utf-8")
        if cpu_only:
            example = example.replace('asr_device = "auto"', 'asr_device = "cpu"')
        f.write(example)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["preflight", "defaults"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--cuda", action="store_true")
    args = p.parse_args()
    try:
        if args.action == "preflight":
            print(preflight(args.root, cpu_only=args.cpu, require_cuda=args.cuda))
        else:
            create_defaults(args.root, cpu_only=args.cpu)
    except (ValueError, OSError) as exc:
        print(f"Setup prerequisite error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
