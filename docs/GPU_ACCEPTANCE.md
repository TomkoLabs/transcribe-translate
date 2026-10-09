# Manual Linux NVIDIA acceptance

## Reported runtime execution and outstanding installation

The user reports successful execution of the previous runtime on Debian 12,
Linux x86-64, RTX 3080 10 GB, NVIDIA driver 550.163.01, CTranslate2 4.7.2,
faster-whisper 1.2.1, cuBLAS 12.8.4.1 and cuDNN 9.10.2.21. A 594.5-second
Mandarin video completed source transcription and native English translation using
multilingual Whisper large-v3, both on CUDA/float16: 1 completed, 0 failed.

That run followed manual installation of `requirements-cuda.lock`, because the
previous setup's 570.26 threshold had installed only CPU dependencies. It is
**user-reported hardware validation of that runtime**, not an independently
reproduced result or proof of fresh installation through current setup (driver
eligibility corrected in 0.2.2, installation identity/rebuild in 0.2.6, local
models/cleanup in 0.2.7).
Fresh installation through 0.2.7 remains unverified on this hardware.
No private filename, recording contents, personal paths or raw logs are published.
The corrected installer and new GPU smoke path remain to be confirmed on that
hardware. This environment has no GPU/access to it; no remote GPU probes occurred.
Runtime completion does not reverse the documented semantic fidelity failures.

## Compatibility policy

The system NVIDIA driver must already work. The installer queries driver and GPU
compute capability. Known pre-Blackwell capabilities 5.x–9.x are eligible at Linux **525.60.13+**;
Blackwell 10.0/12.0 retains **570.26+**. Unknown or older architectures/versions and
failed detection select CPU; explicit `--cuda` setup fails clearly. At least one
queried GPU must be eligible; actual CUDA visibility/selected device is checked
at runtime. Pinned cuBLAS/cuDNN dependencies are unchanged.

[NVIDIA's CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)
distinguish the 570.26 toolkit driver from the 525.60.13 minor-compatibility floor.
The [cuDNN 9.10.2 matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/reference/support-matrix.html)
includes the older-driver path on pre-Blackwell hardware. Under
[minor compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html),
PTX JIT and features requiring newer driver APIs may still fail; compatible target
architecture kernels are required. These checks allow installation candidates,
not every CUDA 12 application or kernel on every older driver. Drivers newer than
the CUDA 12 minor-compatibility range remain eligible via backward compatibility.

Setup installs isolated project libraries, never a system driver/global CUDA.
Python 3.11–3.13 with venv/pip and
`libgomp1` are prerequisites; Python 3.11 is the actual CPU test baseline. Follow
your supported distro procedure if the driver is missing or too old.

## Verify corrected setup without recordings

Use a separate clean clone/directory for **fresh-installer** validation; do not
delete a working environment or its results. Existing compatible model caches can
be imported into Models without downloading; the original shared copy remains.
Explicit external model storage is retained. Installer/transfer caches use
Work/Installation; see [storage](STORAGE.md). From the updated project directory:

```bash
pwd -P
readlink -f ./setup.sh
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
./setup.sh
./transcribe-translate --installation-info
.venv/bin/python -m pip show ctranslate2 faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12
./transcribe-translate --check --asr-device cuda --asr-compute-type float16
./transcribe-translate --cuda-smoke-check --asr-model large-v3 --asr-compute-type float16
```

For an existing moved/stale runtime, `./setup.sh --reinstall` explicitly rebuilds
at the final path while preserving data and retaining recoverable runtimes. See
[installation details](INSTALLATION.md). Confirm version 0.2.7 and all identity paths
belong to the intended canonical checkout; an alias to old source still selects old source.
Expect eligible NVIDIA detection, installation of both pinned CUDA packages, and
`CUDA smoke passed: large-v3 on cuda/float16`. Normal CUDA setup runs smoke after
weight preparation; `--skip-model-downloads` defers that automatic check. The
explicit smoke uses only cached/local weights and errors if they are unavailable.
It runs one encoder window of zero synthetic features and at most six decoder
tokens per task, with no recording, API request or transcript/checkpoint writes.
The default multilingual model exercises both task decoders; an explicitly selected
English-only model exercises its transcription decoder. CPU fallback is forbidden.
This is bounded kernel execution, not a speech-quality benchmark or full coverage.

Detection and `--check` test configuration/capability. Smoke tests actual kernels.
Neither substitutes for source/native acceptance on permitted known-reference speech.
Save the setup/smoke console result locally for fresh-installer verification.

## Optional speech and API acceptance

These commands are provided for later use; they were not run on GPU here. Use a
short permitted Mandarin recording with an independently checked reference:

```bash
# Replace with your short permitted non-English recording:
GPU_TEST_VIDEO='/absolute/path/to/mandarin-video.mp4'
GPU_ACCEPTANCE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/transcribe-gpu.XXXXXX")"
printf 'Keep this acceptance directory: %s\n' "$GPU_ACCEPTANCE_DIR"
./transcribe-translate --input "$GPU_TEST_VIDEO" --output "$GPU_ACCEPTANCE_DIR/Transcribed" \
  --work "$GPU_ACCEPTANCE_DIR/Work" \
  --translation-backend local --source-language zh --target-language en \
  --asr-model large-v3 --asr-device cuda --asr-compute-type float16
```

The fresh directory excludes old CPU/checkpoint reuse. Explicit CUDA must fail if
unusable. Expect `Actual ASR: large-v3 on cuda/float16` and `Actual translation:
large-v3 on cuda/float16`, with a new source and independent native English result.
An empty batch or `--check` alone is not acceptance. Assert saved runtime facts:

```bash
.venv/bin/python - "$GPU_ACCEPTANCE_DIR" <<'PY'
import json, sys
from pathlib import Path
records = [json.loads(p.read_text())['payload']
           for p in (Path(sys.argv[1]) / 'Work').rglob('record.json')]
native = [r for r in records if r.get('translation_backend') == 'local']
assert len(native) == 1 and native[0]['status'] == 'complete'
record = native[0]
assert record['source_language'] == 'zh'
for runtime in (record['actual_asr_runtime'],
                record['native_translation']['actual_runtime']):
    assert runtime == {'device': 'cuda', 'compute_type': 'float16', 'model': 'large-v3'}, runtime
print('Both native speech passes recorded CUDA/float16/large-v3')
PY
```

If float16 exhausts VRAM, repeat the inference and assertion sequence using a new
directory and `int8_float16` everywhere. Keep both attempts; this is a separately
reported compute choice, not evidence that float16 passed. No smaller model is
required. Check both TXT files against the recording/reference; runtime assertions
cannot establish semantic quality. Native fidelity limitations still apply.

For optional API acceptance, the GPU deployment machine must itself reach the
gateway. Prepare ignored `config.api.local.toml` from the public example using a
reachable URL, exact authorized alias and the key environment variable. Development
host connectivity is not deployment connectivity. Then run:

```bash
./transcribe-translate --config config.api.local.toml --api-models
./transcribe-translate --config config.api.local.toml --api-check
./transcribe-translate --config config.api.local.toml \
  --input "$GPU_TEST_VIDEO" --output "$GPU_ACCEPTANCE_DIR/Transcribed" \
  --work "$GPU_ACCEPTANCE_DIR/Work" \
  --translation-backend api --source-language zh --target-language en \
  --asr-model large-v3 --asr-device cuda --asr-compute-type float16
```

This deliberately reuses the just-tested CUDA source transcription, and writes a
separate API generation. Inspect that generation's completed `record.json`, chunk
coverage, source TXT and API English. Compare API output with saved source and
with the recording separately. API success does not prove additional GPU kernels
ran or that ASR errors were fixed. Keep driver output, actual runtime messages,
records and qualitative findings with the isolated directory.
