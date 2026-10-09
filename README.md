# TLDR

1. Run **`./setup.sh`**.
2. Place videos in **`Video Source/`**.
3. Run **`./transcribe-translate`**.
4. Open the source and translated TXT files directly in **`Transcribed/`**.

For example, French speech in `Interview.mp4` produces **`Interview.fr.txt`** and
**`Interview.en.txt`** directly in `Transcribed/`. Automatic source-language
detection is the default. Processing is entirely local with multilingual **Whisper large-v3**;
no API key/server is required. Reruns resume saved work and skip valid completed
inputs. OpenAI-compatible text translation is an optional backend.

**Quality limit:** native English translation made material omissions and invented
meaning in our bounded speech tests. Optional API translation preserved saved text
more faithfully, but inherits recognition errors. No released path has passed full
end-to-end fidelity acceptance. Review valuable recordings; see
[the measured quality follow-up](docs/QUALITY_FOLLOWUP.md).

## Quick start

Prerequisites: **Linux x86-64**, Python **3.11–3.13** with `venv`/`pip`, and free disk
for the runtime, approximately 3 GB of weights in `Models/`, and your recordings.
Python 3.11 is our tested baseline. On Debian/Ubuntu, install missing prerequisites first:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv libgomp1 util-linux
```

For GPU execution, a supported NVIDIA system driver must already work. Setup
checks it and automatically installs project CUDA libraries; it never installs
a driver. CPU execution is available when a supported NVIDIA GPU is unavailable.

In the downloaded/cloned project:

```bash
cd transcribe-translate
./setup.sh
# Place videos in "Video Source" using your file manager, then:
./transcribe-translate
```

For French speech in `Interview.mp4`, open **`Transcribed/Interview.fr.txt`** for
the source and **`Transcribed/Interview.en.txt`** for English. The command prints
both direct paths. Each file contains its corresponding UTF-8 transcript text.
No config editing is needed for automatic language detection → English.
Setup preserves existing configuration and recordings. It imports matching cached
weights or downloads them into `Models/`; setup reports which occurred.

## Updating, reinstalling and moving

After updating the current checkout/download to the new release, run **`./setup.sh`**
again. It refreshes the current source even at the same version, reuses matching
pinned dependencies and cached weights, and preserves existing configuration and data.
Setup installs the files present; it never pulls Git or replaces source. For a Git clone:

```bash
git status --short
git fetch origin
git switch main
git pull --ff-only origin main
./setup.sh
./transcribe-translate --installation-info
```

Preserve local source changes before switching/updating. To rebuild the runtime
explicitly, use **`./setup.sh --reinstall`**. Configuration, credentials, recordings,
TXT files, the entire Work folder and model caches remain. Setup creates `.venv`
at its final location, retains the old runtime as an ignored `.venv.backup-*`, and
restores it at `.venv` if rebuilding fails. Failed replacements remain as
`.venv.failed-*` for inspection. A restored moved/broken runtime may still need
repair; fix the reported error and rerun setup. Stop processing before setup;
active processing/setup makes it fail clearly without replacing the runtime.

For relocation, move/copy the **whole project including configuration, Video Source,
Transcribed and Work**, then run `./setup.sh` at the destination. Copied/moved
virtual environments are detected and rebuilt automatically. Verified copied Work
state for project-relative output/Work paths gets a new namespace with relocated
relative input identities; all generations and selected indexes are retained,
and the old namespace stays unchanged. Run `./transcribe-translate` to resume or
rebuild missing TXT without repeating compatible completed inference. Explicit
absolute paths remain exactly configured. Legacy layouts, interrupted migrations,
external Work/output folders or ambiguous/edited state require preserving the
original paths and finishing recovery there first; see [installation details](docs/INSTALLATION.md).

Symlinked access reports the checkout's **canonical physical root**. Confirm that
root before updating: an alias to an older checkout still selects that older checkout.
The absolute launcher and `.venv/bin/transcribe-translate` work from other directories,
including paths with spaces, without activating a virtual environment.
`--installation-info` verifies version, root, interpreter, imported source and
launcher locations without model execution. It should report **0.2.7** for this
release and show model/cache locations.

## Models, caches and uninstall

Selected speech weights live in **`Models/` beside the launcher**. The large-v3
revision is fixed. Setup, prefetch, inference and smoke checks use the same storage;
a path change alone does not reset compatible saved results. Matching legacy
Hugging Face weights are verified and copied without downloading. Snapshot links
are dereferenced, so the local copy works independently. The old shared copy stays
unchanged and uses additional disk space. LiteLLM translation models stay on their servers.

Disposable pip and Hugging Face transfer caches live in **`Work/Installation/`**,
separate from the durable output namespaces in Work. Set top-level
`models_folder = "/external/shared/Models"` to use external storage, or a relative
folder to use another project location. Existing explicit converted model paths and
explicit Hugging Face cache environment choices are preserved. `--models` overrides
the folder for a command. `--installation-info` reports resolved paths and scopes;
see [storage and cleanup details](docs/STORAGE.md) for exceptions and partial downloads.

```bash
./uninstall.sh                 # Remove owned runtime and installation caches
./uninstall.sh --remove-models # Also remove owned model downloads inside this project
```

Both commands preserve source/launchers, configuration, credentials, recordings,
TXT and processing checkpoints. Default uninstall also preserves models. External/shared
model and cache locations are retained with either command. Cleanup uses ownership
and file checksums, refuses edited/unrelated files, and never traverses cleanup links
into another installation. It works without a healthy `.venv` and is safe to repeat;
processing/setup must finish first. Run `./setup.sh` afterward to reinstall.

Back up configuration, recordings, `Transcribed/` and durable processing state in
`Work/`. Models and installation caches can generally be downloaded again;
checkpoints preserve resume and publication ownership. You may omit
`Work/Installation/` from backups, but keep the other Work namespaces.

## Configuration and languages

Use **`config.local.toml` beside the launcher**. It is created with mode 0600,
ignored by Git, and preserved byte-for-byte on setup reruns.
[config.example.toml](config.example.toml) is the complete credential-free example.
Default and configuration-relative paths resolve against the installation
folder, independently of the caller's current directory.

A complete minimal French-speech-to-English-text configuration:

```toml
source_language = "fr"
target_language = "en"

[translation]
backend = "local"
```

Keep `source_language = "auto"` for automatic detection, or set a known source
language to avoid ambiguity in short/mixed recordings. Setup and upgrades preserve
your existing local configuration and language overrides.

| Language | Code / accepted aliases |
| --- | --- |
| Automatic source detection | `auto` |
| French | `fr`, French |
| English | `en`, English |
| Other supported Whisper languages | For example `es`, `ja` |

See the [language reference](docs/LANGUAGES.md) for additional codes and aliases.
Source characters are preserved without a separate script conversion step.

| Setting | Valid values/default |
| --- | --- |
| `source_folder`, `output_folder`, `work_folder` | Nonempty paths relative to the installation; defaults `Video Source`, `Transcribed`, `Work` |
| `models_folder` | Default `Models`; relative to installation, or explicit external/shared folder |
| `source_language` | `auto` or a supported Whisper language code/alias |
| `target_language` | `en`; local backend accepts **only `en`** |
| `asr_model` | `large-v3`; supported faster-whisper name or converted model directory |
| `asr_device` | `auto`, `cpu`, `cuda` |
| `asr_compute_type` | `auto`, `int8`, `int8_float32`, `float32`, `float16`, `int8_float16`; must be supported on the device |
| `[translation].backend` | Exactly `local` or `api`; missing value means `local` |
| `[translation].base_url`, `model` | API-only base/prefix and exact exposed alias |
| `[translation].api_key_env`, `api_key` | API-only variable name (default `TT_TRANSLATION_API_KEY`) or ignored literal key |
| `[translation].timeout_seconds` | API 1–600 seconds; default 180 |
| `[translation].chunk_chars`, `context_chars` | API 50–2000 / 0–300 characters; defaults 1200 / 160 |
| `[translation].attempts` | API transient attempts 1–5; default 3 |

CLI overrides win over TOML for folders, languages, backend and ASR settings.
`--models` selects model storage and `--work` selects a processing folder; like
`--source`/`--output`, an explicit relative
CLI path uses the caller's directory. The default/TOML Work path uses the installation
root, even when launched elsewhere.
For API mode, `TT_TRANSLATION_BASE_URL`, `TT_TRANSLATION_MODEL` and the configured
key variable win over corresponding TOML values. Local mode ignores these API
values and credentials. `--config` selects another TOML file; relative folders
still use the installation root. Explicit CLI paths use the caller's directory.
The supplied launchers always use their canonical checkout root.

```bash
./transcribe-translate --source-language fr
./transcribe-translate --input '/path with spaces/Interview.mp4'
./transcribe-translate --source '/other/Video Source' --output '/other/Transcribed'
./transcribe-translate --check
```

The absolute launcher works from any directory. The installed command is also
available at `.venv/bin/transcribe-translate`, without environment activation.
Add that directory to your own PATH to use the bare command.

## Local versus API translation

**Local:** source transcription saves TXT and its checkpoint first. A separate
native Whisper `task="translate"` pass consumes the **original audio**, not saved
source text. Both passes share one loaded model and run sequentially. Native
translation uses supported `without_timestamps=True` text decoding; stored native
ranges describe decoding windows, not translated-word alignment. If the source is
English, its valid transcript is copied instead. No extra translator weights,
credential or local model service is needed. Non-English targets are rejected
early. Turbo, distillation and English-only named models are refused for local
translation; use a translation-trained multilingual model, normally large-v3.
See [Whisper's translation guidance](https://github.com/openai/whisper#command-line-usage).

**API:** local ASR saves the source, then bounded source-text chunks are translated
by one compatible Chat Completions endpoint. This path supports other validated
target-language codes. Matching source/target languages reuse the source without
requesting translation. Audio stays local; transcript text and continuity context
are sent to the configured endpoint.

Local Whisper is the simple baseline. API translation can use a larger bilingual
model and retranslate the same saved source text. Native translation can make
different errors because it hears audio independently. Neither is universally
best; review valuable names, numbers, negation and meaning against the recording.
Better text translation cannot recover speech missing or misrecognized in the source.
Our bounded speech tests found material native translation errors, including
lost facts and invented wording. Native fidelity did **not** pass on those clips;
see [the measured comparison](docs/QUALITY_FOLLOWUP.md) before relying on it for valuable content.

## Optional LiteLLM / compatible API setup

The deployment machine must be able to reach the gateway. Reachability from a
development machine does not establish reachability from another machine.

1. Obtain the client-reachable base URL (including its API prefix, usually `/v1`)
   and an authorized gateway API key from the gateway operator. A `0.0.0.0` bind
   address is not a client URL. This credential is separate from SSH or model-download
   credentials.
2. Copy the complete [API example](config.api.example.toml) into a separate ignored
   file, preserving your existing local configuration:

   ```bash
   cp config.api.example.toml config.api.local.toml
   chmod 600 config.api.local.toml
   ```

3. Edit `base_url` in that file. Enter the key without printing it, then list the
   aliases accessible to that key. Listing does not require a selected model or
   execute inference:

   ```bash
   read -rsp 'Gateway API key: ' TT_TRANSLATION_API_KEY
   printf '\n'
   export TT_TRANSLATION_API_KEY
   ./transcribe-translate --config config.api.local.toml --api-models
   ```

4. Set `model` to an exact listed alias. Do not guess the underlying weight filename.
   Check the route, then process your folder:

   ```bash
   ./transcribe-translate --config config.api.local.toml --api-check
   ./transcribe-translate --config config.api.local.toml
   ```

Ordinary `./transcribe-translate` still uses the preserved `config.local.toml`.
Each example has one `[translation]` table; do not append a second table when editing
one file. Preserve your existing source/target/folder/ASR choices when copying them
into the API file. Source recognition remains local and uses the `asr_*` settings;
the `[translation]` settings select the text model and request bounds. Source text
and neighboring context are sent to that gateway; audio remains local.

Trailing base slashes are stripped; `/models` or `/chat/completions` is appended
once. No additional `/v1` is inserted. Full operation URLs are rejected. For a
key-free API server, set `api_key_env = ""` and omit `api_key`.

`--api-models` lists authenticated model IDs. `--api-check` additionally verifies
your configured alias and translates one short authored text including a name,
negation and numbers.
It does not run audio inference or write transcript checkpoints. Only the answer
is printed, never separate reasoning or credentials. A successful check establishes
bounded authenticated connectivity and answer format, not translation-quality
acceptance. API failures remain errors; there is no silent native fallback.
To return to local processing, set `backend = "local"` and `target_language = "en"`, or use
`--translation-backend local --target-language en` for one run. Local mode performs
no API configuration validation/network calls; initial model downloads are separate
from offline processing once weights are available.

API requests contain only `model` and `messages`, respecting endpoint sampling,
output and reasoning defaults. The prompt requests faithful natural translation
without summarizing/commentary and treats spoken instructions as data. Answers
must be complete JSON containing the requested chunk ID and translated string.
Wrong IDs, extra/duplicate fields, empty/malformed answers, embedded thinking tags
and unfinished responses are rejected; separate model reasoning is ignored.

Every source character belongs to exactly one chunk; neighbor context aids
continuity and must not be repeated. Provide several thousand output tokens and
adequate input budget at the endpoint. Explicit truncation is rejected; reduce
`chunk_chars` if its defaults are insufficient. Chunk bookkeeping cannot prove
semantic completeness. Finite connection/read timeouts and bounded retries apply
to transient failures; auth/configuration and bad answers fail promptly. Successful
chunks remain saved.

## Installation, GPU and CPU

`./setup.sh` checks Python/venv/pip, Linux x86-64, disk and NVIDIA driver availability,
installs pinned packages into its own `.venv`, and downloads/reuses required model
files. Supported NVIDIA installations automatically receive isolated cuBLAS/cuDNN
libraries; there is no mandatory second `--cuda` setup step. CPU-only installation:

```bash
./setup.sh --cpu
```

A **new** config then selects CPU; existing config/environment/data are preserved
and existing CUDA packages are not removed. Runtime overrides:

```bash
./transcribe-translate --asr-device cpu --asr-compute-type int8
./transcribe-translate --asr-device cuda --asr-compute-type float16
./transcribe-translate --asr-device cuda --asr-compute-type int8_float16
```

Automatic selection prefers supported CUDA/float16, otherwise CPU/int8. Actual
model/device/compute are reported for both passes. Appropriate CUDA availability
or resource failures can fall back to CPU with the **same model**; explicit CUDA
fails and unrelated inference errors are not masked. `int8_float16` is an explicit
lower-VRAM option. Only one model is kept loaded for the sequential speech passes.
The selected `int8` alias can resolve to engine `int8_float32` on CPU. New result
records and `Actual` messages report the loaded engine type; older checkpoints
retain their original requested-type metadata. A reporting fix does not rerun them.

Pinned compatibility baseline: Python 3.11, faster-whisper 1.2.1, CTranslate2 4.7.2,
PyAV 18.1.0, optional cuBLAS 12.8.4.1/cuDNN 9.10.2.21. CUDA installation eligibility
requires Linux driver **525.60.13+** on known pre-Blackwell hardware (compute capability
5.x–9.x); Blackwell 10.0/12.0 requires **570.26+**. Unknown/unsupported capability
or driver detection selects CPU; explicit `./setup.sh --cuda` fails instead.
This follows [CUDA minor compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
and the [pinned cuDNN matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/reference/support-matrix.html).
Eligibility is not a guarantee: PTX JIT or newer driver-dependent features can
require a newer driver even within CUDA 12.
Verify `nvidia-smi`, fix/install the system driver through your supported distro
procedure, then rerun normal setup. Drivers/services are never changed automatically.
Launcher and installed CLI configure isolated library paths before inference.
On Linux x86-64 with the working driver already installed, the GPU steps are:

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
./setup.sh
./transcribe-translate --check --asr-device cuda
./transcribe-translate --cuda-smoke-check --asr-compute-type float16
# Run a short permitted recording to exercise actual CUDA inference:
./transcribe-translate --input '/path/to/test-video.mp4' --asr-device cuda
```

The final command must report actual CUDA execution for both speech passes to
establish GPU acceptance; choose a non-English recording to exercise translation.
No separate CUDA toolkit installation or environment activation is required by
this project's pinned pip-library path. See the
[faster-whisper GPU requirements](https://github.com/SYSTRAN/faster-whisper#gpu).
Normal CUDA setup runs the smoke check after preparing the selected weights. It
uses synthetic features, executes encoder/decoder kernels, and prints actual
device/compute/model; it never reads recordings or downloads more weights. Failure
is reported without CPU fallback. `--skip-model-downloads` also skips automatic
smoke; the explicit flag requires cached weights. Neither smoke nor `--check`
establishes speech quality or every possible CUDA code path.

**User-reported CUDA execution succeeded** on Debian 12 / RTX 3080 10 GB / driver
550.163.01 with these pins and large-v3: a 594.5-second video completed
source and native translation on CUDA/float16, with 1 completed and 0 failed, after
manual CUDA dependency installation. It was not reproduced in the CPU development
environment. **Fresh installation through corrected setup remains unverified**;
semantic quality limitations are unchanged. `--check` is not a GPU kernel test.
Windows/macOS/ARM installation is not accepted
by this release's installer; no Apple Metal backend or separate Mac port is added.
Upstream [CTranslate2 wheels](https://opennmt.net/CTranslate2/installation.html)
support macOS CPU execution, while their CUDA GPU support is for Linux/Windows;
that does not provide Apple GPU acceleration for this application.
Use [the isolated GPU acceptance sequence](docs/GPU_ACCEPTANCE.md) for fresh
inference and recorded-runtime assertions; cached CPU results cannot satisfy it.

PyAV wheels bundle FFmpeg libraries; normal use needs no separate FFmpeg executable,
PyTorch or alignment dependencies. Default model snapshot:
`Systran/faster-whisper-large-v3`, revision
`edaa852ec7e145841d8ffdb056a99866b5f0a478`. Standard `HF_HOME`/`HF_HUB_CACHE` can
select the cache. Preflight reserves 6 GiB without cached weights or 2 GiB with
them, plus recordings/outputs/temporary audio capacity; these checks are not
capacity guarantees. Decoded mono PCM uses about 115 MB/hour and the recognizer
loads audio into RAM. Very long files need proportionate RAM/disk. Optional
`--skip-model-downloads` deliberately defers weights (e.g. CI); `--python` selects
an interpreter on first install. Setup never changes global Python or other apps.

## Discovery, outputs and recovery

Discovery is deterministic and **top-level only** by default. `--recursive` scans
source subfolders. Hidden entries, symlinks and the output subtree are excluded.
Common video extensions are recognized; other containers are included only when
the decoder finds video. Codec support follows FFmpeg/PyAV. The first audio track
is streamed to private mono 16 kHz PCM without VAD silence removal/aggressive
filtering. An explicit single input may also be decodable audio. Filenames never
become constructed shell commands.

The normal folders stay together beside the launcher:

```text
Video Source/
  Interview.mp4
Transcribed/
  Interview.fr.txt
  Interview.en.txt
Work/
  Transcribed--<output-id>/
    owner.json
    batch.lock
    publications.json
    inputs/
      Interview.mp4--<input-id>/
        owner.json
        index.json
        asr/
        temporary/
        fr-to-en--local--<generation-id>/
          source.fr.txt
          translated.en.txt
          record.json
    temporary/
```

`Transcribed/` receives only readable source/target TXT. The video extension is
omitted; normalized canonical language codes supply the suffixes. Spaces and
Unicode in the video stem are preserved. Genuine stem collisions (including
extensions, subfolders, case and Unicode normalization) get a deterministic
input-path suffix. Both languages for that input share the same stem, for example
`Interview--<collision-id>.fr.txt` and `Interview--<collision-id>.en.txt`. Identical
source/target languages produce one TXT. Very long stems are shortened only to fit
filesystem limits. Plain source/native TXT omits generated confidence prefixes;
internal annotated artifacts and segment metadata remain available for review.

**Work is deliberately visible and separate.** It contains recovery records,
selected-generation indexes, ASR results, publication ownership, historical
translations, locks and temporary audio/writes. Each configured output path has a
separate namespace there. Nothing moves model weights, virtual environments or
unrelated project files. The Work root may be changed with `work_folder` or
`--work`; keep it outside Transcribed. Work and the output must share a filesystem
so publication can atomically rename a completed TXT from Work. For outputs on
another mount, configure Work on that same mount. Custom Work locations inside a
checkout should also be excluded from Git; the default `Work/` is already ignored.

Source TXT publishes after valid source transcription. Target TXT publishes after
complete translation. A translation failure retains source TXT and any previously
published valid target. Successful reprocessing updates the corresponding owned
public files atomically while keeping internal history. Checksums and publication
intent support interrupted-write recovery. Edited/unrelated TXT or checkpoints,
unowned state and symlinks are refused rather than overwritten. An output-directory
lock serializes current callers even with different Work overrides; the visible
Work lock and any existing legacy lock are held during migration/processing.

```bash
./transcribe-translate                         # migrate / resume / skip automatically
./transcribe-translate --reprocess translation  # reuse source, new translation
./transcribe-translate --reprocess all          # new source and translation
```

Input SHA-256/size and relevant model/runtime settings identify reusable ASR.
Translation/backend changes reuse compatible source and select distinct results;
local mode ignores API chunk/model settings. A native pass is checkpointed in full;
API saves each completed chunk. Completed selected generations rebuild missing
source/target TXT without decoding, ASR or translation. Modification time never
selects a generation. JSON retains the checksummed owner/payload envelope; technical
ranges and confidence flags are not guarantees of alignment or semantic accuracy.

**Automatic upgrade from older layouts:** ordinary reruns recognize legacy owned
hashed directories and 0.2.3 `.state/`. Migration copies all recognized generations,
validates ownership/content and records a journal in Work before retiring the old
owned files. Interrupted migration resumes from that journal. Existing target
publication ownership is converted so paired stems remain consistent. Compatible
model signatures and selected-generation relationships are unchanged. No manual
checkpoint move or fresh inference is needed for valid completed work with the
same inputs, configuration and overrides. Unrelated files remain in place with an
explanation; modified or ambiguous owned state is refused and preserved. Do not
run an older application version concurrently with an upgrade.

**Backups:** retain recordings in `Video Source/`, the entire `Work/`, public
`Transcribed/` files and ignored local configuration. Keep input/output identities
and processing settings consistent when restoring. Whole-project relocation via setup
can copy verified Work to the new relative paths; arbitrary input moves create new
identities. Work contains durable caches, generations, ownership and migration
journals, so deleting it loses automatic resume/skip and safe public-file updates.
Only `temporary/` audio/write leftovers are disposable when no batch is running;
model weights in Models can be fetched again separately; installer caches in
Work/Installation are disposable through uninstall. TXT alone remains readable but
cannot reconstruct recovery state. Keeping only TXT requires fresh inference into
a fresh output/state location; existing unowned TXT is protected.

**Upgrading 0.1.0:** configuration and old API results remain untouched. A missing
`[translation].backend` now means local even if API fields remain. Set
`backend = "api"` to opt into API and verify its current alias. Compatible legacy
source/API checkpoints remain reusable through the safe automatic Work migration;
no model-result reset is forced.

## Troubleshooting and limitations

| Problem | Action |
| --- | --- |
| Wrong source/version or moved runtime | Check `--installation-info` and canonical root; update that checkout, then run `./setup.sh --reinstall` |
| Missing prerequisites | Install supported Python/venv/pip and OS runtime prerequisites; rerun setup |
| CUDA unavailable/library error | Check driver/pinned libraries, rerun setup, or explicitly choose CPU |
| Local non-English target | Choose API and its endpoint/model, or target `en` |
| API 401/403 | Verify gateway API key and alias access, not SSH/Hugging Face credentials |
| API 400/404 | Check base/prefix and exact `/models` alias |
| Timeout/429/5xx | Check availability/cold loading and finite timeout; rerun to resume |
| Truncated/bad API answer | Check output defaults/budget, reduce chunks, rerun |
| Corrupt video/no audio/no speech | Complete file copying; inspect first audio track and language override |
| Edited/incompatible output | Preserve it and choose a fresh output folder |
| Empty source | Place videos there and rerun; helpful message and success exit |

Exit codes: 0 success/empty batch, 1 any per-file failure (others continue), 2
configuration/setup/batch error, 130 interruption. Source survives translation
failure. Whisper can hallucinate, miss quiet speech, confuse names or mistranslate;
API models can omit nuance or guess wording. Beam search 5 and previous-text
conditioning retain the quality baseline; no smaller model is silently substituted.
Review important facts against audio. Bounded read-speech tests are not broad
accuracy or production-readiness certification. There is no diarization, speaker/
emotion/relationship analysis, subtitles/alignment, web UI, dubbed audio or service.

## Development and licensing

```bash
./setup.sh --cpu --skip-model-downloads
.venv/bin/python -m pip install -r requirements-dev.txt
./scripts/check.sh
```

CI uses these model-free checks without production credentials; doubles test state
and errors, not accuracy. See [validation](docs/VALIDATION.md),
[architecture](docs/ARCHITECTURE.md),
[quality configuration](docs/QUALITY_CONFIGURATION.md), and
[manual publication](docs/PUBLICATION.md).
The project is independent of `transcribe-media`: its MIT-licensed folders,
installer conventions, CTranslate2 pin and atomic recovery approach informed fresh
code. Native translation uses faster-whisper's supported task; the existing API
text path remains optional for larger bilingual models/other targets. The app is
[MIT licensed](LICENSE); packages/models/provider terms are separate. Attribution,
including the licensed human fixtures, is in [THIRD_PARTY.md](THIRD_PARTY.md).
Never commit credentials, media, results, checkpoints, weights or environments.
