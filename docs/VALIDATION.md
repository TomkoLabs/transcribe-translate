# Validation — 0.2.7

## Local models, caches and uninstall

Shell syntax, compile checks, pip dependency consistency, the 0.2.7 wheel/packaged
model manifest, relative documentation links and public-data audit passed.
220 regression tests passed on Linux x86-64 / Python 3.11.16. Added storage tests
use tiny byte fixtures (not a smaller ASR model), transfer doubles and model-free
safety checks. They cover dereferenced legacy import, independent offline reuse,
unchanged named-model identity across storage paths, staged import/transfer
interruption and atomic promotion, checksum/disk-space checks, edited/unrelated
files, explicit external choices, reserved cache recovery, external link refusal,
uninstall/repeat/model removal, partial cleanup retry and active locks. All existing
installation, relocation, recovery, publication, decoding and device tests passed.

Real setup used actual pinned dependencies and approximately 3 GB of genuine
large-v3 weights. An interrupted real-file import retained a partial/staging copy;
normal setup resumed and imported from the legacy shared snapshot offline. Files
match the exact upstream revision: model.bin's LFS SHA256 and the other four Git
blob hashes were independently checked, with SHA256 stored for local integrity.
Snapshot symlinks were copied as regular independent files. The real multilingual
CPU engine loaded from Models with the legacy cache unavailable, reporting
CPU/int8_float32. This is weight/runtime loading, not speech or GPU acceptance.

Ordinary rerun and explicit --reinstall reused the same local weights offline
without redownload and preserved configuration/durable state. Real default
uninstall retained models/data, repeated uninstall succeeded, reinstall reused
models, and --remove-models removed owned local weights while retaining configuration,
source/public TXT and all processing state. External model storage and explicit
legacy Hub reuse/uninstall left their original files byte-identical. Uninstall
worked with an unhealthy copied runtime; journaled runtime links were removed
locally without changing external targets. Active root-lock uninstall was refused.

The bounded real-artifact check used a previously completed, permitted CC-BY Google
FLEURS human recording and its genuine CPU source/native results and API history.
Initial isolated fixture preparation changed only path ownership metadata and
verified the original input fingerprint; original evidence remained unchanged.
All generation IDs, result content, model settings and selected indexes were retained.
Real copy/move setup rebinding recovered the saved selected generation, including
when the former copied installation path no longer existed. Missing direct TXT
rebuilt with **0 completed, 1 skipped, 0 failed**. Decoding, model loading, ASR and
translation entry points were guarded during recovery: **zero calls**. Recovery
also passed after uninstall/reinstall. Edited public TXT was refused and retained;
an unrelated TXT and every durable Work/configuration/source file survived cleanup.
No fresh speech inference or model-quality comparison was performed.

The actual main checkout's configuration/private companion and existing Work
files remained byte-identical. Library pins, large-v3 revision, decoding settings,
CUDA eligibility and translation behavior remain unchanged. CPU installation,
weight loading and saved-result recovery do not validate GPU execution or semantic
fidelity. Kraken's current installation/smoke still requires manual acceptance;
the supplied older GPU run remains user-reported historical evidence. No remote
GPU access, service change or GitHub publication occurred.

The following sections are historical release evidence.

## Installation, upgrade and relocation (0.2.6, October 8, 2026)

206 regression tests passed on Linux x86-64 / Python 3.11.16. New installer tests
use pip/model/GPU doubles and label that scope. They cover current-source refresh,
CUDA 550.163.01 routing via the unchanged eligibility policy, dependency reuse,
CPU-only routing, configuration/private-route preservation, external runtime links,
stale editable metadata/entry points, pip/Python isolation, rebuild rollback,
version disagreement, processing locks and legacy active processes. Relocation
checks cover all generations/selected-index reuse, original present/unavailable,
completed result rebuilding without inference, partial chunk/source resume, edited
TXT refusal, atomic namespace failure/retry, absolute paths and unrelated state.
Existing output/recovery, quality-setting and supported-device regressions passed.

Real setup executed fresh installation, ordinary same-version refresh, explicit
reinstall, dependency-failure rollback/retry, copied and renamed checkouts, original
path unavailable, directory/file aliases, external `.venv` symlink detachment,
paths with spaces, and launch from another working directory. Actual pinned
packages were installed using a local wheelhouse; normal setup reused cached
large-v3 weights offline. Foreign activation, PYTHONHOME/PYTHONPATH, PATH pip,
pip target/interpreter variables and pip configuration did not redirect installation.
Identity diagnostics and `pip check` passed for each accepted runtime. A source
edit at the same version was imported after ordinary refresh. Broken pip entry
points caused automatic recreation. Active-runtime and shared installation-lock
attempts failed before replacement; simulated CUDA loader-path reexec passed on CPU.

The real CLI skipped a model-free saved completion with two historical generations
after copying, then rebuilt its missing direct TXT after moving with the original
path unavailable: **0 completed, 1 skipped, 0 failed**. The fixture was generated
with inference doubles, not real speech; no fresh ASR/native/API call occurred.
Both original/copied old namespaces were retained, and the selected generation
was reused. Configuration/data bytes and modes remained unchanged in rerun,
reinstall, rollback and foreign-environment cases. The actual current checkout's
local/private configuration and existing Work files also remained byte-for-byte
unchanged after setup. No recordings or model caches were removed. Shell syntax,
compile checks, the 0.2.6 wheel build, public-source/private-data audit and
documentation links passed.

A saved explicit CUDA/float16 + API configuration stayed unchanged on the CPU
host: setup prepared the selected large-v3 weights without device execution;
explicit CUDA runtime/installer checks failed clearly with exit 2. A fresh ordinary
install retained `auto/auto` and large-v3. CUDA package installation routing is
simulated; **no GPU or Kraken access/test occurred**. Corrected fresh/rebuilt setup,
CUDA libraries/kernels/VRAM on Kraken remain manual acceptance. The supplied older
RTX 3080/550.163.01 run remains user-reported evidence. These installation/capability/
recovery checks do not establish speech fidelity or change semantic-quality limits.

The following sections are historical release evidence.

## Quality/configuration follow-up

192 regression tests passed on Linux x86-64 / Python 3.11.16, with shell syntax,
compileall, diff checks and dependency validation. The new model-free tests cover
authenticated inference-free alias discovery, missing model selection, malformed
discovery/authentication failures, clear API errors, and separate valid API TOML
with repeated setup preservation of existing local choices. These doubles do not
establish translation quality or GPU execution. Existing output/recovery tests
continue to cover migration, atomic publication, resume, reprocessing, collisions
and edited/unrelated-file protection.

Real CPU application inference reproduced all three permitted failed baseline
source/native results exactly. Four deployed text-model families completed the
same five saved/authored cases (24 strict complete responses), including a long
French context/mixed-language check. A selected-route real CLI batch rejected one
malformed JSON answer; one bounded rerun finished it without new ASR. Completed
checkpoint rebuild then reported 0 completed, 3 skipped, 0 failed with inference
blocked, restoring six clean TXT and preserving all six historical generations
and Work-state bytes. API listing and authored --api-check passed from the
development host; deployment-host connectivity was not tested.

An isolated official Qwen3-ASR-1.7B CPU experiment completed four EOS-terminated
calls, and its three source texts were actually translated through the selected
API. Some important facts improved, but residual reference failures remain. The
experiment did not change the application stack or integrate another ASR backend.
See [quality follow-up](QUALITY_FOLLOWUP.md) for methodology, precision controls,
model selection and stage-specific limits. Authored French text is not French
acoustic acceptance; historical Mandarin fixtures were not relabeled.

Normal CPU setup reused the pinned cached model offline and preserved configuration
byte-for-byte. Editable 0.2.5 installation, pip check, wheel/source verification,
relative documentation links and the private-data audit passed. Existing local
configuration and all 77 snapshotted prior runtime/evidence files remain identical.
The launcher also ran from another directory with the empty default source folder.
CUDA eligibility/pins, ASR/translation signatures, decoding and recovery machinery
are unchanged. No Kraken access or GPU test occurred; corrected fresh installation,
GPU memory/runtime and deployment gateway connectivity remain manual acceptance.

The following sections are historical release evidence.

## Visible Work and paired TXT release

178 regression tests passed on Linux x86-64 / Python 3.11.16. Model/driver tests use
doubles; the effective supported defaults are also inspected from the pinned
library. Focused coverage includes both language TXT, same-language single-file
publication, Unicode/spaces/collisions, source surviving API/native failure, atomic
publication/rebuild, both legacy migrations, selected-generation reuse, API/local
resume and reprocessing, copy/publish/retire/directory-removal interruptions,
checksum/ownership/symlink/unrelated-file protection, paired-stem upgrade collisions,
multiple outputs and different Work overrides, and cross-process directory locking.
Work anchoring from another caller directory and the clean TXT-only output are tested.

The real CLI migrated copies of three previously completed human-audio CPU results
from each older layout, including all historical generations. Every migration and
missing-TXT rebuild reported **0 completed, 3 skipped, 0 failed**, yielding three
source and three English TXT files. Inference methods were blocked and never called;
offline mode was enabled. All saved-generation bytes and original ignored evidence
were preserved; owned old state and locks were retired only after validation.

Normal setup on the CPU host reused the pinned cached large-v3 weights offline,
installed 0.2.4, passed dependency validation and preserved the existing local
configuration byte-for-byte. It created visible Work without changing CUDA driver
eligibility, package pins or model settings. The absolute launcher ran successfully
from a different current directory against the empty ready source folder, with its
Work namespace beside the launcher and no working artifacts remaining in Transcribed.
The wheel build and public-source/link audit passed. No alternative model framework
or weights were installed, no fresh speech-quality comparison was run, and no GPU
host was accessed. [Quality configuration](QUALITY_CONFIGURATION.md) records the
primary-source review and a separate concrete Qwen3-ASR evaluation proposal.

The following output-layout, installer and quality results are historical release
evidence. New functional checks do not change the failed semantic acceptance or
prove fresh GPU installation/execution.

## Direct translated TXT layout

144 model-free regression tests passed on Linux x86-64 / Python 3.11.16. The focused
layout tests cover spaced/Unicode/long filenames, filename/target-language/case/
normalization collisions, hidden new state, plain native text, complete-checkpoint
rebuild without decoding/inference, API failure/resume, reprocessing with retained
history, atomic rename failure and interruption at publication boundaries,
legacy local/API selected-generation reuse, edited/unrelated/symlink protection,
checksummed registry integrity and incomplete-result refusal. Model calls use
doubles; these tests do not establish speech quality or GPU acceptance.

The real application CLI also published three direct TXT files from copies of
previously completed human-audio CPU checkpoints. Both initial legacy publication
and a rerun rebuilding a missing direct TXT reported **0 completed, 3 skipped,
0 failed**. Inference methods were blocked and never called; offline mode was
enabled. All 28 copied legacy files and the original evidence remained byte-for-byte
unchanged. Legacy jobs remained in place and `.state/` held only publication state.
This validates real saved-result compatibility, without fresh model execution.

The existing local configuration (including its source-language override), media
and output files were preserved. The corrected CUDA installer and dependency pins
are unchanged. Editable installation, `pip check`, the 0.2.3 wheel build, relative
documentation links and the tracked-source/private-data audit passed.
The observations below remain historical evidence; this layout
release does not reopen semantic quality acceptance or GPU validation.

## Installer compatibility correction

113 tests passed on the CPU development host, including simulated driver
550.163.01/RTX-class capability eligibility, exact 525.60.13 boundaries, the higher
Blackwell floor, unknown/unsupported GPUs, missing detection, CPU-only selection,
strict smoke failure and setup rerun preservation. The shell installer test uses
real isolated Python/preflight with stubbed package installs, GPU detection and
model execution; it verifies routing, not GPU acceptance. Cached-only smoke
orchestration is tested with a GPU engine double. Real CPU execution of the same
encoder/decoder API calls passed with cached large-v3; explicit CUDA smoke on this
GPU-less host failed clearly with exit 2. These CPU results do not validate CUDA.

The user reports successful previous-runtime execution on Debian 12 / RTX 3080
10 GB / driver 550.163.01 with the unchanged pinned stack, after manual CUDA package
installation. A 594.5-second Mandarin video completed both large-v3 speech passes
on CUDA/float16, with 1 completed and 0 failed. This is user-reported runtime
validation, not agent reproduction, fresh corrected setup validation, or semantic
quality acceptance. See [GPU acceptance](GPU_ACCEPTANCE.md) for compatibility
sources, limits and fresh-installer commands. No private media/logs are committed.

The following quality/installation observations are historical 0.2.1/0.2.0 evidence;
this correction does not change speech settings or reopen quality acceptance.

The [focused quality closeout](QUALITY.md) adds fresh CPU processing of both failed
clips and a 66.72-second human-read compilation, direct-library/float32 controls,
and an audio-based API comparison of the exact saved source across three chunks.
Source ASR, native translation and API end-to-end fidelity **failed** the bounded
semantic criteria. API fidelity to saved source passed the bounded qualitative
check after one rejected malformed response and resume. Neither backend is accepted
for unreviewed deployment. No GPU host was accessed or tested.

0.2.1 fixes actual runtime reporting: loaded CPU `int8_float32` replaces the requested
`int8` label in new result metadata. A fresh two-pass run verified both fields with
unchanged output, and all 87 regression tests passed. Installation/recovery logic
was unchanged; the earlier clean-install and SIGINT checks below are historical
0.2.0 evidence, not newly repeated checks. See
[unexecuted manual GPU commands](GPU_ACCEPTANCE.md).

This release has bounded **functional CPU acceptance**, with material observed
translation accuracy limits. It is not broad Mandarin accuracy acceptance or a
production-readiness certification. Previous 0.1.0 checks remain historical;
current local/backend behavior is described below.

## Automated and installation checks

87 focused tests passed on Linux x86-64 / Python 3.11.16, including the updated
engine-type reporting regression. They cover local defaults
without API settings/network clients, backend validation and early local target/
model rejection, tracked/minimal config examples, Unicode and deterministic
bounded discovery, actual FFmpeg/PyAV decoding and missing audio/corrupt media,
per-file failure continuation, exact API chunk coverage and malformed/truncated
answers, finite retries and reasoning exclusion, backend/settings invalidation,
legacy API reuse, source reuse across backends, native-pass/API-chunk recovery,
publication interruption, unchanged-input skipping, same-language behavior,
operator-edit refusal, locking, model sharing, supported device selection,
installer CPU/CUDA decisions and configuration/data preservation. Model/driver
calls in unit tests are doubles; they do not establish real inference or GPU use.

Normal setup installed the updated application in its isolated environment,
passed `pip check`, preserved the existing configuration byte-for-byte, selected
CPU dependencies on the CPU host and reused the exact cached large-v3 weights.
A separate clean installation in a directory containing spaces required no API
configuration. Repeated setup preserved edited config and existing source/output
files. Checkout and installed CLI ran from another current directory. No system
packages, drivers, other application environments or production services changed.
A 0.2.0 wheel built successfully. Release files, their relative documentation
links, wheel contents and prior Git history passed the private-data audit.

The GitLab account/project/CI status was inspected through existing authorized
keyring-backed CLI access. The previous remote jobs failed with
`stuck_pending_no_matching_runners`, not a test failure. CI remains model-free;
remote runner execution is outstanding unless matching infrastructure becomes
available. This task does not change runner infrastructure.

## Actual human Mandarin local processing

Two human-read Google FLEURS `cmn_hans_cn` validation samples were reused under
CC-BY-4.0 with known Chinese references. Dataset revision:
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`; validation audio filenames
`10116334326946811387.wav` and `10149760684641961155.wav`, approximately 18.38 and
7.44 seconds. See [attribution](../THIRD_PARTY.md). Their actual human audio was
placed in generated FFV1/PCM Matroska video containers with spaced/Unicode paths;
speech was not synthesized and no personal recording was used. Fixtures, references,
outputs and raw evidence stay outside tracked source.

The default folder workflow performed actual decoding, source transcription and
separate native speech-to-English translation on CPU/int8 using multilingual
large-v3. Source TXT was preserved before the translation pass. Both speech passes
share a single loaded model; the second file reused it. Source detection reported
Chinese at approximately 0.995/0.996 probability. UTF-8 source/English files and
independent native result records completed. A rerun skipped both without inference.
An interruption retained completed source ASR; resume repeated only unfinished
native translation and published no incomplete target.
A separate offline run with the explicit `Mandarin` override normalized to `zh`
and completed both real CPU passes on the 7.44-second clip. Its observed fidelity
errors persisted; a language override alone did not fix them.

Actual stack: Python 3.11.16, faster-whisper 1.2.1, CTranslate2 4.7.2, PyAV 18.1.0;
large-v3 revision `edaa852ec7e145841d8ffdb056a99866b5f0a478`, CPU int8, beam size 5,
previous-text conditioning enabled, no VAD silence removal. Native translation uses
supported text-only `without_timestamps=True`. No smaller/turbo/English-only model
was used for acceptance.

The initial native candidate with generated timestamp tokens repeated/invented
English after the real recording ended. It was rejected for the release. Text-only
native decoding removed that timestamp-driven repetition on the tested sample;
validation also rejects segment ends beyond actual audio duration (with a small
rounding tolerance). This guard cannot detect arbitrary semantic errors inside a
valid audio window.

## Observed fidelity and optional API

Source transcription of the first clip matched its Chinese reference characters
when punctuation/whitespace were excluded. The second clip misrecognized the word
for mammals and the island name. Native English retained the first clip's negation
but changed its sectarian-war/list meaning. On the second clip it lost important
facts, including mammals/giant tortoises/herbivores, and introduced flying/predator
wording and a wrong island name. **Native translation fidelity was not accepted on
these samples**, despite successful decoding, output, model-sharing and recovery
checks. Low-confidence annotations did not identify every error. Source preservation
makes these failures inspectable; valuable content requires review against audio.

The optional API path was validated separately through a currently configured
quality-oriented bilingual Qwen 27B alias. Current gateway defaults include thinking;
only `model`/`messages` were sent. Authenticated `/models` discovery and one bounded
Mandarin text translation passed: the answer retained a name, negation, ticket count
and price without executing transcript instructions or saving reasoning. Actual API
translation of a saved human-speech source reused existing local ASR and produced a
separate generation. Private endpoint/alias details are in the ignored handoff.
That API answer compressed the first clip's war/violence/chaos list relationship,
following a source TXT whose punctuation also merged the list. It therefore does
not establish perfect end-to-end fidelity.

A second current authored-text check retained Wang Xiaoming, Li Hua, Zhang Wei,
the date October 2, 2026, 30 tickets at 300 yuan totaling 9,000 yuan, cancellation
of two tickets, no refund yet, uncertainty and negation. The quoted instruction
to ignore prior instructions remained translated content rather than an action.
No omissions, summarization or separate reasoning appeared in this bounded answer.

The earlier October 2 API checks also covered dates, names, numbers, uncertainty,
spoken instructions as data and a real chunk-1-of-2 SIGINT/resume. Current unit tests
retain chunk recovery coverage. Those checks do not prove universal model accuracy,
and a bounded authored text check is separate from speech-recognition accuracy.
No conversational/noisy/quiet-speech benchmark or broad long-recording fidelity
study was performed.

## Hardware boundaries

This development environment has no GPU or access to the reported hardware. No
remote GPU probes/deployment occurred. The separately supplied user run validates
reported prior-runtime execution; fresh corrected installation, new CUDA smoke,
broader kernel/VRAM/fallback behavior and speech quality remain unverified here.
Detection tests use doubles, and CPU execution cannot establish GPU behavior. The
documented driver floors account for NVIDIA minor compatibility and its limits;
eligibility does not guarantee every CUDA feature/code path.
macOS/Windows/ARM installation and Apple GPU execution are outside this release's
accepted installer platform.
