# Changes

## 0.2.7

Store selected weights in visible project-local Models by default. Verify and
import the exact pinned large-v3 files from the former implicit shared cache,
dereferencing blobs/links without modifying the original. Retain explicit external
model/cache choices and report resolved locations. Scope pip and Hugging Face
transfer caches to Work/Installation. Add checksum/ownership-based uninstall and
optional local model removal, retaining credentials, configuration, recordings,
transcripts, checkpoints and external storage. Preserve runtime repair/rollback,
locking and relocation, model signatures and quality defaults. Validate genuine
completed public-human-audio CPU results across relocation and missing-TXT recovery
with zero inference; GPU and semantic-quality acceptance remain unchanged.

## 0.2.6

Resolve setup and launchers to one canonical checkout, isolate Python/pip from
foreign environments and verify installed version, editable source and runtime.
Ordinary setup refreshes current source while reusing pinned dependencies/cached
weights. Automatically rebuild moved/copied/broken environments at the final path;
add `--reinstall`, retained runtime backups, rollback and processing locks. Detach
runtime symlinks without modifying external targets. Copy verified project-relative
Work to relocated identities, retaining original state, selected generations and
publication integrity. Preserve configuration, private API choices, media, results,
model quality, decoding and CUDA driver eligibility. CPU installation/recovery tests
and simulated GPU routing do not establish GPU execution or speech fidelity.

## 0.2.5

Add authenticated model listing without inference and a complete separate API
configuration example, preserving existing local choices. Keep the README
language-neutral with French as its primary example; retain detailed language
support and original quality evidence separately. Reproduce the native failures,
compare four deployed text-model families and prepare private API settings from
measured behavior. Evaluate Qwen3-ASR-1.7B in isolation: some source words improve,
but residual errors and unverified deployment do not justify integration. Preserve
large-v3/local defaults, decoding, CUDA installer, recovery signatures and paired
TXT/visible Work behavior. No full end-to-end fidelity acceptance is claimed.

## 0.2.4

Publish source and target TXT directly with one shared readable/disambiguated stem
per input; matching languages publish one file. Valid source survives translation
failure. Move working state into visible, configurable Work namespaces outside the
output, including caches, generations, publication ownership, temporary files and
locks. Journaled copy/validation/retirement migrates both older layouts and resumes
interruptions without resetting compatible results. Directory locking serializes
callers even with differing Work overrides. Document backups and review the pinned
quality configuration without changing models, decoding defaults or CUDA eligibility.
Fidelity failures remain unresolved; outline a separate Qwen3-ASR comparison.

## 0.2.3

Publish complete translated text directly as `<video-stem>.<target-language>.txt`
in the output folder, preserving spaces/Unicode and disambiguating real collisions.
Owned atomic publication protects edited/unrelated files and resumes interrupted
writes. New internal jobs use hidden `.state/`; legacy generations remain in place
and valid completed results publish direct TXT on rerun without fresh inference.
Successful reprocessing updates the direct TXT while retaining history. Make
French speech → English text the primary README example, retaining automatic
detection and the Mandarin language reference. Installer, local configuration,
model settings and semantic quality limitations are unchanged.

## 0.2.2

Correct automatic CUDA installation eligibility to account for CUDA 12 minor
compatibility: pre-Blackwell GPUs use the documented Linux 525.60.13 driver floor,
while Blackwell retains 570.26. Unknown/unsupported capabilities select CPU or fail
explicit CUDA setup. Pinned dependencies are unchanged. Normal CUDA setup now
performs a bounded cached-weight encoder/decoder smoke check without recordings
or additional downloads. Document user-reported RTX 3080/550.163.01 execution after
manual dependency installation; fresh corrected installation remains unverified.
Transcription/translation settings and semantic quality limitations are unchanged.

## 0.2.1

Report the loaded CTranslate2 engine's actual device/compute type rather than the
requested compute alias (CPU `int8` resolved to `int8_float32` in acceptance).
No inference settings or default backend change. Preserve failed native results,
document direct-library/float32 controls and a longer human-audio API comparison,
put the native fidelity failure beside the quick start, and add isolated manual
GPU acceptance commands. Native translation remains unaccepted for tested fidelity.

## 0.2.0

Default local speech-to-English translation through a separate multilingual
Whisper pass sharing large-v3 weights with source transcription. Supported
text-only translation decoding and duration checks reject impossible segments.
Optional API text translation remains explicit; legacy source/API checkpoints
remain compatible. Backend generations preserve earlier results. Setup detects
supported NVIDIA drivers and installs isolated CUDA libraries automatically,
offers CPU-only installation, checks platform/disk prerequisites and preserves
existing configuration/data. Updated local quick start, API discovery/check,
recovery tests and bounded real acceptance documentation.

## 0.1.0

Initial standalone sequential video transcription/translation package, pinned
multilingual large-v3 baseline, automatic CPU/CUDA selection, explicit overrides,
bounded compatible-endpoint translation, atomic resumable generations, Unicode
paths, repeatable isolated setup, focused tests and public-safe release branch.
