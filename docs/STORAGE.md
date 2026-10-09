# Model storage, installation caches and uninstall

The default portable layout is:

```text
project/
  .venv/                           project Python runtime
  Models/
    Whisper-large-v3--<revision>/   five verified weight/config/tokenizer files
  Video Source/                    your recordings
  Transcribed/                     complete source and translated TXT
  Work/
    Installation/                  disposable pip/Hub transfer caches and ownership
    <output-namespace>/            durable ASR, generations, indexes and publication state
  config.local.toml                 private configuration, retained byte-for-byte
```

Configuration-relative paths use the canonical installation root. CLI `--models`
relative paths use the caller directory, like other explicit CLI folders. The default
model is still multilingual large-v3 revision
`edaa852ec7e145841d8ffdb056a99866b5f0a478`; device/compute selection, decoding,
local translation and optional text API settings are unchanged.

## Import, reuse and download

With no explicit model/cache location, setup finds the previous implicit Hugging
Face cache under `~/.cache/huggingface/hub` (or the XDG cache root). It considers
only the exact pinned large-v3 snapshot and its five required files. Their sizes and
SHA256 must match the packaged manifest. The manifest was verified against the
[official pinned model metadata](https://huggingface.co/Systran/faster-whisper-large-v3/tree/edaa852ec7e145841d8ffdb056a99866b5f0a478):
model.bin matches its LFS SHA256 and JSON files match their upstream Git blob hashes.
No unrelated models, tokens or credential files are imported.

Hub snapshots can link into a separate blob store, as described in
[Hugging Face's cache guide](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).
Import copies bytes into independent regular files. The local model remains usable
when the original cache is unavailable; the original shared copy is neither modified
nor removed. This consumes an additional approximately 3 GB in Models. Nothing
automatically deletes the older shared copy: it may support other applications.

If no complete matching legacy copy exists, download only the selected pinned
files into a stable owned `.installing` directory inside Models. Disk reserve is
checked before import/download. Completed files and recoverable partial transfers
are retained on interruption. Rerunning setup verifies/reuses staged files and
publishes the complete model directory by atomic rename on the model filesystem.
Corrupt, edited or unrelated state is retained with an error rather than overwritten.
Normal reuse verifies the pinned files and does not redownload them. Setup reports
whether weights were imported, reused or downloaded. Setup's `--skip-model-downloads`
defers preparation; `--prepare-model` prepares the selected weights. CUDA smoke
uses cached weights only and never requests a model download.

Storage paths do not enter the named large-v3 identity, so moving valid weights
alone does not invalidate completed source/translation generations. Content,
model revision, runtime settings and library versions still govern reuse. An
explicit converted directory retains the existing path-and-file identity convention;
changing that explicit model identity can require reprocessing. Optional named models
retain their snapshot/content identity and use an owned `Whisper-named-models` Hub
cache. This release does not select a smaller model or new ASR backend.

## External choices and transfer caches

Set a top-level folder in the private TOML, for example:

```toml
models_folder = "/external/shared/Models"
```

A relative value resolves at the project root. `--models` takes precedence. An
existing explicit converted `asr_model` directory is used directly. Without
`models_folder`, explicit `HF_HUB_CACHE` or `HF_HOME` retains the former shared Hub
storage behavior; it is not automatically imported/deleted. `models_folder` takes
precedence over those ambient cache settings. No existing TOML choice is rewritten.
Diagnostics identify the selected model location and project-local/external scope.

Pip cache defaults to `Work/Installation/pip`. Hugging Face Xet/asset caches default
to `Work/Installation/huggingface/xet` and `assets`. Hub transfers receive an explicit
local cache path; `local_dir` downloads keep their small `.cache/huggingface` partial/
metadata sidecars with the owned model staging/folder. These sidecars are technical
model-transfer state, not transcription checkpoints. See the
[download guide](https://huggingface.co/docs/huggingface_hub/guides/download) and
[environment-variable reference](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).
The pinned library's supported cache controls are set before its imports.

Explicit `PIP_CACHE_DIR`, `HF_XET_CACHE` and `HF_ASSETS_CACHE` remain external/cache
exceptions and are reported. Settings are scoped to application/setup child
processes; HOME, global shell files and other applications are untouched. Normal
installation may download the existing pinned Python dependencies and the selected
ASR files. LiteLLM translation weights remain on their servers. Models or recording/
output/processing folders inside `.venv` or Work/Installation are refused to prevent
mixing preserved data with disposable runtime/caches. Cache-directory symlinks are
also refused; configure an external processing Work folder explicitly instead.

## Uninstall and recovery

```bash
./uninstall.sh
./uninstall.sh --remove-models
./setup.sh
```

Default uninstall removes the current owned runtime, verified owned retained
runtime backups/failed replacements and disposable Work/Installation caches.
Configuration, credentials, source/launchers, Models, recordings, public TXT and
processing namespaces remain. `--remove-models` also removes verified owned model
folders/staging inside the default Models directory and the currently configured
local model folder. Model parent directories may remain empty. External/shared
models, shared Hub/transfer caches and unrelated files are preserved.

Each removable area needs a checksummed ownership record and matching remaining
files. A configured path or familiar directory name alone is insufficient. Cleanup
never follows symlink targets. Installer-created runtime links are removed only
with their verified journal; their external targets stay untouched. Old unowned
backups/custom locations and edited/unknown files remain with an explanation and
exit 2. Resolve the reported ownership/integrity issue without discarding useful data.
Partially removed owned files may be absent on rerun; remaining files are rechecked.
An interrupted mutable cache/import should be recovered by setup before cleanup.

Uninstall uses base Python and works without a healthy project `.venv`. It requires
the same exclusive root/output locks and legacy process checks as setup, so active
processing/setup is refused. No process is terminated. It is safe to repeat and
reinstall afterward. Uninstall does not remove system prerequisites/drivers, Git,
source or a different project. Linux x86-64, Python 3.11–3.13, Bash/readlink/flock
and the documented runtime prerequisites remain required for setup.

Back up configuration/credentials, recordings, Transcribed and **durable Work
namespaces**. Model weights and installer caches are generally downloadable again;
resume/selection/publication metadata is not reconstructable from TXT alone.
Backing up all Work is safe, or omit only Work/Installation to save space. Preserve
legacy shared caches until other consumers and your own backups have been checked.
Installation/CPU recovery success does not establish GPU or semantic quality;
[GPU acceptance](GPU_ACCEPTANCE.md) and [quality limitations](QUALITY_FOLLOWUP.md)
remain separate.
