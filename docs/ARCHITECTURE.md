# Architecture

`config` validates TOML/environment/CLI settings and anchors default paths.
`languages` normalizes identifiers. `media` discovers a bounded source tree and
streams FFmpeg/PyAV decoding into temporary PCM. `asr` resolves a pinned/cached
model, checks supported runtime capabilities and performs source transcription.
`translation` partitions exact source character spans and sends optional bounded
sequential compatible-API requests. The local default uses `asr.Transcriber.translate`
on original audio in a separate native Whisper pass, sharing the same loaded model.
Native translation uses supported text-only decoding; its segments describe audio
windows rather than generated timestamp-token alignment. `storage` owns checksummed JSON, private
atomic writes, input hashing and locking. `pipeline` coordinates immutable
generations and recovery. `cli` reports per-file outcomes and exit status.

Input bytes + ASR settings form an ASR signature. A per-input index chooses its
current generation. ASR completion is atomically checkpointed before translating.
ASR generation + rendered source + translation settings form a translation
signature; the index similarly selects its current run. Local/API translations use
distinct signatures/generations. Native translation commits its complete result
before TXT publication; an interruption repeats only that pass, retaining source
ASR. Every successful API chunk commits immediately. Chunk identity includes integer ID, exact source character
range and source hash. Saved chunks must form a valid prefix of the current plan.
Target publication follows complete validated coverage, with output hashes saved
as intent before the atomic TXT write, then a completed record. Missing artifacts
can be recreated from valid saved work; edited artifacts cannot be overwritten.

Generation folders preserve historical results and isolate settings changes.
Reprocess selects a new generation through the atomic index, so an interruption
resumes that generation instead of accidentally returning to an old completion.
Input paths prevent cross-file collisions; SHA-256 checks detect middle-of-file
edits. The output lock coordinates all files in a batch. Only a completed
translation publishes a target TXT. Source TXT is independently valid.

Per-input jobs live in visible Work under a namespace bound to the canonical
output path. Each job retains its owner/index, immutable generations and a visible
`asr/` cache. Only source/target TXT belongs in the public output. Temporary PCM and
publication staging files live in Work `temporary/` directories; atomic publication
requires Work and output on the same filesystem. Config/TOML paths resolve against
the installation; explicit CLI paths resolve against the caller.

Checksummed publication schema 2 assigns a shared readable stem per input, with
canonical language artifacts and old/new checksum intent. A complete valid source
publishes independently before translation; identical source/target uses one file.
Missing copies rebuild, edited/unowned copies are refused, and successful new
selected generations replace public files while preserving internal history.
Plain presentation does not enter model signatures or change historical annotated
artifact checksums. Source and native confidence metadata remain internal.

Migration recognizes legacy owned jobs and `.state/`. It validates every cache,
generation, index and existing owned artifact, rejects ambiguous destinations or
unrecognized contents inside owned state, then saves a checksummed file manifest
and phase journal. Files are copied and verified in Work; migrated generations are
validated again before old owned files are retired. Retirement rechecks both copies
and originals and syncs directories. Every interruption boundary is resumable.
Existing publication tracking is upgraded and owned filename changes are backed up
before replacement. Raw migration tracking/backups remain in Work. Unrelated output
entries stay in place; modification time never selects a generation.

Linux `flock` on the output directory inode serializes callers independently of
Work configuration. A visible Work `batch.lock` and an existing empty owned legacy
lock are also acquired; the legacy lock is retired only after a successful batch.
Current callers keep the directory lock throughout this transition. Older versions
must not be launched concurrently after migration. Separate output paths use
separate namespaces and directory locks; unrelated Work files are untouched.

Back up recordings, configuration, public TXT and all durable Work state. Deleting
Work removes cache/selection/publication ownership; TXT alone cannot resume. Only
inactive `temporary/` leftovers are disposable. Model caches/environments stay at
their existing locations. The decoding baseline and remaining fidelity limits are
assessed in [quality configuration](QUALITY_CONFIGURATION.md).

Installation orchestrates repairs from a supported base Python outside the project
runtime, while all pip calls use the intended project Python. Source, metadata,
imports, launchers and default data roots share one canonical identity. Shared
installation-directory inode locks precede launcher imports; setup needs exclusive
access and also checks legacy output locks/processes. Rebuilt environments are
created at the final path, with retained backups and rollback on failure.

Relocation copies verified Work into the new relative output namespace, retaining
the old namespace. Checksummed ownership and generation input paths are rebound;
source/result content, generation IDs, selected indexes, model signatures and TXT
stems are preserved. Publication identities are rekeyed with unchanged integrity
intent. All generations are validated, with atomic namespace publication and no
mtime-based selection. Absolute configured paths and arbitrary legacy identities
are not rewritten. [Installation details](INSTALLATION.md) describe the boundaries.

`models` coordinates the selected weight location and process-scoped transfer caches.
The pinned large-v3 manifest records the exact upstream revision, sizes and SHA256
for only the five required files. Legacy import verifies bytes, copies snapshot/blob
links as independent files into stable staging, then publishes by atomic rename.
Model storage does not enter the named large-v3 processing signature. Explicit
converted/shared paths remain supported; LiteLLM model weights are not handled here.

`managed` records checksummed file ownership separately for runtime, model and
installation-cache directories. Reserved mutation intent supports interrupted
transfers; cleanup verifies remaining owned files and refuses edits/unknown files.
Uninstall uses supported base Python with exclusive installation/output locks and
legacy process detection. It removes local owned artifacts without traversing links
or discarding durable Work state, configuration, recordings or external caches.
