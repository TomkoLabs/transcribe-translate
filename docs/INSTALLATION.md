# Installation identity, updates and relocation

Normal installation and updates use `./setup.sh`. Explicit runtime rebuilding uses
`./setup.sh --reinstall`; neither command downloads newer application source or
changes existing local configuration. Linux x86-64, Bash, `readlink`, `flock`
(util-linux), Python 3.11–3.13 with venv/ensurepip and libgomp1 are prerequisites.
Python 3.11 is the tested baseline. System Python, drivers, global CUDA, sibling
checkouts and foreign virtual environments are not installation targets.

## Choosing the checkout and interpreter

Setup and both launchers resolve file/directory aliases to one physical project
root. Requirements, source, `.venv`, local configuration and default/relative data
paths use that root. Existing explicit absolute data/model paths remain configured;
setup does not rewrite them. A directory alias to an older checkout legitimately
selects its older source. Check before updating:

```bash
pwd -P
readlink -f ./setup.sh
git rev-parse --show-toplevel
git status --short
```

For a clean Git checkout, fetch and fast-forward the intended branch, then setup:

```bash
git fetch origin
git switch main
git pull --ff-only origin main
./setup.sh
./transcribe-translate --installation-info
```

Do not discard unrelated local changes to make an update work. Setup reports the
canonical root and source version before installation. It uses a supported base
Python outside `.venv` to orchestrate repairs, then the exact project `.venv/bin/python
-I -m pip` for package work. `--python /path/to/python3.11` selects a supported base;
a change of base rebuilds the environment. Otherwise a healthy existing base is
reused. Foreign activated environments, Python injection variables and PATH pip
commands do not select the installed package. Pip location/extra-requirement
variables and configuration files are ignored; intentional index, local-wheel,
cache and network controls remain available. Credentials/model-cache settings
are inherited without being printed.

The installed console command routes through the same isolated, locked checkout
launcher. Successful setup verifies exactly one project-local editable distribution,
matching source/package/installed versions, and the imported application path.
`--installation-info` verifies the identity without configuration, media discovery,
model loading or endpoint requests. Expected fields include:

```text
version: 0.2.7
root: /absolute/current-checkout
python: /absolute/current-checkout/.venv/bin/python
environment: /absolute/current-checkout/.venv
import: /absolute/current-checkout/src/transcribe_translate/__init__.py
launcher: /absolute/current-checkout/transcribe-translate
installed_cli: /absolute/current-checkout/.venv/bin/transcribe-translate
```

All paths should name the intended canonical checkout. Version output alone is
insufficient: a copied environment can report a local interpreter while importing
another checkout's application. Setup diagnoses stale editable metadata, identity
markers, external environment links, damaged interpreter/pip and stale entry points.

## Observed failure mechanisms and remaining uncertainty

On the CPU development host, a real copied virtual environment reported its new
local interpreter/prefix but imported the application from the original checkout's
editable source. Reinstalling from a wrong root cannot correct which source was
selected. Directory and file aliases were also exercised: physical root resolution
selects the aliased checkout, which can legitimately be an older source tree.
The old setup and launcher differed in file-symlink resolution.

The reported older application version and obsolete unconditional GPU-driver floor
indicate old source/preflight code was executed; a stale editable import alone does
not explain every symptom. An alias into an older checkout or outdated source in
the invoked location is plausible. The affected machine's filesystem/Git state was
not accessed, so its exact cause and repair remain unconfirmed. Update the intended
physical source first, then inspect the installation diagnostics. No production
or sibling checkout was changed during this development task.

## Repeat installation and rebuild recovery

Ordinary reruns refresh editable application metadata and entry points with
`--force-reinstall --no-deps`; pinned dependencies are installed normally so matching
versions are reused. Checked-hash bytecode prevents stale same-version source edits.
Cached selected weights in Models are reused. Matching old shared weights are
imported without download. [Storage and cleanup](STORAGE.md) describes model
identity, ownership, local transfer caches, external exceptions and uninstall.
No smaller model/backend is substituted.

On rebuild, setup renames only the canonical root's literal `.venv` to a unique
`.venv.backup-*`, then creates the replacement **at `.venv` itself**. Python documents
[virtual environments' absolute paths and non-portability](https://docs.python.org/3.11/library/venv.html#how-venvs-work);
a temporary venv renamed afterward would leave invalid shebangs. External `.venv`
symlink targets are left untouched; the link itself is retained in a backup.
Unsuccessful rebuilding retains `.venv.failed-*` and restores the prior runtime
at its original path. Backups are ignored by Git and retained for manual inspection;
do not execute a backup in place because its scripts refer to `.venv`.
Fix the reported failure and rerun `./setup.sh --reinstall`. Setup does not delete
recordings, outputs, Work, model caches or retained runtimes.
Use the ownership-checked uninstall commands to remove disposable runtime/cache artifacts.

A shared canonical directory-inode lock is held by launchers before application
imports and throughout processing, including CUDA library reexec. Setup requires
an exclusive lock before mutation. Existing output locks and a Linux process check
also protect older processing commands. Stop a batch or wait for it; setup does not
terminate it. These are cooperative locks, not protection against unrelated programs
editing the same files. Retained backup environments use disk space until the user
removes them after confirming the current installation.

## Relocated results

Copy/move the whole project with its local configuration, source, output and Work,
then run setup at the destination. Runtime recreation alone does not make absolute
checkpoint identities portable. For copied **Work-layout** state and project-relative
output/Work folders inside the destination, setup validates existing ownership,
checksums, indexes, source caches, all generations and published TXT before copying
the old namespace into a new one. Relative configured source identities are rebound
to the new source folder; explicit absolute/external source paths remain unchanged.
The old Work namespace is retained byte-for-byte; no original checkout is modified
or needed. Publication stems and model signatures are unchanged. Content hash/size
and existing settings still decide cache reuse; path rebinding does not make changed
recordings compatible. Missing published TXT may be rebuilt on the next normal run.

The complete copy is published by atomic namespace rename. Interrupted staging
copies remain in `.relocation-*` inside Work for inspection; setup never selects
one as completed state. A previously completed relocation is validated and reused
on retry. Edited, linked, unowned or ambiguous artifacts are refused and retained.
An existing unrelated destination namespace is never overwritten. Absolute paths
in local configuration are never rewritten. External/absolute output or Work paths
keep their existing identity and receive no automatic rebinding.

For old hashed-output/`.state` layouts or an unfinished legacy migration, first
finish the normal application migration at the original paths before relocating.
If already relocated, preserve the legacy data and restore access at its original
paths to complete that migration. This release does not infer new paths for arbitrary
legacy layouts, disconnected external folders or individually moved inputs.
After successful setup, `./transcribe-translate` resumes compatible saved results;
source and target TXT remain directly in Transcribed and durable recovery remains
in Work. Do not delete old namespaces or original files to make the directory tidy.

## Device and quality boundaries

Setup reports three separate facts: dependency set installed; configured ASR
model/device/precision; and validation actually performed. A CPU-only development
host installing CPU dependencies leaves existing `auto`, CUDA, language, precision
and optional API choices unchanged. A fresh ordinary install uses large-v3 and
`auto/auto`; runtime prefers supported CUDA, otherwise CPU with the same model.
Explicit `--cpu` selects CPU dependencies and only a newly created config gets a
CPU device default. Explicit `--cuda` requires eligible detection and cannot quietly
succeed as CPU. Runtime explicit CUDA failure is also reported without fallback.

The released CUDA policy is unchanged: pre-Blackwell Linux driver 525.60.13+,
including reported RTX 3080/550.163.01; Blackwell 10.0/12.0 requires 570.26+.
Unknown/unsupported devices and detection failures retain their existing handling.
Minor compatibility has kernel/PTX/newer-driver-feature limits. CUDA package
installation or `--check` is eligibility/capability evidence, not kernel execution.
Normal eligible CUDA setup prepares the selected weights and performs the existing
bounded CUDA smoke with the configured precision. `--skip-model-downloads` defers
preparation/smoke. Manual `--cuda-smoke-check` uses cached weights only.
See [GPU acceptance](GPU_ACCEPTANCE.md) for the pinned stack and exact checks.
CPU tests here cannot establish Kraken GPU behavior. Successful setup, smoke or
inference does not establish transcription/translation fidelity; the documented
[semantic limitations](QUALITY_FOLLOWUP.md) remain.
