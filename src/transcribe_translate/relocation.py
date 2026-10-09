"""Copy verified Work state to relocated relative paths; never retire originals."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import tomllib

from .migration import validate_job
from .publication import validate_registry, verify_public
from .storage import StateError, digest, file_hash, job_name, load, save, sync_directory


def namespace(work: Path, output: Path) -> Path:
    label = output.name.lstrip('.').encode()[:70].decode(errors='ignore') or 'output'
    return work / f'{label}--{digest(str(output))[:16]}'


def relocate(config, previous: Path, *, progress=print):
    """Called by setup under installation/output locks, after runtime validation.

    Only the Work layout is copied. Legacy layouts and interrupted migrations
    remain untouched and require resuming at their original paths first.
    """
    root = config.root
    previous = Path(os.path.abspath(previous))
    if previous == root:
        return
    local = root / 'config.local.toml'
    values = tomllib.loads(local.read_text()) if local.exists() else {}
    relative = {key: not Path(values.get(key, default)).expanduser().is_absolute()
                for key, default in [('source_folder', 'Video Source'),
                                     ('output_folder', 'Transcribed'), ('work_folder', 'Work')]}
    if (not relative['output_folder'] or not relative['work_folder']
            or not config.output.is_relative_to(root) or not config.work_root.is_relative_to(root)):
        progress('Relocation: absolute/external output or Work paths retained; no state rebinding performed')
        return
    old_output = previous / config.output.relative_to(root)
    source = namespace(config.work_root, old_output)
    target = namespace(config.work_root, config.output)
    if not source.exists() and not source.is_symlink():
        progress('Relocation: no copied Work namespace for the previous output path; legacy state retained')
        return
    if source.is_symlink() or load(source / 'owner.json') != {'layout': 'work-state-v1', 'output_path': str(old_output)}:
        raise StateError('Relocation refused: copied Work ownership does not match the previous output')
    journal = source / 'migration.json'
    if journal.exists() and load(journal).get('phase') != 'complete':
        raise StateError('Finish the interrupted legacy migration at its original paths before relocating')
    # Reject links/special files before copying, and compare every original file
    # afterward. Neither the original checkout nor its runtime is read or changed.
    hashes = {}
    for path in source.rglob('*'):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise StateError(f'Relocation refused: symlink/special artifact {path}')
        if path.is_file():
            hashes[path.relative_to(source)] = file_hash(path)
    old_source = previous / config.source.relative_to(root) if relative['source_folder'] and config.source.is_relative_to(root) else None

    def identity(value):
        path = Path(value)
        if old_source is not None and path.is_relative_to(old_source):
            return str((config.source / path.relative_to(old_source)).resolve())
        return value  # Explicit absolute/external input identities are preserved.

    jobs = source / 'inputs'
    mappings = {}
    if jobs.exists():
        for job in jobs.iterdir():
            validate_job(job)
            old = load(job / 'owner.json')['input_path']
            new = identity(old)
            if new in mappings.values():
                raise StateError('Relocation refused: colliding input identities')
            mappings[old] = new
    registry = source / 'publications.json'
    public = validate_registry(load(registry)) if registry.exists() else None
    if public is not None:
        changed = {}
        for entry in public['inputs'].values():
            for language, artifact in entry['files'].items():
                verify_public(config.output, f"{entry['stem']}.{language}.txt", artifact)
            entry['input_path'] = identity(entry['input_path'])
            key = digest(entry['input_path'])
            if key in changed:
                raise StateError('Relocation refused: colliding publication identities')
            changed[key] = entry
        public['inputs'] = changed
        validate_registry(public)
    if target.exists() or target.is_symlink():
        if (target.is_symlink() or load(target / 'owner.json') != {'layout': 'work-state-v1', 'output_path': str(config.output)}
                or load(target / 'relocation.json') != {'schema': 1, 'previous_output': str(old_output),
                    'output_path': str(config.output), 'original_files': {str(p):h for p,h in hashes.items()}}):
            raise StateError('Relocation refused: destination Work namespace already exists; preserve both states')
        if (target / 'inputs').exists():
            for job in (target / 'inputs').iterdir():
                validate_job(job)
        if (target / 'publications.json').exists():
            for entry in validate_registry(load(target / 'publications.json'))['inputs'].values():
                for language, artifact in entry['files'].items():
                    verify_public(config.output, f"{entry['stem']}.{language}.txt", artifact)
        progress(f'Relocation: retaining already verified copy {target}')
        return
    staging = Path(tempfile.mkdtemp(prefix='.relocation-', dir=config.work_root))
    # Failed copies are retained for inspection. A completed copy appears in one
    # atomic rename; rerun never selects a half-published namespace.
    progress(f'Relocation staging: {staging}')
    shutil.copytree(source, staging, dirs_exist_ok=True)
    if any(file_hash(staging / p) != h or file_hash(source / p) != h for p, h in hashes.items()):
        raise StateError('Relocation refused: checkpoint changed during copy')
    save(staging / 'owner.json', {'layout': 'work-state-v1', 'output_path': str(config.output)})
    if jobs.exists():
        for old, new in mappings.items():
            job = staging / 'inputs' / job_name(Path(old))
            save(job / 'owner.json', {'input_path': new})
            for record_file in job.glob('*/record.json'):
                record = load(record_file)
                record['input_path'] = new
                save(record_file, record)
            destination = job.with_name(job_name(Path(new)))
            if destination != job:
                if destination.exists() or destination.is_symlink():
                    raise StateError('Relocation refused: job-name collision')
                job.rename(destination)
            validate_job(staging / 'inputs' / job_name(Path(new)))
    if public is not None:
        save(staging / 'publications.json', public)
    if journal.exists():
        # Keep the completed historical migration journal at the old namespace;
        # it refers to old output paths and has no pending recovery work.
        (staging / 'migration.json').unlink()
    save(staging / 'relocation.json', {'schema': 1, 'previous_output': str(old_output),
                                     'output_path': str(config.output), 'original_files': {str(p):h for p,h in hashes.items()}})
    current = {}
    for path in source.rglob('*'):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise StateError('Relocation refused: original state changed during validation')
        if path.is_file():
            current[path.relative_to(source)] = file_hash(path)
    if current != hashes:
        raise StateError('Relocation refused: original state changed during validation')
    sync_directory(staging)
    os.rename(staging, target)
    sync_directory(target.parent)
    progress(f'Relocated verified Work copy: {target}; original namespace retained: {source}')
