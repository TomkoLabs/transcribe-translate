"""Journaled migration of recognized owned output state into visible Work."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from . import asr, translation
from .publication import upgrade_registry, valid_hash, validate_registry
from .storage import StateError, atomic_bytes, digest, file_hash, job_name, load, private_dir, save, sync_directory


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _tree(folder: Path) -> dict[str, str]:
    if folder.is_symlink() or not folder.is_dir():
        raise StateError(f"Unowned or symlink migration folder: {folder}")
    files = {}
    for path in folder.rglob('*'):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise StateError(f"Unowned or symlink migration artifact: {path}")
        if path.is_file():
            files[path.relative_to(folder).as_posix()] = file_hash(path)
    return files


def validate_job(job: Path) -> dict[str, str]:
    """Reuse saved-result validators, and reject extra files before any retirement."""
    from .pipeline import Processor
    files = _tree(job)
    owner = load(job / 'owner.json')
    identity = owner.get('input_path')
    if (not isinstance(identity, str) or set(owner) != {'input_path'}
            or not Path(identity).is_absolute() or job.name != job_name(Path(identity))):
        raise StateError(f"Invalid input ownership: {job}")
    allowed = {'owner.json'}
    if 'index.json' not in files and set(files) != {'owner.json'}:
        raise StateError(f"Missing selected-generation index: {job}; preserve the state")
    index = load(job / 'index.json') if 'index.json' in files else {'asr': {}, 'translations': {}}
    if 'index.json' in files:
        allowed.add('index.json')
    if any(not isinstance(index.get(k), dict) or any(not valid_hash(a) or not valid_hash(b)
           for a,b in index[k].items()) for k in ['asr','translations']):
        raise StateError(f"Invalid generation index: {job}")
    cache_names = [name for name in ['.asr', 'asr'] if (job / name).exists()]
    if len(cache_names) > 1:
        raise StateError(f"Ambiguous ASR cache folders: {job}")
    caches = {}
    for cache_name in cache_names:
        for path in (job / cache_name).iterdir():
            relative = path.relative_to(job).as_posix()
            if path.is_file() and path.suffix == '.json' and valid_hash(path.stem):
                cached = load(path)
                try:
                    expected = digest({'input': {k: cached['input'][k] for k in ['sha256','size']}, 'asr': cached['settings']})
                    if cached['signature'] != expected:
                        raise ValueError('cache signature')
                    asr.validate_result(cached['result'])
                except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                    raise StateError(f"Invalid source cache: {path}") from exc
                caches[path.stem] = cached
                allowed.add(relative)
            elif path.is_file() and re.fullmatch(r'\.audio-[\w-]+\.wav|\.write-[\w-]+\.tmp', path.name):
                allowed.add(relative)  # Owned orphan temporary audio/writes are preserved.
            else:
                raise StateError(f"Unrecognized files inside owned source cache: {path}; preserve them")
    for signature, cache_id in index['asr'].items():
        if cache_id in caches and caches[cache_id]['signature'] != signature:
            raise StateError(f"Selected source-cache settings mismatch: {job}")
    records = {}
    for folder in job.iterdir():
        if folder.name == 'temporary' and folder.is_dir():
            for path in folder.iterdir():
                if path.is_file() and re.fullmatch(r'audio-[\w-]+\.wav|write-[\w-]+\.tmp',path.name):
                    allowed.add(path.relative_to(job).as_posix())
                else:
                    raise StateError(f"Unrecognized files inside owned temporary state: {path}; preserve them")
            continue
        if not folder.is_dir() or folder.name in cache_names:
            continue
        record_file = folder / 'record.json'
        if not record_file.is_file():
            raise StateError(f"Unrecognized files inside owned state: {folder}; preserve them")
        record = load(record_file)
        try:
            cache_id = record['asr_cache_id']
            source, target = record['source_language'], record['target_language']
            settings = record['translation_settings']
            if (not valid_hash(record['run_id']) or not valid_hash(record['signature'])
                    or record['input_path'] != identity or cache_id not in caches
                    or source != caches[cache_id]['result']['language']
                    or record['source_segments'] != caches[cache_id]['result']['segments']
                    or record['status'] not in {'translating','failed','interrupted','complete'}):
                raise ValueError('record identity')
            backend = record.get('translation_backend', settings.get('backend', 'api'))
            label = 'local--' if backend == 'local' else ''
            if folder.name != f"{source}-to-{target}--{label}{record['run_id'][:16]}":
                raise ValueError('generation name')
            asr.validate_result({'language':source,'segments':record['source_segments']})
            source_text = asr.render_source(record['source_segments'])
            signature = digest({'asr_cache':cache_id, 'source_text':_text_hash(source_text), 'translation':settings})
            if record['signature'] != signature:
                raise ValueError('translation signature')
            plan = []
            if source != target and backend == 'api':
                limit, context = settings['chunk_chars'], settings['context_chars']
                if not isinstance(limit,int) or not 50 <= limit <= 2000 or not isinstance(context,int) or not 0 <= context <= 300:
                    raise ValueError('chunk settings')
                plan = translation.chunks(source_text, limit, context)
            Processor._check_chunks(record, plan)
            native = record.get('native_translation')
            if native is not None:
                asr.validate_result(native)
                if native['language'] != 'en':
                    raise ValueError('native language')
            source_name, target_name = f'source.{source}.txt', f'translated.{target}.txt'
            outputs = record['outputs']
            if not isinstance(outputs,dict) or outputs.get(source_name) != _text_hash(source_text):
                raise ValueError('source checksum')
            if source == target:
                target_text = source_text
            elif backend == 'local':
                target_text = asr.render_source(native['segments']) if native is not None else None
            else:
                target_text = '\n\n'.join(c['translation'] for c in record['chunks']) + '\n' if len(record['chunks']) == len(plan) else None
            if ((record['status'] == 'complete' or target_name in outputs)
                    and (target_text is None or outputs.get(target_name) != _text_hash(target_text))):
                raise ValueError('incomplete completed result')
            for name, expected in outputs.items():
                if name not in {source_name,target_name} or not valid_hash(expected):
                    raise ValueError('artifact name/checksum')
                artifact = folder / name
                if artifact.exists() and file_hash(artifact) != expected:
                    raise ValueError('edited TXT')
            for name in [source_name, target_name]:
                if (folder / name).exists() and name not in outputs:
                    raise ValueError('unowned TXT')
            records[record['run_id']] = record
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise StateError(f"Invalid, edited or incomplete generation state: {record_file}") from exc
        allowed.update(f'{folder.name}/{name}' for name in ['record.json',source_name,target_name])
    for signature, run_id in index['translations'].items():
        if run_id in records and records[run_id]['signature'] != signature:
            raise StateError(f"Selected-generation settings mismatch: {job}")
    if set(files) - allowed:
        raise StateError(f"Unrecognized files inside owned state: {job}; preserve them")
    return files


def _safe_path(root: Path, name: str) -> Path:
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise StateError('Unsafe migration journal path')
    path = root
    for part in p.parts:
        path = path / part
        if path.is_symlink():
            raise StateError(f'Symlink migration path: {path}')
    return path


def _prepare(output: Path, state: Path) -> dict:
    hidden = output / '.state'
    sources = []
    operations = []
    directories = []
    registry, public = {'schema':2,'inputs':{}}, []
    if hidden.exists() or hidden.is_symlink():
        if hidden.is_symlink() or load(hidden / 'owner.json') != {'layout':'hidden-state-v1'}:
            raise StateError('Unowned or modified legacy .state folder')
        operations.append({'source':'.state/owner.json','target':'legacy-layout.json','sha256':file_hash(hidden/'owner.json')})
        directories.append('.state')
        for path in hidden.iterdir():
            if path.name == 'owner.json':
                continue
            if path.name == 'publications.json':
                registry, public = upgrade_registry(load(path), output)
                operations.append({'source':'.state/publications.json','target':'legacy-publications.json','sha256':file_hash(path)})
            elif path.is_dir() and (path/'owner.json').is_file():
                sources.append(path)
            else:
                raise StateError(f'Unrecognized or modified hidden state: {path}; preserve it')
    for path in output.iterdir():
        if path != hidden and re.search(r'--[0-9a-f]{16}$',path.name) and path.is_dir() and (path/'owner.json').is_file():
            sources.append(path)
    seen = set()
    for job in sources:
        if job.name in seen or (state/'inputs'/job.name).exists():
            raise StateError(f'Ambiguous legacy and Work state: {job}; preserve both')
        seen.add(job.name)
        files = validate_job(job)
        directories.extend(p.relative_to(output).as_posix() for p in [job,*job.rglob('*')] if p.is_dir())
        for name, checksum in files.items():
            target = name.replace('.asr/', 'asr/', 1) if name.startswith('.asr/') else name
            if '/.audio-' in target or '/.write-' in target:
                target = 'temporary/' + Path(target).name.lstrip('.')
            operations.append({'source':(job.relative_to(output)/name).as_posix(),
                               'target':f'inputs/{job.name}/{target}', 'sha256':checksum})
    if operations and (state/'publications.json').exists():
        raise StateError('Ambiguous legacy publication and Work state; preserve both')
    original_names = {move['old'] for move in public}
    for move in public:
        destination = output / move['new']
        if move['new'] not in original_names and (destination.exists() or destination.is_symlink()):
            raise StateError(f'Unrelated transcript collision during migration: {destination}')
    return {'schema':1,'output_path':str(output.resolve()),'operations':operations,
            'directories':directories,'publication':registry,'public':public,'phase':'copying'}


def _verify_copy(path: Path, checksum: str) -> bool:
    if not path.exists():
        return False
    if not path.is_file() or file_hash(path) != checksum:
        raise StateError(f'Modified migration copy: {path}; preserve it')
    return True


def _resume(output: Path, state: Path, journal: dict, journal_file: Path) -> None:
    if (journal.get('schema') != 1 or journal.get('output_path') != str(output.resolve())
            or journal.get('phase') not in {'copying','published','retiring','complete'}
            or not isinstance(journal.get('operations'),list) or not isinstance(journal.get('public'),list)):
        raise StateError('Invalid migration journal')
    validate_registry(journal['publication'])
    operations = journal['operations']
    temporary = state / 'temporary'
    private_dir(temporary)
    for op in operations:
        if not valid_hash(op.get('sha256')):
            raise StateError('Invalid migration checksum')
        source, target = _safe_path(output, op['source']), _safe_path(state, op['target'])
        if journal['phase'] == 'copying' and source.exists() and not _verify_copy(source, op['sha256']):
            raise StateError('Missing migration source')
        if not _verify_copy(target, op['sha256']):
            if journal['phase'] != 'copying' or not _verify_copy(source, op['sha256']):
                raise StateError(f'Missing migration source and copy: {source}')
            target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            atomic_bytes(target, source.read_bytes())
    # Back up owned public files before any paired-stem rename or retirement.
    for move in journal['public']:
        if not valid_hash(move.get('sha256')) or Path(move['old']).name != move['old'] or Path(move['new']).name != move['new']:
            raise StateError('Invalid public migration filename')
        backup = _safe_path(state, f"migration-public/{move['old']}")
        if not _verify_copy(backup, move['sha256']):
            original = _safe_path(output, move['old'])
            if journal['phase'] != 'copying' or not _verify_copy(original, move['sha256']):
                raise StateError(f'Missing public migration backup: {original}')
            backup.parent.mkdir(mode=0o700, exist_ok=True)
            atomic_bytes(backup, original.read_bytes())
    # All copied owned generations must still pass structural/content checks.
    inputs = state / 'inputs'
    if inputs.is_symlink():
        raise StateError('Refusing symlink migrated inputs')
    for job in inputs.iterdir() if inputs.exists() else []:
        validate_job(job)
    if journal['phase'] == 'copying':
        old_hashes = {move['old']:move['sha256'] for move in journal['public']}
        for move in journal['public']:
            target = _safe_path(output, move['new'])
            if target.exists():
                actual = file_hash(target)
                if actual not in {move['sha256'],old_hashes.get(move['new'])}:
                    raise StateError(f'Edited transcript during migration: {target}')
            if not target.exists() or file_hash(target) != move['sha256']:
                atomic_bytes(target, _safe_path(state,f"migration-public/{move['old']}").read_bytes(), temporary=temporary)
        save(state/'publications.json',journal['publication'])
        journal['phase'] = 'published'
        save(journal_file,journal)
    new_names = {move['new'] for move in journal['public']}
    for move in journal['public']:
        target = _safe_path(output,move['new'])
        if not _verify_copy(target,move['sha256']):
            atomic_bytes(target,_safe_path(state,f"migration-public/{move['old']}").read_bytes(),temporary=temporary)
        if move['old'] not in new_names:
            original = _safe_path(output,move['old'])
            if original.exists():
                _verify_copy(original,move['sha256'])
                original.unlink()
                sync_directory(original.parent)
    journal['phase'] = 'retiring'
    save(journal_file,journal)
    # Re-check each remaining original before unlinking; missing originals resume safely.
    for op in sorted(operations,key=lambda op: Path(op['source']).name == 'owner.json'):
        original = _safe_path(output,op['source'])
        if original.exists():
            if not _verify_copy(_safe_path(state,op['target']),op['sha256']):
                raise StateError('Migrated state disappeared before retirement')
            _verify_copy(original,op['sha256'])
            original.unlink()
            sync_directory(original.parent)
    parents = {parent for op in operations for parent in (output/op['source']).parents if parent != output and parent.is_relative_to(output)}
    parents.update(_safe_path(output,name) for name in journal['directories'])
    for parent in sorted(parents,key=lambda p:len(p.parts),reverse=True):
        if parent.exists():
            try:
                parent.rmdir()
                sync_directory(parent.parent)
            except OSError as exc:
                raise StateError(f'Unrelated files appeared during migration: {parent}; preserved') from exc
    journal['phase'] = 'complete'
    save(journal_file,journal)


def migrate_output(output: Path, state: Path, progress=print) -> None:
    if output.stat().st_dev != state.stat().st_dev:
        raise StateError('Work and output must share a filesystem for atomic TXT publication; configure work_folder on the output mount')
    journal_file = state / 'migration.json'
    if journal_file.exists() or journal_file.is_symlink():
        journal = load(journal_file)
        if journal.get('phase') != 'complete':
            progress('Resuming verified output-state migration into Work')
            _resume(output,state,journal,journal_file)
    journal = _prepare(output,state)
    if journal['operations']:
        progress('Migrating owned recovery state into visible Work (copy, verify, retire)')
        save(journal_file,journal)
        _resume(output,state,journal,journal_file)
    leftovers = [p.name for p in output.iterdir() if p.name != '.transcribe-translate.lock' and (not p.is_file() or p.suffix != '.txt')]
    if leftovers:
        progress('Unrelated non-TXT output entries preserved: ' + ', '.join(sorted(leftovers)))
