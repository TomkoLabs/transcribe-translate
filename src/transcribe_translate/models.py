"""Portable selected weights and process-scoped download caches."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil

from .managed import no_links, operation, owner
from .storage import StateError, file_hash, sync_directory

PIN = json.loads(Path(__file__).with_name('large_v3_manifest.json').read_text())


def legacy_cache() -> Path:
    return Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache'))/'huggingface/hub'


def environment_path(config, key, default):
    path=Path(os.environ.get(key,str(default))).expanduser()
    return (path if path.is_absolute() else config.root/path).resolve()


def locations(config) -> dict:
    external_model = Path(config.asr_model) if Path(config.asr_model).is_absolute() else None
    selected = external_model or (config.shared_hub or config.models_root)
    actual = external_model or (config.shared_hub/f"models--{PIN['repository'].replace('/','--')}/snapshots/{PIN['revision']}" if config.shared_hub and config.asr_model in {'large','large-v3'} else model_path(config))
    data={'models': str(selected), 'models_scope': 'project-local' if selected.is_relative_to(config.root) else 'external/shared',
          'model_path': str(actual), 'installation_cache': str(config.installation_cache),
          'hf_transfer_cache': str(config.installation_cache/'huggingface'),
          'legacy_hf_cache': str(config.shared_hub or legacy_cache())}
    for key,env,default in [('pip_cache','PIP_CACHE_DIR',config.installation_cache/'pip'),
                            ('hf_xet_cache','HF_XET_CACHE',config.installation_cache/'huggingface/xet'),
                            ('hf_assets_cache','HF_ASSETS_CACHE',config.installation_cache/'huggingface/assets')]:
        path=environment_path(config,env,default)
        data[key]=str(path)
        data[key+'_scope']='project-local' if path.is_relative_to(config.root) else 'external/shared'
    return data


def model_path(config) -> Path:
    return config.models_root/(f"Whisper-large-v3--{PIN['revision']}" if config.asr_model in {'large','large-v3'} else 'Whisper-named-models')


def configure_transfer(config):
    # Explicit shared cache settings are retained. No HOME or global config edits.
    for key,child in [('HF_XET_CACHE','xet'),('HF_ASSETS_CACHE','assets')]:
        marker='TT_MANAGED_'+key
        if key not in os.environ or os.environ[key]==os.environ.get(marker):
            os.environ[key]=str(config.installation_cache/'huggingface'/child)
            os.environ[marker]=os.environ[key]
        else:
            os.environ[key]=str(environment_path(config,key,config.installation_cache/'huggingface'/child))
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')


@contextmanager
def cache_operation(config):
    area = config.installation_cache
    no_links(area, config.root)
    area.mkdir(mode=0o700,parents=True,exist_ok=True)
    fd = os.open(area,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StateError('Installation caches are in use; wait and rerun') from exc
        with operation(area, 'installation-cache-v1', mutable=('pip','huggingface','runtime-links.json')):
            configure_transfer(config)
            yield area
    finally:
        os.close(fd)


def space(path: Path, needed: int):
    ancestor = path
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < needed + 512*1024**2:
        raise StateError(f'Insufficient model-storage disk space: need {needed/1024**3:.2f} GiB plus 0.5 GiB reserve at {path}')


def verify_pinned(folder: Path):
    for name, expected in PIN['files'].items():
        p = folder/name
        if not p.is_file() or p.stat().st_size != expected['size'] or file_hash(p) != expected['sha256']:
            raise StateError(f'Incomplete/edited pinned large-v3 weights: {p}')


def complete_legacy(folder: Path) -> bool:
    if not all((folder/name).is_file() for name in PIN['files']):
        return False
    verify_pinned(folder)
    return True


def check(config):
    if config.asr_model in {'large','large-v3'}:
        path = Path(locations(config)['model_path'])
        if path.exists():
            verify_pinned(path)
            return f'Pinned weights verified: {path}'
        return f'Weights not prepared at {path}; run ./setup.sh or --prepare-model'
    return 'Selected optional/custom weights are checked when prepared/loaded'


def prepare(config, *, local_files_only=False, progress=print) -> Path:
    if Path(config.asr_model).is_absolute():
        progress(f'Reused explicit converted weights: {config.asr_model}')
        return Path(config.asr_model)
    configure_transfer(config)
    # Serialize prefetch/inference imports across batches with different outputs.
    config.models_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(config.models_root, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _prepare(config,local_files_only=local_files_only,progress=progress)
    except BlockingIOError as exc:
        raise StateError('Another process is preparing models; wait and rerun') from exc
    finally:
        os.close(fd)


def _prepare(config, *, local_files_only, progress):
    if config.asr_model not in {'large','large-v3'}:
        from faster_whisper.utils import download_model, _MODELS
        def fetch(cache):
            try:
                result=Path(download_model(config.asr_model,cache_dir=str(cache),local_files_only=True))
                progress(f'Reused selected optional weights: {result}')
            except FileNotFoundError:
                if local_files_only:
                    raise
                result=Path(download_model(config.asr_model,cache_dir=str(cache)))
                progress(f'Downloaded selected optional weights: {result}')
            return result
        if config.shared_hub:
            with cache_operation(config):
                return fetch(config.shared_hub)
        hub = model_path(config)
        repository = _MODELS.get(config.asr_model,config.asr_model)
        with cache_operation(config), operation(hub,'named-model-cache-v1',mutable=('models--'+repository.replace('/','--'),'.locks')):
            return fetch(hub)
    if config.shared_hub:
        from huggingface_hub import snapshot_download
        with cache_operation(config):
            try:
                result = Path(snapshot_download(PIN['repository'],revision=PIN['revision'],cache_dir=config.shared_hub,allow_patterns=list(PIN['files']),local_files_only=True))
            except Exception as exc:
                if local_files_only:
                    raise RuntimeError('Cached shared ASR weights missing/incomplete; CUDA smoke never downloads weights') from exc
                if not isinstance(exc,FileNotFoundError):
                    raise
                result = None
            if result is not None:
                verify_pinned(result)
                progress(f'Reused explicit shared pinned weights: {result}')
                return result
            space(config.shared_hub,sum(v['size'] for v in PIN['files'].values()))
            result = Path(snapshot_download(PIN['repository'],revision=PIN['revision'],cache_dir=config.shared_hub,allow_patterns=list(PIN['files'])))
            verify_pinned(result)
            progress(f'Downloaded pinned weights to explicit shared cache: {result}')
            return result
    target = model_path(config)
    if target.exists() or target.is_symlink():
        owner(target,'pinned-model-v1')
        verify_pinned(target)
        progress(f'Reused verified project weights: {target}')
        return target
    stage = target.with_name('.'+target.name+'.installing')
    # Stable staging plus Hub's partial metadata supports interruption recovery.
    # Completed files are checked against the pinned upstream manifest before reuse.
    source = legacy_cache()/f"models--{PIN['repository'].replace('/','--')}/snapshots/{PIN['revision']}"
    legacy = complete_legacy(source)
    if local_files_only and not legacy and not stage.exists():
        raise RuntimeError('Cached ASR weights missing/incomplete; CUDA smoke never downloads weights')
    valid = {n for n,v in PIN['files'].items() if (stage/n).is_file() and (stage/n).stat().st_size==v['size'] and file_hash(stage/n)==v['sha256']}
    needed = sum(v['size'] for n,v in PIN['files'].items() if n not in valid)
    space(config.models_root,needed)
    with cache_operation(config), operation(stage,'pinned-model-v1',mutable=(*PIN['files'],*(n+'.part' for n in PIN['files']),'.cache')):
        if any(p.is_symlink() for p in stage.rglob('*')):
            raise StateError('Linked model staging is refused; preserve it')
        if legacy:
            for name, expected in PIN['files'].items():
                p = stage/name
                if p.is_file() and p.stat().st_size==expected['size'] and file_hash(p)==expected['sha256']:
                    continue
                # Dereference snapshot/blob links: Models contains independent files.
                partial = stage/(name+'.part')
                shutil.copyfile(source/name,partial,follow_symlinks=True)
                if file_hash(partial)!=expected['sha256']:
                    raise StateError('Shared weights changed during import; original retained')
                with partial.open('rb') as stream:
                    os.fsync(stream.fileno())
                os.replace(partial,p)
            progress('Imported pinned weights from legacy cache; the additional shared copy remains unchanged')
        elif not all((stage/n).is_file() for n in PIN['files']):
            if local_files_only:
                raise RuntimeError('Cached ASR weights incomplete; CUDA smoke never downloads weights')
            from huggingface_hub import snapshot_download
            snapshot_download(PIN['repository'],revision=PIN['revision'],allow_patterns=list(PIN['files']),
                              local_dir=stage,cache_dir=config.installation_cache/'huggingface/hub')
            progress('Downloaded pinned weights to project Models staging')
        else:
            progress('Reused completed model staging after interruption')
        verify_pinned(stage)
    # An owned, complete staging directory is published on its filesystem.
    sync_directory(stage)
    os.rename(stage,target)
    sync_directory(target.parent)
    return target
