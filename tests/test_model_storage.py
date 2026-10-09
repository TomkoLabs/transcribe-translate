"""Model/storage/uninstall safety tests with tiny byte fixtures, never speech quality."""
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from transcribe_translate import managed, models, asr
from transcribe_translate.config import load_config
from transcribe_translate.storage import load, save

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('remove_project',ROOT/'scripts/remove.py')
remover=importlib.util.module_from_spec(spec);spec.loader.exec_module(remover)


@pytest.fixture
def pin(config,tmp_path,monkeypatch):
    files={name:(name+' fixture bytes').encode() for name in models.PIN['files']}
    data={'repository':models.PIN['repository'],'revision':models.PIN['revision'],
          'files':{n:{'size':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in files.items()}}
    monkeypatch.setattr(models,'PIN',data)
    cache=tmp_path/'shared'
    monkeypatch.setattr(models,'legacy_cache',lambda:cache)
    source=cache/f"models--{data['repository'].replace('/','--')}/snapshots/{data['revision']}"
    blobs=source.parent.parent/'blobs';blobs.mkdir(parents=True);source.mkdir(parents=True)
    for name,b in files.items():
        blob=blobs/data['files'][name]['sha256'];blob.write_bytes(b)
        (source/name).symlink_to(blob)
    return data,source,files


def test_import_independent_offline_and_storage_signature_unchanged(config,pin,monkeypatch):
    before={p.name:p.read_bytes() for p in pin[1].iterdir()}
    identity=asr.model_identity(config)
    result=models.prepare(config)
    assert result.is_relative_to(config.root/'Models')
    assert all(not p.is_symlink() for p in result.iterdir())
    assert all((result/n).read_bytes()==b for n,b in pin[2].items())
    assert before=={p.name:p.read_bytes() for p in pin[1].iterdir()}
    shutil.rmtree(pin[1].parent.parent)  # Fixture only: prove copied links are not dependencies.
    assert models.prepare(config,local_files_only=True)==result
    assert asr.model_identity(replace(config,models=config.root/'different Models'))==identity
    assert 'verified' in models.check(config)


def test_interrupted_import_and_atomic_promotion_retry(config,pin,monkeypatch):
    original=models.os.rename
    target=models.model_path(config)
    def stop(source,destination):
        if destination==target:raise OSError('interrupted promotion')
        return original(source,destination)
    monkeypatch.setattr(models.os,'rename',stop)
    with pytest.raises(OSError,match='promotion'):models.prepare(config)
    assert not target.exists()
    monkeypatch.setattr(models.os,'rename',original)
    assert models.prepare(config)==target


def test_download_uses_only_selected_pin_and_local_caches(config,pin,monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(models,'legacy_cache',lambda:config.root/'unavailable')
    calls=[]
    def download(repo,**kw):
        calls.append(kw);assert repo==pin[0]['repository'] and kw['revision']==pin[0]['revision']
        assert set(kw['allow_patterns'])==set(pin[2])
        for n,b in pin[2].items():(kw['local_dir']/n).write_bytes(b)
        return str(kw['local_dir'])
    monkeypatch.setattr(huggingface_hub,'snapshot_download',download)
    models.prepare(config)
    assert len(calls)==1 and calls[0]['cache_dir'].is_relative_to(config.installation_cache)
    models.prepare(config);assert len(calls)==1
    assert models.locations(config)['hf_xet_cache'].startswith(str(config.root))


def test_model_tampering_and_unrelated_files_prevent_cleanup(config,pin):
    result=models.prepare(config)
    extra=result/'private notes.txt';extra.write_text('retain')
    with pytest.raises(RuntimeError,match='unrelated'):managed.remove(result,config.root,'pinned-model-v1')
    assert extra.read_text()=='retain' and (result/'model.bin').exists()
    extra.unlink();(result/'model.bin').write_bytes(b'edited')
    with pytest.raises(RuntimeError,match='edited'):models.prepare(config)
    with pytest.raises(RuntimeError,match='unrelated'):managed.remove(result,config.root,'pinned-model-v1')


def test_relative_external_locations_and_explicit_choices(config,tmp_path,monkeypatch):
    monkeypatch.setenv('TT_ROOT',str(config.root));monkeypatch.setenv('HF_HUB_CACHE',str(tmp_path/'external shared'))
    c=load_config();assert c.shared_hub==tmp_path/'external shared'
    (config.root/'config.local.toml').write_text('models_folder="Local weights with spaces"\n')
    monkeypatch.chdir(tmp_path.parent)
    c=load_config();assert c.shared_hub is None and c.models_root==config.root/'Local weights with spaces'
    c=replace(c,models=tmp_path.parent/'external weights')
    assert models.locations(c)['models_scope']=='external/shared'


def test_cleanup_never_follows_links_and_runtime_keeps_unrecorded_data(config,tmp_path):
    external=tmp_path/'external';external.mkdir();(external/'keep').write_text('safe')
    linked=config.root/'linked';linked.symlink_to(external,target_is_directory=True)
    with pytest.raises(RuntimeError,match='linked'):managed.remove(linked,config.root,'runtime-v1')
    assert (external/'keep').read_text()=='safe'
    runtime=config.root/'.venv';runtime.mkdir();(runtime/'pyvenv.cfg').write_text('test')
    (runtime/'credentials.env').write_text('fixture secret')
    managed.runtime_stamp(runtime)
    with pytest.raises(RuntimeError,match='unrelated'):managed.remove(runtime,config.root,'runtime-v1')
    assert (runtime/'credentials.env').read_text()=='fixture secret'


def test_uninstall_default_remove_models_repeat_and_preservation(config,pin,monkeypatch):
    monkeypatch.setattr(remover,'ROOT',config.root);monkeypatch.setenv('TT_ROOT',str(config.root))
    monkeypatch.setattr(remover,'active_runtime',lambda _:None)
    model=models.prepare(config)
    for folder in ['Video Source','Transcribed','Work']:
        (config.root/folder).mkdir(exist_ok=True);(config.root/folder/'keep.txt').write_text('preserved')
    (config.root/'config.local.toml').write_text('source_language="zh"\nasr_model="large-v3"\nasr_device="auto"\n')
    runtime=config.root/'.venv'
    with managed.operation(runtime,'runtime-v1'):(runtime/'broken-python').write_text('owned fixture, not an interpreter')
    monkeypatch.setattr(sys,'argv',['uninstall','--root',str(config.root)])
    assert remover.main()==0 and not runtime.exists() and model.exists()
    assert remover.main()==0
    monkeypatch.setattr(sys,'argv',['uninstall','--root',str(config.root),'--remove-models'])
    assert remover.main()==0 and not model.exists()
    assert remover.main()==0
    assert all((config.root/f/'keep.txt').read_text()=='preserved' for f in ['Video Source','Transcribed','Work'])
    assert 'source_language="zh"' in (config.root/'config.local.toml').read_text()
    assert all((pin[1]/n).read_bytes()==b for n,b in pin[2].items())


def test_low_disk_refuses_import_before_mutation(config,pin,monkeypatch):
    monkeypatch.setattr(models.shutil,'disk_usage',lambda _:type('Disk',(),{'free':1})())
    with pytest.raises(RuntimeError,match='Insufficient'):models.prepare(config)
    assert not models.model_path(config).exists()


def test_interrupted_owned_cache_recovers_only_reserved_paths(config):
    with models.cache_operation(config):
        (config.installation_cache/'pip').mkdir()
    marker=config.installation_cache/managed.MARKER
    data=load(marker);data['active']=True;save(marker,data)
    (config.installation_cache/'pip/partial').write_text('owned interrupted transfer')
    with models.cache_operation(config):pass
    (config.installation_cache/'unrelated.txt').write_text('retain')
    with pytest.raises(RuntimeError,match='unrelated'):
        with models.cache_operation(config):pass


def test_download_interruption_reuses_staging_without_promoting_incomplete(config,pin,monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(models,'legacy_cache',lambda:config.root/'not available')
    calls=[]
    def download(repo,**kw):
        calls.append(kw)
        first=next(iter(pin[2]))
        if len(calls)==1:
            (kw['local_dir']/first).write_bytes(pin[2][first])
            raise OSError('interrupted transfer')
        assert (kw['local_dir']/first).read_bytes()==pin[2][first]
        for n,b in pin[2].items():(kw['local_dir']/n).write_bytes(b)
        return str(kw['local_dir'])
    monkeypatch.setattr(huggingface_hub,'snapshot_download',download)
    with pytest.raises(OSError,match='interrupted'):models.prepare(config)
    assert not models.model_path(config).exists()
    assert models.prepare(config)==models.model_path(config)
    assert calls[0]['local_dir']==calls[1]['local_dir'] and len(calls)==2


def test_uninstall_refuses_active_installation_without_mutation(config,pin,monkeypatch):
    from transcribe_translate.installation import installation_lock
    model=models.prepare(config)
    monkeypatch.setattr(remover,'ROOT',config.root);monkeypatch.setenv('TT_ROOT',str(config.root))
    monkeypatch.setattr(sys,'argv',['uninstall','--root',str(config.root),'--remove-models'])
    with installation_lock(config.root):
        assert remover.main()==2
    assert (model/'model.bin').read_bytes()==pin[2]['model.bin']


def test_preserved_model_and_checkpoint_paths_cannot_overlap_disposable_cache(config,monkeypatch):
    monkeypatch.setenv('TT_ROOT',str(config.root))
    (config.root/'config.local.toml').write_text('models_folder="Work/Installation/models"\n')
    with pytest.raises(ValueError,match='outside'):load_config()


def test_interrupted_cache_external_link_is_refused_before_writes(config,tmp_path):
    with models.cache_operation(config):(config.installation_cache/'pip').mkdir()
    marker=config.installation_cache/managed.MARKER
    data=load(marker);data['active']=True;save(marker,data)
    external=tmp_path/'external confidential';external.write_text('retain')
    (config.installation_cache/'pip/external').symlink_to(external)
    with pytest.raises(RuntimeError,match='External link'):
        with models.cache_operation(config):pass
    assert external.read_text()=='retain'


def test_interrupted_cleanup_completes_from_remaining_owned_files(config,pin):
    result=models.prepare(config)
    (result/'tokenizer.json').unlink() # Simulate interruption after one owned deletion.
    assert managed.remove(result,config.root,'pinned-model-v1')
    assert not result.exists()
