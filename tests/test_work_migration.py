"""Simulated legacy layouts and interruption points; no model inference validation."""
from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from conftest import FakeASR, FakeTranslator, processor
from test_local_release import NativeASR, local_processor
from test_output_layout import forbid_inference, video
from transcribe_translate import cli, migration, storage
from transcribe_translate.config import load_config
from transcribe_translate.publication import Publication
from transcribe_translate.storage import StateError, batch_lock, digest, load, save, state_directory


def legacy_fixture(config, layout, backend='local', failed=False):
    """Generate owned checkpoints with doubles, then reproduce the released layouts."""
    c = replace(config, translation_backend=backend)
    path = video(c, 'Échange 中文 interview.mp4')
    recognizer = NativeASR()
    recognizer.text *= 4
    translator = FakeTranslator()
    p = local_processor(c, recognizer, translator=translator)
    _, old = p.process(path)
    if failed:
        if backend == 'local':
            recognizer.fail = RuntimeError('simulated native failure')
        else:
            translator.fail = 1
        with pytest.raises((RuntimeError, TimeoutError)):
            p.process(path, reprocess='translation')
        selected_name = next(r.parent.name for r in p.state.rglob('record.json') if load(r)['status']=='failed')
    else:
        _, selected = p.process(path, reprocess='translation')
        selected_name = selected.name
    entry = next(iter(load(p.state/'publications.json')['inputs'].values()))
    target_name = f"{entry['stem']}.en.txt"
    artifact = entry['files']['en']
    p.source_path.unlink()
    hidden = config.output / '.state'
    parent = hidden if layout == 'hidden' else config.output
    if layout == 'hidden':
        hidden.mkdir()
        save(hidden/'owner.json',{'layout':'hidden-state-v1'})
        raw = {'schema':1,'files':{digest({'input_path':str(path),'target_language':'en'}):
               {'input_path':str(path),'target_language':'en','filename':target_name,**artifact}}}
        save(hidden/'publications.json',raw)
    else:
        (config.output/target_name).unlink()
    job = p.state/'inputs'/old.parent.name
    legacy = parent / job.name
    job.rename(legacy)
    (legacy/'asr').rename(legacy/'.asr')
    os.utime(legacy/old.name,(2_000_000_000,2_000_000_000))
    contents = {f.relative_to(legacy).as_posix():f.read_bytes() for f in legacy.rglob('*') if f.is_file()}
    (config.output/'.transcribe-translate.lock').touch(mode=0o600)
    shutil.rmtree(c.work_root)
    return c,path,legacy,selected_name,contents


def assert_migrated(p, legacy, contents):
    copied = p.state/'inputs'/legacy.name
    for name,content in contents.items():
        name = name.replace('.asr/','asr/',1) if name.startswith('.asr/') else name
        assert (copied/name).read_bytes() == content
    assert not legacy.exists() and not (p.config.output/'.state').exists()
    assert all(x.is_file() and x.suffix=='.txt' for x in p.config.output.iterdir())
    assert load(p.state/'migration.json')['phase']=='complete'


@pytest.mark.parametrize('layout',['legacy','hidden'])
@pytest.mark.parametrize('backend',['local','api'])
def test_migrate_completed_selected_generation_and_republish(config,layout,backend):
    c,path,legacy,selected,contents = legacy_fixture(config,layout,backend)
    with batch_lock(c.output,c.work_root):
        p = local_processor(c,NativeASR(),translator=FakeTranslator())
        forbid_inference(p)
        status,run = p.process(path)
        assert status=='skipped' and run.name==selected
        assert p.source_path.name=='Échange 中文 interview.zh.txt'
        assert p.translated_path.name=='Échange 中文 interview.en.txt'
        saved = {f:f.read_bytes() for f in [p.source_path,p.translated_path]}
        for f in saved:f.unlink()
        assert p.process(path)[0]=='skipped'
        assert all(f.read_bytes()==content for f,content in saved.items())
    assert_migrated(p,legacy,contents)


@pytest.mark.parametrize('layout',['legacy','hidden'])
@pytest.mark.parametrize('point',['copy','publish','retire','rmdir'])
def test_interrupted_migration_resumes_without_inference(config,layout,point,monkeypatch):
    c,path,legacy,selected,contents = legacy_fixture(config,layout)
    atomic = migration.atomic_bytes
    saving = migration.save
    unlink = Path.unlink
    rmdir = Path.rmdir
    interrupted = []
    def copy(target,*args,**kwargs):
        atomic(target,*args,**kwargs)
        if point=='copy' and not interrupted:
            interrupted.append(1)
            raise KeyboardInterrupt()
    def save_journal(target,data):
        saving(target,data)
        if point=='publish' and target.name=='migration.json' and data['phase']=='published' and not interrupted:
            interrupted.append(1)
            raise KeyboardInterrupt()
    def remove(target,*args,**kwargs):
        unlink(target,*args,**kwargs)
        if point=='retire' and target.is_relative_to(legacy) and not interrupted:
            interrupted.append(1)
            raise KeyboardInterrupt()
    def remove_dir(target):
        rmdir(target)
        if point=='rmdir' and target.is_relative_to(legacy) and not interrupted:
            interrupted.append(1)
            raise KeyboardInterrupt()
    with monkeypatch.context() as patch:
        patch.setattr(migration,'atomic_bytes',copy)
        patch.setattr(migration,'save',save_journal)
        patch.setattr(Path,'unlink',remove)
        patch.setattr(Path,'rmdir',remove_dir)
        with pytest.raises(KeyboardInterrupt), batch_lock(c.output,c.work_root):
            local_processor(c,NativeASR())
    assert interrupted
    with batch_lock(c.output,c.work_root):
        p = local_processor(c,NativeASR())
        forbid_inference(p)
        assert p.process(path)[0]=='skipped'
        assert p.process(path)[1].name==selected
    assert_migrated(p,legacy,contents)


@pytest.mark.parametrize('backend',['local','api'])
def test_failed_selected_generation_resumes_after_migration(config,backend):
    c,path,legacy,selected,contents = legacy_fixture(config,'hidden',backend,failed=True)
    recognizer,translator = NativeASR(),FakeTranslator()
    with batch_lock(c.output,c.work_root):
        p = local_processor(c,recognizer,translator=translator)
        status,run = p.process(path)
        assert status=='completed' and run.name==selected and recognizer.calls==0
        if backend=='local':assert len(recognizer.native_calls)==1
        else:assert translator.calls[0]==1
        assert len(list((p.state/'inputs'/legacy.name).glob('*/record.json')))==2
        _,fresh=p.process(path,reprocess='translation')
        assert fresh!=run and run.exists()
        forbid_inference(p)
        assert p.process(path)[1]==fresh
    assert not legacy.exists()


@pytest.mark.parametrize('edited',['internal','public','unknown_inside','symlink'])
def test_modified_or_unrelated_legacy_state_is_preserved(config,edited):
    c,path,legacy,selected,contents = legacy_fixture(config,'hidden')
    if edited=='internal':
        (legacy/selected/'source.zh.txt').write_text('manual edit')
    elif edited=='public':
        (c.output/'Échange 中文 interview.en.txt').write_text('manual edit')
    elif edited=='unknown_inside':
        (legacy/'personal-notes.txt').write_text('unrelated')
    else:
        (legacy/'outside').symlink_to(c.source)
    before={f:f.read_bytes() for f in c.output.rglob('*') if f.is_file() and not f.is_symlink()}
    with pytest.raises(StateError),batch_lock(c.output,c.work_root):
        local_processor(c,NativeASR())
    assert all(f.read_bytes()==content for f,content in before.items())
    assert legacy.exists()


def test_ambiguous_legacy_and_work_refused(config):
    c,path,legacy,selected,contents=legacy_fixture(config,'hidden')
    state=state_directory(c.output,c.work_root)
    shutil.copytree(legacy,state/'inputs'/legacy.name)
    with pytest.raises(StateError,match='Ambiguous'),batch_lock(c.output,c.work_root):
        local_processor(c,NativeASR())
    assert legacy.exists()


def test_unrelated_output_entries_are_left_with_explanation(config,capsys):
    path=video(config)
    note=config.output/'notes.json';note.write_text('user notes')
    folder=config.output/'other results';folder.mkdir();(folder/'owner.json').write_text('unrelated owner')
    p=local_processor(replace(config,translation_backend='local'),NativeASR())
    p.progress=print
    migration.migrate_output(config.output,p.state,progress=print)
    p.process(path)
    assert note.read_text()=='user notes'
    assert (folder/'owner.json').read_text()=='unrelated owner'
    assert 'Unrelated non-TXT output entries preserved' in capsys.readouterr().out


def test_separate_outputs_and_concurrency_even_with_different_work(config):
    a=config.output;b=config.root/'Other Transcripts'
    with batch_lock(a,config.work_root):
        with pytest.raises(StateError,match='Another batch'):
            with batch_lock(a,config.root/'Different Work'):pass
        with batch_lock(b,config.work_root):pass
    assert state_directory(a,config.work_root)!=state_directory(b,config.work_root)
    assert all(p.name=='batch.lock' for p in config.work_root.rglob('*.lock'))
    assert not list(a.iterdir()) and not list(b.iterdir())


def test_legacy_lock_is_honored_and_new_lock_lives_in_work(config):
    import fcntl
    config.output.mkdir()
    old=config.output/'.transcribe-translate.lock';old.touch(mode=0o600)
    with old.open('rb') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(StateError,match='Another batch'):
            with batch_lock(config.output,config.work_root):pass
    assert old.exists()
    with batch_lock(config.output,config.work_root):pass
    assert not old.exists()
    assert (state_directory(config.output,config.work_root)/'batch.lock').exists()


def test_visible_work_and_relative_override_from_other_cwd(config,monkeypatch):
    monkeypatch.setenv('TT_ROOT',str(config.root))
    monkeypatch.chdir(config.root.parent)
    cfg=config.root/'config.local.toml'
    cfg.write_text('source_language="Mandarin"\n')
    c=load_config();assert c.work_root==config.root/'Work' and c.source_language=='zh'
    cfg.write_text('work_folder="Visible Processing"\nsource_language="Mandarin"\n')
    c=load_config();assert c.work_root==config.root/'Visible Processing'
    path=video(c)
    p=local_processor(c,NativeASR());p.process(path)
    assert p.state.is_relative_to(c.work_root)
    assert not any(part.startswith('.') for part in p.state.relative_to(c.work_root).parts)
    assert all(x.suffix=='.txt' for x in c.output.iterdir())


def test_cli_work_override_and_no_inference_republication(config,monkeypatch,capsys):
    monkeypatch.setenv('TT_ROOT',str(config.root))
    monkeypatch.chdir(config.root.parent)
    path=video(config)
    monkeypatch.setattr(cli.asr,'runtime_check',lambda c:c)
    created=[]
    def factory(c,**kwargs):
        p=processor(c)
        if created:forbid_inference(p)
        created.append(p);return p
    monkeypatch.setattr(cli,'Processor',factory)
    args=['--source-language','zh','--translation-backend','api','--work',str(config.root/'Visible Work')]
    monkeypatch.setenv('TT_TRANSLATION_BASE_URL',config.base_url)
    monkeypatch.setenv('TT_TRANSLATION_MODEL',config.model)
    assert cli.main(args)==0
    source,target=created[0].source_path,created[0].translated_path
    source.unlink();target.unlink()
    assert cli.main(args)==0
    assert source.exists() and target.exists() and created[-1].state.is_relative_to(config.root/'Visible Work')
    assert '1 skipped' in capsys.readouterr().out
    assert all(x.suffix=='.txt' for x in config.output.iterdir())


def test_owned_source_edit_refused_before_new_inference(config):
    path=video(config)
    p=processor(config);_,run=p.process(path)
    p.source_path.write_text('manual source correction')
    before=(run.parent/'index.json').read_bytes()
    forbid_inference(p)
    with pytest.raises(StateError,match='edited'):
        p.process(path,reprocess='all')
    assert (run.parent/'index.json').read_bytes()==before
    assert p.config.output.joinpath('Interview.zh.txt').read_text()=='manual source correction'


def test_source_publishes_on_native_failure(config):
    c=replace(config,translation_backend='local')
    path=video(c)
    p=local_processor(c,NativeASR(fail=RuntimeError('failure')))
    with pytest.raises(RuntimeError):p.process(path)
    assert (c.output/'Interview.zh.txt').exists()
    assert not (c.output/'Interview.en.txt').exists()


def test_publication_registry_edits_and_symlink_work_refused(config):
    path=video(config)
    p=processor(config);p.process(path)
    registry=p.state/'publications.json'
    registry.write_text(registry.read_text().replace('Interview','../escaped'))
    with pytest.raises(StateError,match='Invalid or edited checkpoint'):p.process(path)
    alternate=config.root/'Linked Work';alternate.symlink_to(config.work_root)
    with pytest.raises(StateError,match='symlink Work'):
        state_directory(config.root/'Alternate Outputs',alternate)


def test_legacy_api_without_backend_field_and_missing_artifacts(config):
    c,path,legacy,selected,contents=legacy_fixture(config,'legacy','api')
    for record_file in legacy.rglob('record.json'):
        record=load(record_file);record.pop('translation_backend');save(record_file,record)
    (legacy/selected/'source.zh.txt').unlink()
    (legacy/selected/'translated.en.txt').unlink()
    with batch_lock(c.output,c.work_root):
        p=local_processor(c,NativeASR(),translator=FakeTranslator())
        forbid_inference(p)
        assert p.process(path)[0]=='skipped'
        assert p.source_path.exists() and p.translated_path.exists()
    assert not legacy.exists()


def test_paired_stem_migration_preserves_overlapping_owned_public_names(config):
    import hashlib
    a=video(config,'Interview.mp4');b=video(config,'nested/Interview.mp4')
    hidden=config.output/'.state';hidden.mkdir()
    save(hidden/'owner.json',{'layout':'hidden-state-v1'})
    raw={'schema':1,'files':{}}
    values=[(a,'en','Interview.en.txt','English A'),
            (a,'fr','Interview--old.fr.txt','French A'),
            (b,'fr','Interview.fr.txt','French B')]
    for path,code,name,text in values:
        (config.output/name).write_text(text)
        raw['files'][digest({'input_path':str(path),'target_language':code})]={
            'input_path':str(path),'target_language':code,'filename':name,
            'sha256':hashlib.sha256(text.encode()).hexdigest()}
    save(hidden/'publications.json',raw)
    state=state_directory(config.output,config.work_root)
    with batch_lock(config.output,config.work_root):
        migration.migrate_output(config.output,state,progress=lambda s:None)
        data=load(state/'publications.json')
        owner_a=data['inputs'][digest(str(a))];owner_b=data['inputs'][digest(str(b))]
        assert owner_a['stem']=='Interview' and owner_b['stem'].startswith('Interview--')
        assert (config.output/'Interview.fr.txt').read_text()=='French A'
        assert (config.output/f"{owner_b['stem']}.fr.txt").read_text()=='French B'
        publication_a=Publication(config.output,a,'en',state=state)
        source=publication_a.publish('Source A',language='zh')
        assert source.stem.rsplit('.',1)[0]==publication_a.path.stem.rsplit('.',1)[0]
    assert not hidden.exists() and not (config.output/'Interview--old.fr.txt').exists()
    assert all(x.suffix=='.txt' for x in config.output.iterdir())


def test_changed_copy_after_interrupted_migration_is_refused(config,monkeypatch):
    c,path,legacy,selected,contents=legacy_fixture(config,'legacy')
    atomic=migration.atomic_bytes
    written=[]
    def stop(target,*args,**kwargs):
        atomic(target,*args,**kwargs);written.append(target);raise KeyboardInterrupt()
    with monkeypatch.context() as patch:
        patch.setattr(migration,'atomic_bytes',stop)
        with pytest.raises(KeyboardInterrupt),batch_lock(c.output,c.work_root):
            local_processor(c,NativeASR())
    written[0].write_text('modified working copy')
    with pytest.raises(StateError,match='Modified migration copy'),batch_lock(c.output,c.work_root):
        local_processor(c,NativeASR())
    assert all((legacy/name).read_bytes()==content for name,content in contents.items())


def test_unowned_hidden_folder_refused_without_moving_files(config):
    video(config)
    hidden=config.output/'.state';hidden.mkdir();note=hidden/'keep.txt';note.write_text('unrelated')
    with pytest.raises(StateError):processor(config)
    assert note.read_text()=='unrelated'


def test_work_must_share_output_filesystem(config,monkeypatch):
    from types import SimpleNamespace
    video(config)
    state=state_directory(config.output,config.work_root)
    real=Path.stat
    def fake_stat(path,*args,**kwargs):
        result=real(path,*args,**kwargs)
        return SimpleNamespace(st_dev=result.st_dev+1) if path==state else result
    monkeypatch.setattr(Path,'stat',fake_stat)
    with pytest.raises(StateError,match='share a filesystem'):
        migration.migrate_output(config.output,state,progress=lambda s:None)


def test_cross_process_output_lock(config):
    with batch_lock(config.output,config.work_root):
        command=[sys.executable,'-c',
          'from pathlib import Path; from transcribe_translate.storage import batch_lock; '
          'import sys; '\
          '\ntry:\n with batch_lock(Path(sys.argv[1]),Path(sys.argv[2])): pass'\
          '\nexcept Exception as e:\n print(type(e).__name__, str(e)); sys.exit(7)',
          str(config.output),str(config.root/'Other Work')]
        result=subprocess.run(command,text=True,capture_output=True)
        assert result.returncode==7 and 'Another batch' in result.stdout
    assert not list(config.output.iterdir())


def test_symlink_public_backup_refused_without_touching_external_files(config):
    c,path,legacy,selected,contents=legacy_fixture(config,'hidden')
    state=state_directory(c.output,c.work_root)
    outside=config.root/'Unrelated Backup';outside.mkdir()
    note=outside/'keep.txt';note.write_text('unrelated')
    (state/'migration-public').symlink_to(outside)
    with pytest.raises(StateError,match='Symlink migration path'),batch_lock(c.output,c.work_root):
        local_processor(c,NativeASR())
    assert list(outside.iterdir())==[note] and note.read_text()=='unrelated'
    assert all((legacy/name).read_bytes()==content for name,content in contents.items())


@pytest.mark.parametrize('change',['missing_index','wrong_selected_cache'])
def test_invalid_selected_source_metadata_refused_before_retirement(config,change):
    c,path,legacy,selected,contents=legacy_fixture(config,'legacy')
    index_file=legacy/'index.json'
    if change=='missing_index':index_file.unlink()
    else:
        index=load(index_file)
        signature,cache_id=next(iter(index['asr'].items()))
        index['asr']={digest('wrong signature'):cache_id};save(index_file,index)
    before={p:p.read_bytes() for p in legacy.rglob('*') if p.is_file()}
    with pytest.raises(StateError,match='selected-generation index|source-cache settings mismatch'),batch_lock(c.output,c.work_root):
        local_processor(c,NativeASR())
    assert all(p.read_bytes()==content for p,content in before.items())
    assert legacy.exists()
