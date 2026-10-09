"""Installer/identity/relocation regressions. Models, pip and GPU are doubles here."""
from contextlib import ExitStack
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from conftest import processor
from test_output_layout import forbid_inference, video
from transcribe_translate import installation, relocation
from transcribe_translate.storage import file_hash, load, save, state_directory

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('project_installer', ROOT / 'scripts/install.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def root_fixture(tmp_path, monkeypatch):
    root = tmp_path / 'current checkout with spaces'
    root.mkdir()
    shutil.copytree(ROOT / 'src', root / 'src', ignore=shutil.ignore_patterns('__pycache__','*.egg-info'))
    for name in ['pyproject.toml','config.example.toml','requirements.lock','requirements-cuda.lock']:
        shutil.copy2(ROOT / name, root / name)
    monkeypatch.setattr(installer, 'ROOT', root)
    monkeypatch.setenv('TT_ROOT', str(root))
    monkeypatch.setattr(installer, 'active_runtime', lambda p: None)
    monkeypatch.setattr(installer, 'old_checkout', lambda p: root)
    monkeypatch.setattr(installer, 'runtime_health', lambda p: (True,'test runtime',sys.executable))
    return root


def test_rerun_refreshes_current_source_preserves_config_and_cuda_routing(tmp_path, monkeypatch):
    root = root_fixture(tmp_path, monkeypatch)
    config = root / 'config.local.toml'
    content = 'source_language="zh"\nasr_device="cuda"\nasr_compute_type="float16"\n[translation]\nbackend="api"\nbase_url="https://endpoint.example/v1"\nmodel="selected-quality-route"\napi_key="test credential"\n'
    config.write_text(content); config.chmod(0o600)
    runtime = root / '.venv'; runtime.mkdir(); (runtime/'bin').mkdir()
    for name in ['Video Source','Transcribed','Work']:
        folder = root / name; folder.mkdir(); (folder / 'keep').write_bytes(b'preserve')
    calls = []
    monkeypatch.setattr(installer, 'run', lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(installer, 'model_preparation', lambda python, mode: calls.append(['model',mode]))
    # Eligibility follows the actual unchanged driver policy, not a GPU execution.
    mode = installer.preflight.__globals__['choose_runtime'](
        find=lambda _: '/fake/nvidia-smi',
        run=lambda *a, **kw: SimpleNamespace(stdout='550.163.01, 8.6\n',returncode=0),log=lambda _:None)
    monkeypatch.setattr(installer,'preflight',lambda *a,**kw: 'cpu' if kw['cpu_only'] else mode)
    for args in [[],[],['--cpu']]:
        monkeypatch.setattr(sys,'argv',['install','--root',str(root),*args])
        assert installer.main() == 0
    installs = [c for c in calls if 'pip' in c and 'install' in c]
    assert sum(str(root/'requirements-cuda.lock') in c for c in installs) == 2
    refresh = [c for c in installs if '-e' in c]
    assert len(refresh) == 3
    assert all(c[0] == str(root/'.venv/bin/python') and c[-1] == str(root) and '--no-deps' in c for c in refresh)
    assert all('--force-reinstall' not in c for c in installs if '-e' not in c)
    assert calls.count(['model','cuda']) == 2 and ['model','cpu'] in calls
    assert config.read_text() == content and config.stat().st_mode & 0o777 == 0o600
    assert all((root/name/'keep').read_bytes() == b'preserve' for name in ['Video Source','Transcribed','Work'])
    assert not list(root.glob('.venv.backup-*'))


def test_failed_reinstall_restores_original_at_final_path(tmp_path, monkeypatch, capsys):
    root = root_fixture(tmp_path, monkeypatch)
    runtime = root / '.venv'; runtime.mkdir(); (runtime/'original').write_text('keep')
    (root/'config.local.toml').write_text('asr_device="auto"\n')
    monkeypatch.setattr(installer,'preflight',lambda *a,**kw:'cpu')
    def run(cmd,**kw):
        if 'venv' in cmd:
            assert cmd[-1] == str(runtime)  # Creation uses the final path.
            runtime.mkdir(); (runtime/'partial').write_text('inspect')
        else:
            raise subprocess.CalledProcessError(1, cmd)
    monkeypatch.setattr(installer,'run',run)
    monkeypatch.setattr(sys,'argv',['install','--root',str(root),'--reinstall'])
    assert installer.main() == 2
    assert (runtime/'original').read_text() == 'keep'
    assert len(list(root.glob('.venv.failed-*/partial'))) == 1
    assert 'Previous runtime restored' in capsys.readouterr().err
    assert (root/'config.local.toml').read_text() == 'asr_device="auto"\n'


def test_runtime_links_stale_metadata_and_entrypoints(tmp_path, monkeypatch):
    root = tmp_path/'new'; root.mkdir(); monkeypatch.setattr(installer,'ROOT',root)
    other = tmp_path/'other'; other.mkdir(); (other/'keep').write_text('external')
    runtime = root/'.venv'; runtime.symlink_to(other, target_is_directory=True)
    assert not installer.runtime_health(runtime)[0]
    retained = installer.backup(runtime,'backup')
    assert retained.is_symlink() and (other/'keep').read_text() == 'external'
    runtime.mkdir(); (runtime/'pyvenv.cfg').touch(); (runtime/'bin').mkdir()
    (runtime/'bin/python').symlink_to(sys.executable)
    site = runtime/'lib/python3.11/site-packages/transcribe_translate-0.2.1.dist-info'
    site.mkdir(parents=True)
    (site/'direct_url.json').write_text(json.dumps({'url':other.as_uri(),'dir_info':{'editable':True}}))
    assert installer.old_checkout(runtime) == other
    assert 'another checkout' in installer.runtime_health(runtime)[1]
    (site/'direct_url.json').write_text(json.dumps({'url':root.as_uri(),'dir_info':{'editable':True}}))
    (runtime/'bin/pip').write_text('#!'+str(other/'bin/python')+'\n')
    def run(cmd,**kw):
        return SimpleNamespace(stdout=json.dumps({'prefix':str(runtime),'base':sys.executable,
                   'pip':str(runtime/'lib/pip.py'),'version':[3,11]}))
    monkeypatch.setattr(installer,'run',run)
    assert 'shebang' in installer.runtime_health(runtime)[1]
    (runtime/'bin/shared').symlink_to(other/'keep')
    assert installer.unsafe_links(runtime)
    assert not installer.runtime_health(runtime)[0]
    with pytest.raises(ValueError,match='non-project'):
        installer.backup(other,'backup')


def test_foreign_python_environment_removed_but_credentials_retained(monkeypatch):
    monkeypatch.setenv('PYTHONHOME','/foreign'); monkeypatch.setenv('PYTHONPATH','/foreign')
    monkeypatch.setenv('PIP_TARGET','/foreign'); monkeypatch.setenv('PIP_PYTHON','/foreign/bin/python')
    monkeypatch.setenv('PIP_FIND_LINKS','/local-wheels')
    monkeypatch.setenv('VIRTUAL_ENV','/foreign'); monkeypatch.setenv('TT_TRANSLATION_API_KEY','test secret')
    cleaned = installer.clean_environment()
    assert not any(k.startswith('PYTHON') or k=='VIRTUAL_ENV' for k in cleaned)
    assert cleaned['TT_TRANSLATION_API_KEY']=='test secret'
    assert 'PIP_TARGET' not in cleaned and 'PIP_PYTHON' not in cleaned
    assert cleaned['PIP_FIND_LINKS']=='/local-wheels' and cleaned['PIP_CONFIG_FILE']==os.devnull


def test_setup_and_processing_installation_locks(tmp_path):
    with installation.installation_lock(tmp_path):
        with pytest.raises(RuntimeError,match='in use'):
            with installation.installation_lock(tmp_path,exclusive=True):
                pass
    with installation.installation_lock(tmp_path,exclusive=True):
        with pytest.raises(RuntimeError,match='in use'):
            with installation.installation_lock(tmp_path):
                pass


def test_old_output_lock_blocks_installation(tmp_path):
    import fcntl
    output = tmp_path/'Transcribed';output.mkdir()
    with ExitStack() as stack:
        fd = os.open(output,os.O_RDONLY|os.O_DIRECTORY);stack.callback(os.close,fd)
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='Processing'):
            installer.output_locks(stack,output)


def test_older_python_process_blocks_installation(tmp_path):
    runtime = tmp_path/'.venv'
    subprocess.run([sys.executable,'-m','venv','--without-pip',str(runtime)],check=True)
    child = subprocess.Popen([str(runtime/'bin/python'),'-I','-c','import time;time.sleep(10)'])
    try:
        with pytest.raises(RuntimeError,match='PID'):
            installer.active_runtime(runtime)
    finally:
        child.terminate();child.wait()


def hashes(folder):
    return {str(p.relative_to(folder)):file_hash(p) for p in folder.rglob('*') if p.is_file()}


@pytest.mark.parametrize('original_available',[True,False])
def test_copied_or_moved_completed_work_resumes_without_asr(config, tmp_path, original_available):
    p = processor(config); path = video(config,'French interview 日本語.mp4')
    _, first = p.process(path)
    p.translator.translate = lambda c,l: 'Newest selected translation '+str(c.id)
    _, selected = p.process(path,reprocess='translation')
    assert first != selected
    before = hashes(config.root)
    relocated = tmp_path.parent/(tmp_path.name+' relocated checkout with spaces'); shutil.copytree(config.root,relocated)
    if not original_available:
        shutil.move(config.root,tmp_path.parent/(tmp_path.name+' unavailable original'))
    c = replace(config,root=relocated,source=relocated/'Video Source',output=relocated/'Transcribed',work=relocated/'Work')
    relocation.relocate(c,config.root,progress=lambda _:None)
    # Old namespace is byte-for-byte intact in the copied checkout.
    old = relocation.namespace(c.work_root,config.output)
    assert hashes(old)=={k[len(str(p.state.relative_to(config.root)))+1:]:v for k,v in before.items() if k.startswith(str(p.state.relative_to(config.root))+'/')}
    new = processor(c);forbid_inference(new)
    moved = c.source/path.name
    new.translated_path = c.output/f'{path.stem}.en.txt';new.translated_path.unlink()
    status,run = new.process(moved)
    assert status=='skipped' and run.name==selected.name
    assert 'Newest selected' in new.translated_path.read_text()
    assert len(list(run.parent.glob('*/record.json')))==2
    if original_available:
        assert hashes(config.root)==before
    # Repeating the relocation must retain the already usable copy.
    relocation.relocate(c,config.root,progress=lambda _:None)


def test_relocation_protects_edited_files_and_absolute_paths(config, tmp_path):
    p=processor(config);path=video(config);p.process(path)
    new=tmp_path.parent/(tmp_path.name+' copy');shutil.copytree(config.root,new)
    c=replace(config,root=new,source=new/'Video Source',output=new/'Transcribed',work=new/'Work')
    (new/'Transcribed/Interview.en.txt').write_text('Manual edit')
    with pytest.raises(RuntimeError,match='edited or is unrelated'):
        relocation.relocate(c,config.root,progress=lambda _:None)
    assert not relocation.namespace(c.work_root,c.output).exists()
    (new/'config.local.toml').write_text('output_folder='+json.dumps(str(config.output))+'\n')
    before=hashes(new/'Work')
    relocation.relocate(replace(c,output=config.output),config.root,progress=lambda _:None)
    assert hashes(new/'Work')==before


def test_relocated_partial_translation_resumes_saved_chunks_and_source(config, tmp_path):
    from conftest import FakeTranslator
    original = config.root
    p=processor(config,translator=FakeTranslator(fail=1));path=video(config)
    with pytest.raises(TimeoutError):
        p.process(path)
    before=hashes(original)
    new=tmp_path.parent/(tmp_path.name+' partial copy');shutil.copytree(original,new)
    c=replace(config,root=new,source=new/'Video Source',output=new/'Transcribed',work=new/'Work')
    relocation.relocate(c,original,progress=lambda _:None)
    resumed=processor(c)
    resumed.transcriber.transcribe=lambda *a: pytest.fail('Saved source must resume')
    assert resumed.process(c.source/path.name)[0]=='completed'
    assert resumed.translator.calls[0]==1  # Chunk 0 was not requested again.
    assert hashes(original)==before


def test_source_version_mismatch_refused(tmp_path):
    shutil.copytree(ROOT/'src',tmp_path/'src')
    shutil.copy2(ROOT/'pyproject.toml',tmp_path/'pyproject.toml')
    expected=installation.source_version(ROOT)
    assert installation.source_version(tmp_path)==expected
    p=tmp_path/'pyproject.toml';p.write_text(p.read_text().replace('version = "'+expected+'"','version = "0.0.0"'))
    with pytest.raises(ValueError,match='versions disagree'):
        installation.source_version(tmp_path)


def test_relocation_atomic_namespace_failure_preserves_and_retries(config, tmp_path, monkeypatch):
    p=processor(config);path=video(config);p.process(path)
    new=tmp_path.parent/(tmp_path.name+' interrupted copy');shutil.copytree(config.root,new)
    c=replace(config,root=new,source=new/'Video Source',output=new/'Transcribed',work=new/'Work')
    target=relocation.namespace(c.work_root,c.output)
    before=hashes(new/'Work')
    real=relocation.os.rename
    def interrupted(source,destination):
        if Path(destination)==target:
            raise OSError('simulated namespace publication failure')
        return real(source,destination)
    monkeypatch.setattr(relocation.os,'rename',interrupted)
    with pytest.raises(OSError,match='publication failure'):
        relocation.relocate(c,config.root,progress=lambda _:None)
    assert not target.exists() and list(c.work_root.glob('.relocation-*'))
    old=relocation.namespace(c.work_root,config.output)
    assert all(file_hash(new/'Work'/name)==h for name,h in before.items())
    monkeypatch.setattr(relocation.os,'rename',real)
    relocation.relocate(c,config.root,progress=lambda _:None)
    resumed=processor(c);forbid_inference(resumed)
    assert resumed.process(c.source/path.name)[0]=='skipped'
    assert old.exists() and list(c.work_root.glob('.relocation-*'))


def test_explicit_absolute_source_paths_are_not_rebound(config, tmp_path):
    p=processor(config);path=video(config);p.process(path)
    new=tmp_path.parent/(tmp_path.name+' absolute source copy');shutil.copytree(config.root,new)
    (new/'config.local.toml').write_text('source_folder='+json.dumps(str(config.source))+'\n')
    c=replace(config,root=new,output=new/'Transcribed',work=new/'Work')
    relocation.relocate(c,config.root,progress=lambda _:None)
    resumed=processor(c);forbid_inference(resumed)
    assert resumed.process(path)[0]=='skipped'
    owners=[load(p)['input_path'] for p in resumed.state.glob('inputs/*/owner.json')]
    assert owners==[str(path)]


def test_relocation_does_not_overwrite_existing_unrelated_namespace(config, tmp_path):
    p=processor(config);path=video(config);p.process(path)
    new=tmp_path.parent/(tmp_path.name+' colliding state copy');shutil.copytree(config.root,new)
    c=replace(config,root=new,source=new/'Video Source',output=new/'Transcribed',work=new/'Work')
    target=state_directory(c.output,c.work_root)
    (target/'keep').write_text('Unrelated existing state')
    before=hashes(new/'Work')
    with pytest.raises(RuntimeError):
        relocation.relocate(c,config.root,progress=lambda _:None)
    assert hashes(new/'Work')==before
