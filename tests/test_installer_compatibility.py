"""Simulated GPU detection/installation and smoke orchestration, not GPU acceptance."""
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import ctranslate2
import pytest

from transcribe_translate import asr, cli
from transcribe_translate.config import load_config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('installer_runtime', ROOT / 'scripts/setup_runtime.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


@pytest.mark.parametrize('driver,capability,eligible', [
    ('550.163.01', '8.6', True), ('525.60.13', '8.6', True),
    ('525.60.12', '8.6', False), ('525.60.9', '8.6', False),
    ('525.59.99', '8.6', False), ('535.104.05', '8.6', True),
    ('550.163.01', '12.0', False), ('570.26', '12.0', True),
    ('570.25.99', '10.0', False), ('570.26.0', '10.0', True),
    ('580.173.02', '8.6', True), ('580.173.02', '13.0', False),
    ('550.163.01', '8.5', False),
    ('550.163.01', '3.5', False), ('N/A', '8.6', False),
    ('550.163.01', 'N/A', False), ('550.163.01 garbage', '8.6', False),
])
def test_documented_driver_and_architecture_eligibility(driver, capability, eligible):
    assert installer.cuda_candidate(driver, capability) == eligible


def test_detection_is_not_execution_and_cpu_skips_driver_probe():
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout='550.163.01, 8.6\n', returncode=0)
    messages = []
    assert installer.choose_runtime(find=lambda _: '/fake/nvidia-smi', run=run, log=messages.append) == 'cuda'
    assert '--query-gpu=driver_version,compute_cap' in calls[0]
    assert 'does not establish kernel execution' in messages[0]
    assert installer.choose_runtime(cpu_only=True, find=lambda _: pytest.fail('CPU must not probe GPU'), log=lambda _: None) == 'cpu'
    with pytest.raises(ValueError, match='not both'):
        installer.choose_runtime(cpu_only=True, require_cuda=True)


@pytest.mark.parametrize('output,code', [('', 0), ('550.163.01, 8.6', 1), ('550.163.01', 0), ('525.60.12, 8.6', 0)])
def test_missing_or_unsupported_detection(output, code):
    kwargs = dict(find=lambda _: '/fake/nvidia-smi',
                  run=lambda *a, **k: SimpleNamespace(stdout=output, returncode=code), log=lambda _: None)
    assert installer.choose_runtime(**kwargs) == 'cpu'
    with pytest.raises(ValueError, match='driver'):
        installer.choose_runtime(require_cuda=True, **kwargs)


def test_cuda_smoke_uses_cached_weights_and_bounded_encoder_decoders(config, monkeypatch):
    calls = []
    def cached(c, **kwargs):
        assert kwargs == {'local_files_only': True}
        return Path('/cached-model')
    monkeypatch.setattr(asr, 'prepare_model', cached)
    encoded = object()
    class Engine:
        device = 'cuda'
        compute_type = 'float16'
        n_mels = 128
        is_multilingual = True
        def __init__(self, path, **kwargs):
            assert path == '/cached-model' and kwargs['device'] == 'cuda'
        def encode(self, features):
            assert list(features.shape) == [1, 128, 3000]
            calls.append('encode')
            return encoded
        def generate(self, features, prompts, **kwargs):
            assert features is encoded and kwargs == {'beam_size': 1, 'max_length': 6}
            calls.append(prompts[0][2])
            return [object()]
    monkeypatch.setattr(ctranslate2.models, 'Whisper', Engine)
    actual = asr.cuda_smoke_check(replace(config, asr_device='cuda', asr_compute_type='float16'))
    assert calls == ['encode', '<|transcribe|>', '<|translate|>']
    assert actual['device'] == 'cuda' and actual['tasks'] == ('transcribe', 'translate')
    Engine.encode = lambda *a: (_ for _ in ()).throw(RuntimeError('call requires newer driver'))
    with pytest.raises(RuntimeError, match='CUDA smoke failed; no CPU fallback'):
        asr.cuda_smoke_check(replace(config, asr_device='cuda'))
    with pytest.raises(ValueError, match='explicit CUDA'):
        asr.cuda_smoke_check(config)


def test_smoke_cache_miss_never_downloads(config, monkeypatch):
    import huggingface_hub
    calls = []
    def missing(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError('not cached')
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', missing)
    with pytest.raises(RuntimeError, match='never downloads'):
        asr.prepare_model(replace(config,shared_hub=config.root/'missing shared cache'), local_files_only=True)
    assert len(calls) == 1 and calls[0]['local_files_only'] is True


def test_cli_smoke_ignores_api_and_never_discovers_media(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('TT_ROOT', str(tmp_path))
    (tmp_path / 'config.local.toml').write_text('[translation]\nbackend="api"\nbase_url=42\nmodel=false\n')
    def check(c):
        assert c.asr_device == 'cuda' and c.translation_backend == 'local' and not c.api_key
        return replace(c, asr_compute_type='float16')
    monkeypatch.setattr(cli.asr, 'runtime_check', check)
    monkeypatch.setattr(cli.asr, 'cuda_smoke_check', lambda c: {'device': 'cuda', 'compute_type': 'float16', 'model': 'large-v3', 'tasks': ('transcribe', 'translate')})
    monkeypatch.setattr(cli, 'discover', lambda *a, **k: pytest.fail('Smoke must not inspect recordings'))
    assert cli.main(['--cuda-smoke-check']) == 0
    assert 'Runtime execution only, not speech quality' in capsys.readouterr().out
    assert not (tmp_path / 'Transcribed').exists()
    assert cli.main(['--cuda-smoke-check', '--asr-device', 'cpu']) == 2
