"""Model-free API/configuration regressions; not translation-quality tests."""
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from transcribe_translate import cli
from transcribe_translate.config import load_config
from transcribe_translate.translation import TranslationError, Translator

ROOT = Path(__file__).resolve().parents[1]


def test_discovery_authenticates_without_translation_or_selected_model(config):
    calls = []

    def handler(request):
        calls.append(request)
        assert request.method == "GET"
        assert request.headers["Authorization"] == "Bearer test-credential"
        return httpx.Response(200, json={"data": [{"id": "model-a"}, {"id": "model-a"}, {"id": "model-b"}]})

    translator = Translator(replace(config, model="", api_key="test-credential"), transport=httpx.MockTransport(handler))
    try:
        assert translator.models() == ["model-a", "model-b"]
        assert len(calls) == 1 and str(calls[0].url).endswith("/v1/models")
    finally:
        translator.close()


@pytest.mark.parametrize("body", [{}, {"data": {}}, {"data": [None]}, {"data": [{"id": 1}]},
                                  {"data": [{"id": ""}]}, {"data": [{"id": "line\nbreak"}]}])
def test_malformed_discovery_fails_clearly(config, body):
    translator = Translator(config, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)))
    try:
        with pytest.raises(TranslationError, match="API /models discovery"):
            translator.models()
    finally:
        translator.close()


@pytest.mark.parametrize("status", [401, 403, 404, 503])
def test_discovery_http_failure_no_secret_body_or_fallback(config, status):
    translator = Translator(config, transport=httpx.MockTransport(lambda r: httpx.Response(status, text="secret response")))
    try:
        with pytest.raises(TranslationError) as error:
            translator.models()
        assert "secret response" not in str(error.value)
    finally:
        translator.close()


def test_cli_models_requires_no_model_and_does_not_start_asr(config, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda *a: replace(config, model=""))
    monkeypatch.setattr(cli.asr, "runtime_check", lambda *a: pytest.fail("discovery must not start ASR"))
    closed = []

    class Discovery:
        def __init__(self, c):
            assert not c.model

        def models(self):
            return ["exact-alias"]

        def close(self):
            closed.append(True)

    monkeypatch.setattr(cli, "Translator", Discovery)
    assert cli.main(["--api-models"]) == 0
    assert "exact-alias" in capsys.readouterr().out and closed == [True]
    assert not config.output.exists() and not config.work_root.exists()


def test_cli_discovery_preserves_api_error(config, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda *a: config)

    class FailedDiscovery:
        def __init__(self, c):
            pass

        def models(self):
            raise TranslationError("API discovery authentication failed")

        def close(self):
            pass

    monkeypatch.setattr(cli, "Translator", FailedDiscovery)
    monkeypatch.setattr(cli.asr, "runtime_check", lambda *a: pytest.fail("must not fall back"))
    assert cli.main(["--api-models"]) == 2
    assert "authentication failed" in capsys.readouterr().err


def test_separate_api_example_and_setup_preserve_user_config(tmp_path, monkeypatch):
    import importlib.util

    monkeypatch.setenv("TT_ROOT", str(tmp_path))
    for name in ("TT_TRANSLATION_BASE_URL", "TT_TRANSLATION_MODEL", "TT_TRANSLATION_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    local = tmp_path / "config.local.toml"
    original = b'source_language="zh"\n[translation]\nbackend="local"\nmodel="retained-choice"\n'
    local.write_bytes(original)
    (tmp_path / "config.example.toml").write_bytes((ROOT / "config.example.toml").read_bytes())
    api = tmp_path / "config.api.local.toml"
    api.write_bytes((ROOT / "config.api.example.toml").read_bytes())
    original_api = api.read_bytes()
    loaded = load_config(api)
    loaded.validate()
    assert loaded.translation_backend == "api" and loaded.source_language == "auto"
    assert loaded.work_root == tmp_path / "Work" and loaded.asr_model == "large-v3"
    spec = importlib.util.spec_from_file_location("setup_runtime", ROOT / "scripts/setup_runtime.py")
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    for _ in range(2):
        setup.create_defaults(tmp_path, cpu_only=True)
    assert local.read_bytes() == original and api.read_bytes() == original_api
    assert load_config().translation_backend == "local" and load_config().source_language == "zh"
