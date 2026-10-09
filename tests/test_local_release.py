from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from conftest import FakeASR, FakeTranslator, fake_decoder
from transcribe_translate import asr, cli
from transcribe_translate.config import Config, load_config
from transcribe_translate.pipeline import Processor
from transcribe_translate.storage import StateError, load, save
from transcribe_translate.translation import Translator, TranslationError

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("setup_runtime", ROOT / "scripts/setup_runtime.py")
setup_runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup_runtime)


class NativeASR(FakeASR):
    def __init__(self, fail=None):
        super().__init__("這不是三百元。明天不會去北京。")
        self.native_calls = []
        self.fail = fail

    def translate(self, audio, source_language):
        self.native_calls.append((audio, source_language))
        if self.fail:
            raise self.fail
        return {"language": "en", "source_language": source_language,
                "segments": [{"start": 0.0, "end": 2.0, "text": "It is not 300 yuan. I will not go to Beijing tomorrow.", "uncertain": True}],
                "actual_runtime": {"model": "large-v3", "device": "cpu", "compute_type": "int8"}}


def local_processor(config, recognizer, **kwargs):
    return Processor(config, transcriber=recognizer, decoder=fake_decoder,
                     asr_settings={"model": config.asr_model, "source_language": config.source_language},
                     progress=lambda s: None, **kwargs)


def test_local_defaults_ignore_api_config_and_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("TT_ROOT", str(tmp_path))
    monkeypatch.setenv("TT_TRANSLATION_BASE_URL", "not a URL")
    monkeypatch.setenv("TT_TRANSLATION_MODEL", "unused")
    monkeypatch.setenv("TT_TRANSLATION_API_KEY", "invalid\nunused")
    c = load_config()
    c.validate()
    assert (c.translation_backend, c.source_language, c.target_language, c.asr_model) == ("local", "auto", "en", "large-v3")
    assert c.api_key == c.base_url == c.model == ""
    (tmp_path / "config.local.toml").write_text('[translation]\nbase_url=42\nmodel=false\napi_key_env=42\nattempts=false\n')
    c = load_config()
    c.validate()
    assert c.translation_backend == "local" and not c.api_key


def test_tracked_example_and_minimal_mandarin(tmp_path, monkeypatch):
    monkeypatch.setenv("TT_ROOT", str(tmp_path))
    c = load_config(ROOT / "config.example.toml")
    c.validate()
    assert c.translation_backend == "local"
    example = tmp_path / "mandarin.toml"
    example.write_text('source_language="zh"\ntarget_language="en"\n[translation]\nbackend="local"\n')
    c = load_config(example)
    c.validate()
    assert c.source_language == "zh"


def test_api_config_precedence_and_trailing_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("TT_ROOT", str(tmp_path))
    p = tmp_path / "config.local.toml"
    p.write_text('[translation]\nbackend="api"\nbase_url="https://endpoint.example/v1///"\nmodel="exact-alias"\n')
    c = load_config()
    c.validate()
    assert c.base_url == "https://endpoint.example/v1"
    monkeypatch.setenv("TT_TRANSLATION_MODEL", "environment-alias")
    assert load_config().model == "environment-alias"
    assert load_config(overrides={"translation_backend": "local"}).model == ""


@pytest.mark.parametrize("change", [{"target_language": "fr"}, {"target_language": "zh"},
    {"translation_backend": "unknown"}, {"asr_model": "turbo"}, {"asr_model": "small.en"}, {"asr_model": "distil-large-v3"}])
def test_unsupported_local_settings_fail_early(config, change):
    c = replace(config, translation_backend="local", **change) if "translation_backend" not in change else replace(config, **change)
    with pytest.raises(ValueError):
        c.validate()


def test_local_does_not_create_api_client_and_skips(config, monkeypatch):
    c = replace(config, translation_backend="local", model="", base_url="")
    config.output.mkdir()
    video = config.source / "人类 speech.mp4"
    video.write_bytes(b"video")
    from transcribe_translate import translation
    monkeypatch.setattr(translation, "Translator", lambda *a, **k: pytest.fail("local must not create an API client"))
    recognizer = NativeASR()
    p = local_processor(c, recognizer)
    status, run = p.process(video)
    assert status == "completed"
    assert recognizer.calls == 1 and recognizer.native_calls == [(video, "zh")]
    assert "這不是三百元" in (run / "source.zh.txt").read_text()
    assert "[uncertain]" in (run / "translated.en.txt").read_text()
    record = load(run / "record.json")
    assert record["translation_backend"] == "local" and record["native_translation"]["language"] == "en"
    assert not record["chunks"]
    assert p.process(video)[0] == "skipped"
    assert recognizer.calls == len(recognizer.native_calls) == 1


@pytest.mark.parametrize("error", [KeyboardInterrupt(), RuntimeError("native inference failed")])
def test_native_interruption_retains_source_and_resumes(config, error):
    c = replace(config, translation_backend="local")
    c.output.mkdir()
    path = c.source / "video.mp4"
    path.write_bytes(b"video")
    recognizer = NativeASR(fail=error)
    p = local_processor(c, recognizer)
    with pytest.raises(type(error)):
        p.process(path)
    record_file = next(c.work_root.rglob("record.json"))
    record = load(record_file)
    assert record["status"] in {"failed", "interrupted"}
    assert (record_file.parent / "source.zh.txt").exists()
    assert not (record_file.parent / "translated.en.txt").exists()
    recognizer.fail = None
    assert p.process(path)[0] == "completed"
    assert recognizer.calls == 1 and len(recognizer.native_calls) == 2


def test_native_publish_interruption_reuses_completed_pass(config, monkeypatch):
    c = replace(config, translation_backend="local")
    c.output.mkdir()
    path = c.source / "video.mp4"
    path.write_bytes(b"video")
    recognizer = NativeASR()
    p = local_processor(c, recognizer)
    from transcribe_translate import pipeline
    original = pipeline.write_artifact
    def interrupt_target(path, *args):
        if path.name == "translated.en.txt":
            raise KeyboardInterrupt()
        return original(path, *args)
    monkeypatch.setattr(pipeline, "write_artifact", interrupt_target)
    with pytest.raises(KeyboardInterrupt):
        p.process(path)
    monkeypatch.setattr(pipeline, "write_artifact", original)
    assert p.process(path)[0] == "completed"
    assert recognizer.calls == len(recognizer.native_calls) == 1


def test_backend_changes_share_asr_and_preserve_legacy_api(config):
    config.output.mkdir()
    path = config.source / "video.mp4"
    path.write_bytes(b"video")
    recognizer = NativeASR()
    api = local_processor(config, recognizer, translator=FakeTranslator())
    _, api_run = api.process(path)
    record = load(api_run / "record.json")
    record.pop("translation_backend")  # Actual legacy 0.1.0 shape.
    save(api_run / "record.json", record)
    assert api.process(path)[0] == "skipped"
    local = local_processor(replace(config, translation_backend="local"), recognizer)
    _, local_run = local.process(path)
    assert local_run != api_run and recognizer.calls == 1
    assert api.process(path)[1] == api_run
    old_api = (api_run / "translated.en.txt").read_bytes()
    _, reprocessed = local.process(path, reprocess="translation")
    assert reprocessed != local_run and recognizer.calls == 1
    assert (api_run / "translated.en.txt").read_bytes() == old_api
    local_record = load(reprocessed / "record.json")
    local_record["native_translation"]["language"] = "fr"
    save(reprocessed / "record.json", local_record)
    with pytest.raises(StateError, match="incompatible"):
        local.process(path)


def test_transcription_and_native_translation_share_model(config):
    tasks = []
    t = asr.Transcriber(replace(config, translation_backend="local"), lambda s: None)
    class Model:
        model = SimpleNamespace(device="cpu", compute_type="int8_float32")
        def transcribe(self, audio, **options):
            tasks.append(options)
            text = "你好" if options["task"] == "transcribe" else "Hello"
            return iter([SimpleNamespace(start=0., end=1., text=text, avg_logprob=0., no_speech_prob=0.)]), SimpleNamespace(language="zh", language_probability=1., duration=1.)
    model = Model()
    loads = []
    t._load = lambda: loads.append(1) or model
    source = t.transcribe(Path("audio.wav"))
    native = t.translate(Path("audio.wav"), "zh")
    assert source["language"] == "zh" and native["language"] == "en"
    assert source["actual_runtime"] == native["actual_runtime"] == {
        "device": "cpu", "compute_type": "int8_float32", "model": config.asr_model}
    assert t.config.asr_compute_type == "int8"  # Keep requested settings distinct.
    assert len(loads) == 1 and t.model is model
    assert [x["task"] for x in tasks] == ["transcribe", "translate"]
    assert all(x["beam_size"] == 5 and not x["vad_filter"] for x in tasks)
    assert tasks[1]["without_timestamps"] is True


def test_reject_impossible_segment_duration():
    with pytest.raises(RuntimeError, match="Invalid speech segments"):
        asr.validate_result({"language": "en", "duration_seconds": 2.0,
                             "segments": [{"start": 0.0, "end": 8.0, "text": "Invented after EOF"}]})


def test_api_base_prefix_trailing_slashes_and_check(config):
    requests = []
    def handler(request):
        requests.append(str(request.url))
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": config.model}]})
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": '{"id":0,"translation":"He will not go tomorrow."}',
            "reasoning_content": "private model reasoning"}}]})
    t = Translator(config, transport=httpx.MockTransport(handler), progress=lambda s: None)
    assert t.check() == "He will not go tomorrow."
    assert requests == ["https://endpoint.example/v1/models", "https://endpoint.example/v1/chat/completions"]
    t.close()
    for url in ("https://endpoint.example/v1/models", "https://endpoint.example/v1/chat/completions"):
        with pytest.raises(ValueError, match="base URL"):
            replace(config, base_url=url).validate()


def test_cli_rejects_non_english_before_runtime(config, monkeypatch):
    monkeypatch.setattr(cli, "load_config", lambda *a: replace(config, translation_backend="local", target_language="fr"))
    monkeypatch.setattr(cli.asr, "runtime_check", lambda *a: pytest.fail("early config rejection required"))
    assert cli.main([]) == 2


@pytest.mark.parametrize("cpu,driver,output,expected", [(True, True, "580.173.02, 8.6", "cpu"), (False, False, "", "cpu"),
    (False, True, "580.173.02, 8.6\n", "cuda"), (False, True, "524.1, 8.6\n", "cpu"), (False, True, "", "cpu")])
def test_setup_cuda_cpu_selection(cpu, driver, output, expected):
    mode = setup_runtime.choose_runtime(cpu_only=cpu, find=lambda name: "/nvidia-smi" if driver else None,
        run=lambda *a, **k: SimpleNamespace(stdout=output, returncode=0), log=lambda s: None)
    assert mode == expected


def test_setup_preserves_config_media_outputs_and_permissions(tmp_path):
    (tmp_path / "config.example.toml").write_bytes((ROOT / "config.example.toml").read_bytes())
    setup_runtime.create_defaults(tmp_path)
    assert (tmp_path / "Work").is_dir()
    work_note = tmp_path / "Work/keep.txt"
    work_note.write_text("existing processing data")
    cfg = tmp_path / "config.local.toml"
    assert cfg.stat().st_mode & 0o777 == 0o600
    cfg.write_text(cfg.read_text() + "\n# user setting\n")
    video = tmp_path / "Video Source/中文.mp4"
    transcript = tmp_path / "Transcribed/source.zh.txt"
    video.write_bytes(b"preserved recording")
    transcript.write_text("保留原文")
    before = [p.read_bytes() for p in (cfg, video, transcript)]
    setup_runtime.create_defaults(tmp_path, cpu_only=True)
    assert work_note.read_text() == "existing processing data"
    assert [p.read_bytes() for p in (cfg, video, transcript)] == before


def test_cpu_install_new_config_and_required_cuda_error(tmp_path):
    (tmp_path / "config.example.toml").write_bytes((ROOT / "config.example.toml").read_bytes())
    setup_runtime.create_defaults(tmp_path, cpu_only=True)
    assert 'asr_device = "cpu"' in (tmp_path / "config.local.toml").read_text()
    with pytest.raises(ValueError, match="driver"):
        setup_runtime.choose_runtime(require_cuda=True, find=lambda n: None, log=lambda s: None)
