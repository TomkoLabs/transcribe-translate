from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import wave

import av
import numpy as np
import pytest

from transcribe_translate import asr
from transcribe_translate.config import load_config
from transcribe_translate.languages import normalize_language
from transcribe_translate.media import MediaError, decoded_audio, discover


@pytest.mark.parametrize("value", ["Mandarin", "Chinese", "zh", "cmn", "zh-TW", "zh_hans", "普通話"])
def test_mandarin_normalization(value):
    assert normalize_language(value) == "zh"


def test_language_validation():
    assert normalize_language("English") == "en"
    assert normalize_language("auto", auto=True) == "auto"
    for bad in ["auto", "xyz", "zh-../../secret", "", None]:
        with pytest.raises(ValueError):
            normalize_language(bad)


def test_config_anchoring_and_secrets(config, monkeypatch):
    monkeypatch.setenv("TT_ROOT", str(config.root))
    monkeypatch.chdir(config.root.parent)
    p = config.root / "config.local.toml"
    p.write_text('source_language="Mandarin"\n[translation]\nbackend="api"\nbase_url="https://endpoint.example/v1"\nmodel="model"\napi_key="private-value"\n')
    c = load_config()
    c.validate()
    assert c.source == config.source
    assert c.source_language == "zh"
    assert "private-value" not in repr(c)
    monkeypatch.setenv("TT_TRANSLATION_MODEL", "override")
    assert load_config().model == "override"


@pytest.mark.parametrize("changes", [{"target_language": "en", "base_url": ""}, {"asr_model": "invalid"},
    {"asr_device": "mps"}, {"asr_compute_type": "float16"}, {"chunk_chars": 4000},
    {"base_url": "https://user:secret@example.com/v1"}, {"source": None}])
def test_config_invalid_settings(config, changes):
    if changes.get("source", 1) is None:
        changes = {"source": config.output}
    with pytest.raises(ValueError):
        replace(config, **changes).validate()


def test_invalid_config_keys_and_types(config, monkeypatch):
    monkeypatch.setenv("TT_ROOT", str(config.root))
    p = config.root / "config.local.toml"
    for text in ['unexpected=true', '[translation]\nbackend="api"\nattempts=true', 'source_folder=42', 'source_language=4', '[translation]\nbackend="api"\nbase_url=42']:
        p.write_text(text)
        with pytest.raises(ValueError):
            load_config()


def test_same_language_no_endpoint_required(config):
    replace(config, target_language="zh", base_url="", model="").validate()


def test_discovery_deterministic_unicode_spaces_and_scope(config):
    for name in ["中文 名字.MP4", "A.mp4", "a.mp4", "z.mkv", ".hidden.mp4", "readme.txt"]:
        (config.source / name).write_bytes(b"not real video")
    sub = config.source / "sub"
    sub.mkdir()
    (sub / "deep.mp4").write_bytes(b"x")
    (config.source / "outside.mp4").symlink_to(config.root / "A.mp4")
    assert [p.name for p in discover(config.source, config.output)] == ["A.mp4", "a.mp4", "z.mkv", "中文 名字.MP4"]
    assert len(discover(config.source, config.output, recursive=True)) == 5
    assert discover(config.source, config.output) == discover(config.source, config.output)


def test_unknown_extension_decoder_and_missing_audio(config):
    video = config.source / "视频.custom"
    with av.open(str(video), "w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=1)
        stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    assert discover(config.source, config.output) == [video]
    with pytest.raises(MediaError, match="No audio"):
        with decoded_audio(video, config.root):
            pass
    assert not list(config.root.glob(".audio-*.wav"))


def test_real_decoder_unicode_path_and_corrupt_media(config):
    audio = config.source / "你好 音频.wav"
    with wave.open(str(audio), "wb") as w:
        w.setparams((2, 2, 8000, 0, "NONE", "not compressed"))
        w.writeframes(np.zeros(8000 * 2, dtype=np.int16).tobytes())
    with decoded_audio(audio, config.root) as decoded:
        with wave.open(str(decoded)) as w:
            assert w.getnchannels() == 1
            assert w.getframerate() == 16000
            assert w.getnframes() > 0
    bad = config.source / "broken.mp4"
    bad.write_bytes(b"bad")
    with pytest.raises(MediaError, match="Cannot decode"):
        with decoded_audio(bad, config.root):
            pass


@pytest.mark.parametrize("count,device,compute", [(0, "cpu", "int8"), (1, "cuda", "float16")])
def test_auto_runtime(config, monkeypatch, count, device, compute):
    import ctranslate2
    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: count)
    monkeypatch.setattr(ctranslate2, "get_supported_compute_types", lambda d: {"int8", "float32"} if d == "cpu" else {"float16", "int8_float16"})
    actual = asr.runtime_check(replace(config, asr_device="auto", asr_compute_type="auto"))
    assert (actual.asr_device, actual.asr_compute_type, actual.asr_model) == (device, compute, "large-v3")


def test_explicit_cuda_and_bounded_fallback(config, monkeypatch):
    import ctranslate2
    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 0)
    with pytest.raises(ValueError, match="no visible"):
        asr.runtime_check(replace(config, asr_device="cuda"))
    t = asr.Transcriber(replace(config, asr_device="cuda", requested_device="auto", requested_compute_type="auto"), lambda s: None)
    assert not t._fallback(RuntimeError("corrupt model"))
    assert t._fallback(RuntimeError("libcudnn.so.9 cannot be loaded"))
    assert (t.config.asr_device, t.config.asr_model) == ("cpu", "large-v3")
    explicit = asr.Transcriber(replace(config, asr_device="cuda", requested_device="cuda"), lambda s: None)
    assert not explicit._fallback(RuntimeError("libcudnn.so.9 cannot be loaded"))


def test_auto_unsupported_gpu_and_unrelated_error(config, monkeypatch):
    import ctranslate2
    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 1)
    monkeypatch.setattr(ctranslate2, "get_supported_compute_types", lambda d: set() if d == "cuda" else {"int8"})
    selected = asr.runtime_check(replace(config, asr_device="auto", asr_compute_type="auto"))
    assert (selected.asr_device, selected.asr_model) == ("cpu", "large-v3")
    def broken():
        raise RuntimeError("unrelated runtime failure")
    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", broken)
    with pytest.raises(RuntimeError, match="unrelated"):
        asr.runtime_check(replace(config, asr_device="auto"))


def test_asr_quality_settings_and_no_speech(config):
    calls = []
    class Model:
        model = SimpleNamespace(device="cpu", compute_type="int8_float32")
        def transcribe(self, path, **kwargs):
            calls.append(kwargs)
            segment = SimpleNamespace(start=0.0, end=1.0, text="  這不是三百元。  ", avg_logprob=-1.2, no_speech_prob=0.1)
            return iter([segment]), SimpleNamespace(language="zh", language_probability=1.0, duration=1.0)
    t = asr.Transcriber(config, lambda s: None)
    t.model = Model()
    result = t.transcribe(Path("audio.wav"))
    assert result["segments"][0]["text"] == "這不是三百元。"
    assert result["segments"][0]["uncertain"]
    assert calls[0] == {"language": "zh", "task": "transcribe", "beam_size": 5, "vad_filter": False, "condition_on_previous_text": True}
    t.model.transcribe = lambda *a, **k: (iter([]), SimpleNamespace(language="zh", language_probability=1, duration=1))
    with pytest.raises(RuntimeError, match="No intelligible"):
        t.transcribe(Path("audio.wav"))
