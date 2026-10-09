from contextlib import contextmanager
from dataclasses import replace

import pytest

from transcribe_translate.config import Config


@pytest.fixture
def config(tmp_path):
    source = tmp_path / "Video Source"
    source.mkdir()
    return Config(root=tmp_path, source=source, output=tmp_path / "Transcribed",
                  source_language="zh", asr_device="cpu", asr_compute_type="int8",
                  translation_backend="api", base_url="https://endpoint.example/v1", model="bilingual-model", chunk_chars=50)


class FakeASR:
    def __init__(self, text="你好世界。明天不会下雨。" * 14):
        self.text = text
        self.calls = 0

    def transcribe(self, path):
        self.calls += 1
        return {"language": "zh", "segments": [{"start": 0.0, "end": 2.0, "text": self.text}],
                "actual_runtime": {"device": "cpu", "compute_type": "int8", "model": "large-v3"}}


class FakeTranslator:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def translate(self, chunk, source_language):
        self.calls.append(chunk.id)
        if chunk.id == self.fail:
            raise TimeoutError("test interruption")
        return f"Translated chunk {chunk.id}: {chunk.content}"

    def close(self):
        pass


@contextmanager
def fake_decoder(path, scratch):
    if path.read_bytes() == b"bad":
        raise ValueError("corrupt media")
    yield path


def processor(config, asr=None, translator=None):
    from transcribe_translate.pipeline import Processor
    return Processor(config, transcriber=asr or FakeASR(), translator=translator or FakeTranslator(),
                     decoder=fake_decoder, asr_settings={"model": config.asr_model, "source_language": config.source_language}, progress=lambda s: None)
