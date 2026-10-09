from dataclasses import replace
from pathlib import Path

import pytest

from conftest import FakeASR, FakeTranslator, processor
from transcribe_translate import cli
from transcribe_translate.asr import render_source
from transcribe_translate.storage import StateError, batch_lock, job_directory, load, save


def input_file(config, name="中文 名字.mp4", content=b"video"):
    p = config.source / name
    p.write_bytes(content)
    return p


def test_unicode_and_basename_collisions(config):
    a = input_file(config, "same.mp4")
    other = config.source / "nested"
    other.mkdir()
    b = other / "same.mp4"
    b.write_bytes(b"video")
    config.output.mkdir()
    assert job_directory(config.output, a) != job_directory(config.output, b)
    p = processor(config)
    _, run = p.process(input_file(config))
    assert "你好世界" in (run / "source.zh.txt").read_text()
    assert load(run / "record.json")["status"] == "complete"


def test_resume_without_asr_or_repeated_chunks(config):
    config.output.mkdir()
    path = input_file(config)
    recognizer = FakeASR()
    failing = FakeTranslator(fail=1)
    p = processor(config, recognizer, failing)
    with pytest.raises(TimeoutError):
        p.process(path)
    runs = list(config.work_root.rglob("record.json"))
    record = load(runs[0])
    assert record["status"] == "failed" and len(record["chunks"]) == 1
    assert (runs[0].parent / "source.zh.txt").exists()
    assert not (runs[0].parent / "translated.en.txt").exists()
    resumed = FakeTranslator()
    p2 = processor(config, recognizer, resumed)
    status, run = p2.process(path)
    assert status == "completed" and recognizer.calls == 1
    assert resumed.calls and resumed.calls[0] == 1
    previous_calls = list(resumed.calls)
    assert p2.process(path)[0] == "skipped"
    assert resumed.calls == previous_calls and recognizer.calls == 1
    (run / "translated.en.txt").unlink()
    assert p2.process(path)[0] == "skipped"  # Regenerated from valid saved chunks.
    assert (run / "translated.en.txt").exists()


def test_sigint_checkpoint(config):
    config.output.mkdir()
    path = input_file(config)
    t = FakeTranslator()
    def interrupted(chunk, language):
        raise KeyboardInterrupt()
    t.translate = interrupted
    with pytest.raises(KeyboardInterrupt):
        processor(config, translator=t).process(path)
    assert load(next(config.work_root.rglob("record.json")))["status"] == "interrupted"
    assert processor(config).process(path)[0] == "completed"


def test_setting_invalidation_and_reprocessing(config):
    config.output.mkdir()
    path = input_file(config)
    recognizer = FakeASR()
    first = processor(config, recognizer)
    _, run = first.process(path)
    for change in [{"target_language": "fr"}, {"model": "different-model"}, {"chunk_chars": 70}]:
        p = processor(replace(config, **change), recognizer)
        assert p.process(path)[0] == "completed"
        assert recognizer.calls == 1
    second = processor(config, recognizer)
    _, new_run = second.process(path, reprocess="translation")
    assert new_run != run and run.exists() and recognizer.calls == 1
    assert second.process(path)[1] == new_run
    second.process(path, reprocess="all")
    assert recognizer.calls == 2
    assert second.process(path)[0] == "skipped"
    path.write_bytes(b"changed bytes")
    assert second.process(path)[0] == "completed" and recognizer.calls == 3
    processor(replace(config, source_language="auto"), recognizer).process(path)
    assert recognizer.calls == 4


def test_same_language_preserves_simplified_traditional_and_uncertainty(config):
    config.output.mkdir()
    same = replace(config, target_language="zh", base_url="", model="")
    text = "這不是脚本转换。不要删掉名字王小明。"
    recognizer = FakeASR(text)
    translator = FakeTranslator()
    _, run = processor(same, recognizer, translator).process(input_file(config))
    assert not translator.calls
    assert (run / "source.zh.txt").read_bytes() == (run / "translated.zh.txt").read_bytes()
    assert text in (run / "source.zh.txt").read_text()
    assert render_source([{"text": "可能是三十。", "uncertain": True}]) == "[uncertain] 可能是三十。\n"


def test_whitespace_spans_require_no_model_answer(config):
    config.output.mkdir()
    t = FakeTranslator()
    _, run = processor(config, FakeASR("你好" + " " * 130 + "世界"), t).process(input_file(config))
    record = load(run / "record.json")
    assert record["status"] == "complete"
    assert len(t.calls) < record["chunk_count"]


def test_refuse_operator_edits_and_unowned_collisions(config):
    config.output.mkdir()
    path = input_file(config)
    _, run = processor(config).process(path)
    target = run / "translated.en.txt"
    target.write_text("operator edits")
    with pytest.raises(StateError, match="edited"):
        processor(config).process(path)
    assert target.read_text() == "operator edits"
    marker = run.parent / "owner.json"
    marker.write_text("unowned")
    with pytest.raises(StateError):
        processor(config).process(path)


def test_reject_missing_duplicated_checkpoint_chunks(config):
    config.output.mkdir()
    path = input_file(config)
    _, run = processor(config).process(path)
    record = load(run / "record.json")
    record["chunks"][1]["id"] = 0
    save(run / "record.json", record)
    with pytest.raises(StateError, match="chunks"):
        processor(config).process(path)


def test_lock_excludes_concurrent_writer(config):
    with batch_lock(config.output):
        with pytest.raises(StateError, match="Another batch"):
            with batch_lock(config.output):
                pass


def test_cli_batch_continues_failures_and_empty(config, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda *a: config)
    monkeypatch.setattr(cli.asr, "runtime_check", lambda c: c)
    made = []
    def factory(c, **kwargs):
        p = processor(c)
        made.append(p)
        return p
    monkeypatch.setattr(cli, "Processor", factory)
    assert cli.main([]) == 0
    assert "No supported videos" in capsys.readouterr().out
    input_file(config, "A broken.mp4", b"bad")
    input_file(config, "B successful.mp4")
    input_file(config, "C successful.mp4")
    assert cli.main([]) == 1
    out = capsys.readouterr()
    assert "2 completed" in out.out and "1 failed" in out.out
    assert "corrupt media" in out.err
    assert cli.main([]) == 1
    assert "2 skipped" in capsys.readouterr().out


def test_cli_invalid_settings_before_processor(config, monkeypatch):
    monkeypatch.setattr(cli, "load_config", lambda *a: replace(config, model=""))
    monkeypatch.setattr(cli, "Processor", lambda *a, **k: pytest.fail("must not start expensive work"))
    assert cli.main([]) == 2
