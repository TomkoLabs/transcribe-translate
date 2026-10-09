"""Model-free layout/recovery regressions; inference uses explicit doubles."""
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from conftest import FakeASR, FakeTranslator, processor
from test_local_release import NativeASR, local_processor
from transcribe_translate import cli, publication, storage
from transcribe_translate.storage import StateError, digest, load, save


def video(config, name="Interview.mp4"):
    path = config.source / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"video")
    config.output.mkdir(exist_ok=True)
    return path


def forbid_inference(p):
    def fail(*args, **kwargs):
        pytest.fail("Completed checkpoints must not repeat decoding or inference")
    p.decoder = fail
    p.transcriber.transcribe = fail
    if hasattr(p.transcriber, "translate"):
        p.transcriber.translate = fail
    if p.translator is not None:
        p.translator.translate = fail


@pytest.mark.parametrize("name", ["Interview.mp4", "French interview with spaces.mkv", "Échange 日本語 中文.mp4",
                                  "a" * 245 + ".mp4", "é" * 122 + ".mp4", "𝒂" * 61 + ".mp4"])
def test_direct_naming_and_visible_work_state(config, name):
    path = video(config, name)
    p = processor(config)
    _, run = p.process(path)
    stem = Path(name).stem.encode()[:247].decode(errors="ignore")
    assert p.translated_path == config.output / f"{stem}.en.txt"
    assert p.translated_path.read_bytes() == (run / "translated.en.txt").read_bytes()
    assert p.source_path == config.output / f"{stem}.zh.txt"
    assert p.source_path.read_text() == p.transcriber.text + "\n"
    assert run.is_relative_to(config.work_root)
    assert (run / "source.zh.txt").exists()
    assert {x.name for x in config.output.iterdir()} == {p.source_path.name, p.translated_path.name}
    assert all(x.is_file() and x.suffix == ".txt" for x in config.output.iterdir())


def test_filename_and_language_collisions_are_stable(config):
    inputs = [video(config, n) for n in ["Interview.mp4", "Interview.mkv", "nested/Interview.mp4", "interview.mp4"]]
    p = processor(config)
    outputs = []
    for path in inputs:
        p.process(path)
        outputs.append(p.translated_path)
    assert outputs[0].name == "Interview.en.txt"
    assert len(set(x.name.casefold() for x in outputs)) == len(inputs)
    for path, out in zip(inputs[1:], outputs[1:]):
        key = digest(str(path.resolve()))
        assert out.name == f"{path.stem}--{key[:16]}.en.txt"
    forbid_inference(p)
    for path, out in zip(reversed(inputs), reversed(outputs)):
        assert p.process(path)[0] == "skipped"
        assert p.translated_path == out
    french = processor(replace(config, target_language="fr"))
    french.process(inputs[0])
    assert french.translated_path.name == "Interview.fr.txt"
    assert all(out.exists() for out in outputs)


def test_normalization_collision(config):
    p = processor(config)
    p.process(video(config, "Café.mp4"))
    first = p.translated_path
    p.process(video(config, "Cafe\u0301.mp4"))
    assert p.translated_path != first and "--" in p.translated_path.name


def test_local_direct_text_has_no_uncertainty_metadata(config):
    c = replace(config, translation_backend="local")
    p = local_processor(c, NativeASR())
    _, run = p.process(video(config))
    assert "[uncertain]" in (run / "translated.en.txt").read_text()
    assert p.translated_path.read_text() == "It is not 300 yuan. I will not go to Beijing tomorrow.\n"


def test_same_language_direct_text_preserves_spoken_brackets(config):
    c = replace(config, target_language="zh")
    recognizer = FakeASR("他说 [uncertain] 是一个标记。")
    p = processor(c, recognizer)
    p.process(video(config))
    assert p.translated_path.read_text() == recognizer.text + "\n"
    assert p.source_path == p.translated_path
    assert list(config.output.iterdir()) == [p.source_path]


@pytest.mark.parametrize("backend", ["api", "local"])
def test_rebuild_completed_direct_and_internal_txt_without_inference(config, backend):
    c = replace(config, translation_backend=backend)
    p = local_processor(c, NativeASR(), translator=FakeTranslator())
    path = video(config)
    _, run = p.process(path)
    original = p.translated_path.read_bytes()
    original_source = p.source_path.read_bytes()
    p.source_path.unlink()
    p.translated_path.unlink()
    (run / "translated.en.txt").unlink()
    forbid_inference(p)
    assert p.process(path)[0] == "skipped"
    assert p.translated_path.read_bytes() == original
    assert p.source_path.read_bytes() == original_source
    assert (run / "translated.en.txt").exists()


def test_partial_failure_resume_and_failed_reprocessing_preserve_direct(config):
    path = video(config)
    translator = FakeTranslator(fail=1)
    p = processor(config, translator=translator)
    with pytest.raises(TimeoutError):
        p.process(path)
    assert not (config.output / "Interview.en.txt").exists()
    assert (config.output / "Interview.zh.txt").read_text() == p.transcriber.text + "\n"
    assert not list(config.work_root.rglob("translated.en.txt"))
    translator.fail = None
    _, old_run = p.process(path)
    assert translator.calls[:3] == [0, 1, 1]
    assert p.transcriber.calls == 1
    original = p.translated_path.read_bytes()
    translator.fail = 1
    with pytest.raises(TimeoutError):
        p.process(path, reprocess="translation")
    assert (config.output / "Interview.en.txt").read_bytes() == original
    translator.fail = None
    translator.translate = lambda chunk, language: f"New complete chunk {chunk.id}"
    _, new_run = p.process(path)
    assert new_run != old_run and old_run.exists()
    assert "New complete chunk" in p.translated_path.read_text()
    assert (old_run / "translated.en.txt").read_bytes() == original
    assert p.transcriber.calls == 1
    forbid_inference(p)
    assert p.process(path)[1] == new_run


@pytest.mark.parametrize("point", ["before_replace", "after_replace", "after_registry"])
def test_atomic_publication_recovery_without_inference(config, monkeypatch, point):
    path = video(config)
    p = processor(config)
    p.process(path)
    direct = p.translated_path
    old_text = direct.read_text()
    p.translator.translate = lambda chunk, language: f"Updated chunk {chunk.id}"
    real_atomic = publication.atomic_text
    real_save = publication.save
    replacements = []
    def interrupted_atomic(target, text, **kwargs):
        assert target == direct and direct.read_text() == old_text
        assert load(next(r for r in config.work_root.rglob("record.json")
                         if "Updated chunk" in (r.parent / "translated.en.txt").read_text()))["status"] == "complete"
        replacements.append(text)
        if point == "before_replace":
            raise KeyboardInterrupt()
        real_atomic(target, text, **kwargs)
        if point == "after_replace":
            raise KeyboardInterrupt()
    def interrupted_save(target, data):
        entry = next(iter(data["inputs"].values()))["files"]["en"]
        if point == "after_registry" and "pending_sha256" not in entry:
            raise KeyboardInterrupt()
        real_save(target, data)
    monkeypatch.setattr(publication, "atomic_text", interrupted_atomic)
    monkeypatch.setattr(publication, "save", interrupted_save)
    with pytest.raises(KeyboardInterrupt):
        p.process(path, reprocess="translation")
    assert len(replacements) == 1
    assert direct.read_text() == (old_text if point == "before_replace" else replacements[0])
    monkeypatch.setattr(publication, "atomic_text", real_atomic)
    monkeypatch.setattr(publication, "save", real_save)
    forbid_inference(p)
    assert p.process(path)[0] == "skipped"
    assert direct.read_text() == replacements[0]
    entry = next(iter(load(p.state / "publications.json")["inputs"].values()))["files"]["en"]
    assert "pending_sha256" not in entry
    assert entry["sha256"] == hashlib.sha256(direct.read_bytes()).hexdigest()


def test_actual_atomic_replace_failure_keeps_old_complete_file(config, monkeypatch):
    path = video(config)
    p = processor(config)
    p.process(path)
    direct = p.translated_path
    old = direct.read_text()
    p.translator.translate = lambda chunk, language: f"Replacement chunk {chunk.id}"
    real_replace = storage.os.replace
    observed = []
    def replace(source, destination):
        if destination == direct:
            observed.append(Path(source).read_text())
            assert direct.read_text() == old
            assert Path(source).is_relative_to(p.state / "temporary")
            assert "Replacement chunk" in observed[-1]
            raise OSError("simulated rename failure")
        return real_replace(source, destination)
    monkeypatch.setattr(storage.os, "replace", replace)
    with pytest.raises(OSError, match="rename failure"):
        p.process(path, reprocess="translation")
    assert len(observed) == 1 and direct.read_text() == old
    assert not list(config.output.glob(".write-*.tmp"))
    monkeypatch.setattr(storage.os, "replace", real_replace)
    forbid_inference(p)
    assert p.process(path)[0] == "skipped"
    assert direct.read_text() == observed[0]


def test_edit_after_interrupted_publication_is_protected(config, monkeypatch):
    path = video(config)
    p = processor(config)
    real_atomic = publication.atomic_text
    def stop(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(publication, "atomic_text", stop)
    with pytest.raises(KeyboardInterrupt):
        p.process(path)
    direct = config.output / "Interview.en.txt"
    direct.write_text("Manual replacement\n")
    monkeypatch.setattr(publication, "atomic_text", real_atomic)
    forbid_inference(p)
    with pytest.raises(StateError, match="edited or is unrelated"):
        p.process(path)
    assert direct.read_text() == "Manual replacement\n"


@pytest.mark.parametrize("reprocess", [None, "translation", "all"])
def test_edited_direct_refused_before_inference_or_generation_change(config, reprocess):
    path = video(config)
    p = processor(config)
    _, run = p.process(path)
    direct = p.translated_path
    direct.write_text("Manual correction\n")
    index = (run.parent / "index.json").read_bytes()
    forbid_inference(p)
    with pytest.raises(StateError, match="edited or is unrelated"):
        p.process(path, reprocess=reprocess)
    assert direct.read_text() == "Manual correction\n"
    assert (run.parent / "index.json").read_bytes() == index


@pytest.mark.parametrize("kind", ["file", "symlink", "directory", "case_variant"])
def test_unrelated_direct_refused_without_adoption(config, kind):
    path = video(config)
    target = config.output / "Interview.en.txt"
    if kind == "file":
        target.write_text("Unrelated translation\n")
    elif kind == "symlink":
        target.symlink_to(path)
    elif kind == "directory":
        target.mkdir()
    else:
        (config.output / "interview.en.txt").write_text("Unrelated translation\n")
    p = processor(config)
    forbid_inference(p)
    with pytest.raises(StateError, match="unrelated|Unowned|Unrelated"):
        p.process(path)
    assert not list(config.work_root.rglob("record.json"))


def test_incomplete_completed_chunks_never_publish(config):
    path = video(config)
    p = processor(config)
    _, run = p.process(path)
    p.translated_path.unlink()
    record = load(run / "record.json")
    record["chunks"].pop()
    save(run / "record.json", record)
    forbid_inference(p)
    with pytest.raises(StateError, match="missing translation chunks"):
        p.process(path)
    assert not (config.output / "Interview.en.txt").exists()


def test_cli_prominently_prints_direct_path(config, monkeypatch, capsys):
    video(config)
    monkeypatch.setattr(cli, "load_config", lambda *a: config)
    monkeypatch.setattr(cli.asr, "runtime_check", lambda c: c)
    monkeypatch.setattr(cli, "Processor", lambda *a, **kw: processor(config))
    assert cli.main([]) == 0
    assert f"Completed: {config.output / 'Interview.en.txt'}" in capsys.readouterr().out
    assert cli.main([]) == 0
    assert f"Skipped: {config.output / 'Interview.en.txt'}" in capsys.readouterr().out


def test_french_source_and_english_target_use_canonical_codes(config):
    c=replace(config,source_language='fr')
    recognizer=FakeASR('Bonjour, nous ne partirons pas demain.')
    transcribe=recognizer.transcribe
    def french(path):
        result=transcribe(path);result['language']='fr';return result
    recognizer.transcribe=french
    translator=FakeTranslator();translator.translate=lambda chunk,language:'Hello, we will not leave tomorrow.'
    path=video(c,'French interview.mp4')
    p=processor(c,recognizer,translator);p.process(path)
    assert p.source_path.name=='French interview.fr.txt'
    assert p.translated_path.name=='French interview.en.txt'
    assert p.source_path.read_text()==recognizer.text+'\n'
    assert p.translated_path.read_text()=='Hello, we will not leave tomorrow.\n'
    assert sorted(x.name for x in c.output.iterdir())==[p.translated_path.name,p.source_path.name]
