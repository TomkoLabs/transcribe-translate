from __future__ import annotations

import os
import tempfile
import wave
from contextlib import contextmanager
from pathlib import Path

VIDEO_EXTENSIONS = frozenset(".mp4 .mkv .mov .avi .webm .m4v .mpg .mpeg .mts .m2ts .ts .flv .wmv .3gp .ogv .vob".split())


class MediaError(RuntimeError):
    pass


def discover(source: Path, output: Path, *, recursive: bool = False, work: Path | None = None) -> list[Path]:
    if not source.is_dir():
        raise MediaError(f"Source folder does not exist: {source}; create it or use --source")
    files = []
    for directory, dirs, names in os.walk(source, followlinks=False):
        parent = Path(directory)
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not (parent / d).is_symlink() and (parent / d).resolve() not in {output, work})
        for name in names:
            p = parent / name
            if name.startswith(".") or p.is_symlink() or not p.is_file() or p.resolve().is_relative_to(output):
                continue
            if p.suffix.lower() in VIDEO_EXTENSIONS:
                files.append(p.resolve())
            else:
                # Decoder-proven video containers with other extensions are accepted.
                import av
                try:
                    with av.open(str(p)) as container:
                        if container.streams.video:
                            files.append(p.resolve())
                except (av.error.FFmpegError, OSError, ValueError):
                    pass
        if not recursive:
            break
    return sorted(files, key=lambda p: (p.relative_to(source).as_posix().casefold(), p.relative_to(source).as_posix()))


@contextmanager
def decoded_audio(path: Path, scratch: Path):
    """Stream the first audio track to private mono 16 kHz PCM via FFmpeg/PyAV."""
    import av
    descriptor, name = tempfile.mkstemp(prefix="audio-", suffix=".wav", dir=scratch)
    os.close(descriptor)
    decoded = Path(name)
    try:
        try:
            with av.open(str(path)) as container, wave.open(str(decoded), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                if not container.streams.audio:
                    raise MediaError("No audio track; select a video containing spoken audio")
                stream = container.streams.audio[0]
                resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
                samples = 0
                for frame in container.decode(stream):
                    for converted in resampler.resample(frame):
                        wav.writeframesraw(converted.to_ndarray().tobytes())
                        samples += converted.samples
                for converted in resampler.resample(None):
                    wav.writeframesraw(converted.to_ndarray().tobytes())
                    samples += converted.samples
                if samples == 0:
                    raise MediaError("Audio track has no decodable samples")
        except (av.error.FFmpegError, OSError, ValueError) as exc:
            raise MediaError(f"Cannot decode media ({type(exc).__name__}); check that the file is readable and complete") from exc
        yield decoded
    finally:
        decoded.unlink(missing_ok=True)
