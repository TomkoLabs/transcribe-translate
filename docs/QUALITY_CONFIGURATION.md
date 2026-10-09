# Quality configuration review — October 6, 2026

The 0.2.4 review below is historical. The 0.2.5
[quality follow-up](QUALITY_FOLLOWUP.md) performs the proposed isolated ASR
evaluation and deployed API comparison. Application model/decoding settings remain
unchanged; the experiment is not an integrated backend.

The output-layout release keeps the existing multilingual large-v3 model, precision
selection and decoding settings. Inspection found no new settings defect that
justifies a correction. The earlier [failed fidelity evidence](QUALITY.md) is
preserved; successful processing, migration or CUDA kernels do not establish
recognition or translation fidelity.

## Effective configuration

The application calls the sequential `WhisperModel.transcribe` API in pinned
faster-whisper 1.2.1, not its batched API. The following values were checked against
installed code and the [tagged primary implementation](https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/faster_whisper/transcribe.py).

| Area | Effective behavior |
| --- | --- |
| Model / precision | Multilingual large-v3 at the pinned revision. Auto prefers supported CUDA/float16; CPU uses int8, observed as engine int8_float32. Explicit overrides remain available; fallback keeps the model. |
| Language | Generic source `auto` uses library detection with one initial window and threshold 0.5. Known Mandarin can explicitly use `zh`; native translation receives the recognized source language. Mixed-language recordings need review. |
| Decoding | Beam size 5 at temperature 0; default fallback temperatures 0.2, 0.4, 0.6, 0.8, 1.0. Nonzero-temperature sampling uses best-of 5; patience and length penalty are 1. Compression ratio 2.4 and average log probability -1.0 trigger fallback. |
| Previous windows | Conditioning enabled; library resets the prompt when fallback temperature exceeds 0.5. Context helps continuity but can carry mistakes/repetition. |
| Silence | VAD disabled. Decoder no-speech threshold 0.6 combines with low average log probability; quiet speech can still be missed. No additional silence removal or hallucination-silence skipping is enabled. |
| Audio | App streams the first audio track through PyAV into mono 16 kHz signed 16-bit PCM, preserving its timeline without cropping, denoising or volume gates. Decoded audio is temporary; source recording is retained. |
| Long recordings | Library feature windows default to 30 seconds. ASR timestamps determine seek advancement; text-only native translation advances across complete decoding windows, including the final partial window. The app does not independently chop or drop windows. |

Native English translation remains a separate audio pass with
`without_timestamps=True`, sharing one loaded model. This corrected earlier
repetition beyond the recording end; its stored ranges are decoding windows, not
translated-word alignment. Duration checks detect impossible segment ranges but
cannot prove completeness inside valid ranges. Plain public TXT omits generated
confidence markers; internal segments and annotated artifacts retain them.

Beam 5, adaptive fallback and previous-text context follow the established Whisper
long-form baseline described in the [Whisper paper, section A.1](https://cdn.openai.com/papers/whisper.pdf).
They are a supported starting point, not a universal optimum. Increasing beams,
disabling conditioning, forcing temperature zero or removing silence would require
paired evidence on the relevant audio; none is applied here. Pinning the library
also fixes otherwise implicit decoding defaults without changing existing result
signatures.

## Evidence and supported conclusion

Earlier licensed Mandarin clips showed source word/name errors; native English
omitted facts and invented wording. Explicit Mandarin and float32/direct-library
controls did not resolve the documented failures. The optional bilingual API
translated saved source more faithfully in the bounded check, but inherited source
errors and failed end-to-end audio fidelity. No backend has passed unreviewed
fidelity acceptance. The user-reported RTX 3080 / CUDA/float16 run establishes
reported previous-runtime execution, separately from those semantic tests. This
review performed no new speech inference, GPU access or model comparison.

No settings/model replacement is supported by the available evidence. A separate
ASR comparison is warranted, while native translation must also be evaluated
independently of source recognition.

## Evaluation proposed in 0.2.4 (performed separately in 0.2.5)

[Qwen3-ASR-1.7B's official model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
reports multilingual recognition, language identification, Chinese dialect support
and benchmark comparisons. Those aggregate claims justify a candidate, not
superiority on these recordings. Its [official runtime](https://github.com/QwenLM/Qwen3-ASR)
uses Transformers/PyTorch or vLLM; it is not an already validated replacement for
this pinned CTranslate2 application. GPU memory, dependency isolation and long-audio
behavior need acceptance. ASR capability alone does not establish English
translation fidelity. No alternative framework or weights were installed/downloaded.

A separate evaluation should freeze the decoded audio and references for the two
failed human-read clips and the longer compilation, then add permitted quiet,
conversational and mixed-language examples with speech around 30-second boundaries.
Compare pinned large-v3 with Qwen3-ASR-1.7B using explicit Chinese and auto detection.
Measure Mandarin character error rate plus manual names, numbers, negation,
omissions and invented speech; inspect window-boundary and silence errors. Record
actual device/precision, peak memory, reproducibility and installation behavior on
an authorized GPU host. Translate both saved source outputs through the same
unchanged bilingual API configuration to isolate ASR effects, and compare native
Whisper audio translation separately against human English references. Retain all
failures and adopt a replacement only after that paired assessment, rather than
benchmark rankings or throughput.
