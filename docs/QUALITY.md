# Translation quality closeout — October 6, 2026

The results below are preserved historical evidence. The later
[bounded follow-up](QUALITY_FOLLOWUP.md) reproduces these failures, compares four
deployed translation families and evaluates Qwen3-ASR in isolation. It does not
reverse end-to-end fidelity failure or replace the application ASR.

The fully local default remains available, but **native English translation failed
fidelity acceptance** on the two original Mandarin clips and a longer compilation.
Source ASR also made material errors. Functional completion, low character error
rate and plausible English do not establish preservation of meaning.

| Check | Status | Evidence |
| --- | --- | --- |
| Source ASR fidelity | Failed | Mammals/island misrecognition; queen role became a different word |
| Native fidelity | Failed | All three inputs contain material omissions/inventions; direct-library and float32 controls reproduce short-clip errors |
| API fidelity to saved source | Passed, bounded qualitative check | Completed 1/1/3 chunks retain source meaning, numbers and uncertain wording; first malformed JSON response was rejected |
| API end-to-end fidelity | Failed | Incorrect/unclear ASR names and lost roles remain; original list relationship is not recovered |
| CPU execution | Passed | Fresh source/native runs, source reuse by API and fresh corrected runtime metadata |
| Regression checks | Passed | 87 tests, including both passes reporting resolved engine type |
| RTX 3080 execution | User-reported runtime success; corrected setup unverified | Later supplied run completed both CUDA/float16 passes; no GPU execution/access in this development environment. See [GPU acceptance](GPU_ACCEPTANCE.md) |
| GitLab CI | Unverified | No matching runner execution; local checks are separate |

**Release decision:** neither backend meets all tested end-to-end quality criteria
for unreviewed deployment. The API path is better at translating the available
source in these samples, but that is insufficient evidence to recommend it as a
quality-accepted deployment. Keep the local default and review source/English
against recordings. More varied speech and independent human review are needed.

## Scope and criteria

We used the same licensed human audio for local source/native processing and API
translation of that exact saved source. All fresh source/native inference used a
new isolated result directory, explicit `zh`, pinned multilingual large-v3,
CPU `int8`, beam size 5, previous-text conditioning and no VAD filtering. The engine
resolved `int8` to `int8_float32`. Native translation uses supported text-only
`without_timestamps=True`; source transcription retains timestamp decoding.

Two original FLEURS `cmn_hans_cn` dev recordings are 18.38 and 7.44 seconds. The new
66.72-second fixture concatenates three distinct human reads about Carter's foreign
policy, the French royal family and an Australian survey. No samples were trimmed,
filtered, repeated or synthesized. It is an edited compilation with topic/speaker
transitions, not a natural continuous conversation. It exercises three native
30-second decoding windows. API acceptance deliberately uses `chunk_chars = 120`
and `context_chars = 160` to exercise three chunks; the public default remains 1200.

[Fixture metadata](quality-fixtures.json) records the pinned dataset revision,
filenames/IDs, sample counts, hashes, compilation order and settings. Read raw
references from that revision's `data/cmn_hans_cn/dev.tsv` by exact filename;
audio comes from `data/cmn_hans_cn/audio/dev.tar.gz`. See
[dataset attribution](../THIRD_PARTY.md). Audio, full references, logs and generated
result folders remain ignored. Original failed evidence was preserved separately
from the new closeout runs.

A semantic pass requires preservation of factual propositions, negation, subjects,
relations, names and numbers, with no material omission or invention. Valid English
wording and reasonable transliteration variants are allowed; exact string matching
is not the acceptance criterion. An unclear ASR word may remain conservatively
marked uncertain in API output without being guessed. API fidelity to source and
end-to-end fidelity to audio/reference are separate judgments.

## Diagnosis

The original video containers decoded to **exactly the same PCM samples** as the
licensed original WAVs: 294,080 and 119,040 samples at 16 kHz, with zero maximum
sample difference. Dataset filenames/IDs and sample counts match their reference
rows. Chinese recognition closely corresponds to those references, including the
homophonically confused mammals/island wording. This establishes correct fixture
association and supports the text comparison. No independent listening review or
blinded acoustic reference audit was performed here.

The cached 3,087,284,237-byte model matches the pinned upstream LFS SHA-256:
`69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1`.
The model is multilingual and its feature extractor uses 128 mel bands. The app
selects `task="transcribe"` and a separate `task="translate"` with source `zh`.
Each call gets fresh library window/prompt state; both share loaded weights. The
native path consumes original audio, not the saved Chinese text. No custom speech
window slicing, initial prompt, hotwords or reference hints were used.

A direct faster-whisper call on the original 7.44-second WAV, bypassing application
decoding/checkpoints/assembly, reproduced the exact incorrect flying/predator
English. Its actual fallback temperature was **0.0**. The clip occupies one decoding
window, so previous-window conditioning cannot explain this failure. A preselected
float32 control on the same pinned model reproduced that translation exactly and
retained the first clip's wrong Middle East/list interpretation. Both controls also
used temperature zero. Float32 source transcription corrected the mammals word,
but still misrecognized the island name. Precision can affect ASR; it did not
resolve these native translation errors.

The released text-only mode avoids the earlier timestamp-token repetition outside
actual recording duration. Fresh native windows ended at 18.38, 7.44, and
30/60/66.72 seconds respectively, and assembled in order. Plausible ranges do not
catch meaning errors. The longer native output demonstrates a continuity failure:
its survey sentence inherited Versailles from the preceding French-history window.

One additional, preselected control changed only long-sample
`condition_on_previous_text` to `False`, because of that specific carryover.
Australia replaced Versailles in the final sentence and the king's sister relation
improved, confirming a contribution from prior-window context. However, native
English still reversed opposition to becoming a republic, omitted 2010/1,400,
changed 1789 to 1989 and XVI to VI, and misrendered the hostage crisis. This control
also failed fidelity. It is separate diagnostic evidence, not a new default or a
claim that disabling conditioning fixes Mandarin translation.

The evidence localizes the residual native semantic errors to the model/engine
output, upstream of this application's assembly. No evidence-supported application
decoding correction was found. It does **not** distinguish Whisper checkpoint
behavior from conversion/CTranslate2 internals, establish behavior on another
backend, or certify GPU output. We did not search arbitrary decoding settings until
an answer matched a reference.

One separate reporting defect was corrected in **0.2.1**: result metadata used the
requested `int8` alias rather than the loaded engine's `int8_float32`. New records
and `Actual` messages read the engine's supported
[device/type properties](https://opennmt.net/CTranslate2/python/ctranslate2.models.Whisper.html).
A fresh two-pass run confirmed
those fields while reproducing identical Chinese and incorrect English text.
Requested settings/checkpoint signatures remain unchanged, and old results retain
their historical metadata. This is a reporting correction, not a fidelity fix.

## Representative observations

| Fixture | Source ASR | Native English |
| --- | --- | --- |
| Iraq warning, 18.38 s | Reference characters retained, but punctuation merged the sectarian-war/violence list | Negation retained; sectarian war became “Middle East war” and the list relationship changed |
| Island ecology, 7.44 s | `哺乳动物` became `普鲁动物`; `科隆` became `克隆` | Mammals/giant tortoises/herbivores were lost; output invented “can't fly long distances” and “main predators” |
| History/survey, 66.72 s | Dates/ages/survey numbers retained; `王后` became `往后`, with name spelling differences | 1789 became 1989, Louis XVI became VI, the king's sister became the queen's sister; survey year, 1,400 respondents, opposition and Australia were omitted/replaced |

The long reference describes opposition to Australia becoming a republic increasing
8% since 2008, from a pre-2010 survey of 1,400 people. Native English instead says
people in Versailles became a republic. This is a material subject/negation/
continuity failure, not an acceptable paraphrase.

Automated normalized source CER was 0/44, 3/35 and 5/228 edits for the three inputs,
excluding punctuation/whitespace. CER penalizes valid spelling/transliteration
variants and misses logical relationships; these numbers are diagnostics, not
semantic passes. In particular, the low long-sample CER conceals substantial native
English errors. Confidence heuristics did not reliably mark those errors.

## API comparison

The comparison uses a currently exposed bilingual Qwen 27B alias, verified through
live `/models`, with existing authorized gateway access. Private URL/alias/config
are in the ignored handoff. Requests send only `model` and `messages`, preserving
endpoint defaults, including reasoning requirements. Separate reasoning is excluded
from transcript/evidence text. The API processor was given a recognizer that fails
if called: all compared API source TXT therefore comes from the exact completed
local ASR cache, rather than a second recognition attempt.

The first API attempt completed the Iraq clip but rejected a malformed JSON answer
for the ecology clip: an unescaped control character. No incomplete English TXT was
published. One bounded resume kept that failure evidence, reused the first completed
result, and continued the remaining audio-based comparison. The parser was not
relaxed and endpoint defaults were not changed.

The ecology API answer preserves giant tortoises and herbivores, but renders the
misrecognized source as “Pulu [uncertain] animals” and “Clone Islands.” This is more
faithful to the saved source than the native flying/predator invention, yet it does
not recover the mammals/island meaning in the reference. The Iraq API answer follows
the source's merged war/violence relationship; it does not preserve the reference's
separate list. Authored-text connectivity checks from the earlier release are not
counted as audio end-to-end acceptance.

The longer API run completed three chunks with contiguous source ranges
`0..93`, `93..190` and `190..251`. Identical saved source text was verified for local
and API generations. Chunk hashes/IDs cover every source character once; completed
answers contain no repeated neighboring topic or omitted chunk. Qualitative review
found the survey's 2010/2008 dates, 1,400 respondents, opposition and 8% increase
retained, without native translation's Versailles carryover. The 1789 date, XVI,
children's ages and king's sister relation were also retained.

However, source `往后` became “followed by,” so the reference's queen role is still
missing. `卡尔特` and `戴维莹` remain Chinese names marked uncertain in API English,
rather than a confidently identified Carter/Camp David. These conservative choices
are accepted for fidelity to an unclear saved source, but do not pass the reference's
end-to-end name/role criteria. Completed source/API fidelity judgments permit such
uncertainty; they do not certify universal alias accuracy or JSON reliability. Five
chunks completed across the three files; one earlier malformed response remains a
documented structural failure, followed by a successful resume.

## Limits and reproduction

These are bounded read-speech samples, with an artificial longer compilation. There
was no conversational/noisy/quiet-speech benchmark, independent human review panel,
statistical comparison or GPU test. Qualitative judgments above compare propositions
with the known references; another model's opinion is not treated as a validator.
Source ambiguity, pronunciation and valid name variants are distinguished from
clear factual changes such as 1789 → 1989 and herbivores → predators.

Use fresh directories with the manifest's permitted fixtures and `zh`/`en` settings
for reproduction. Run local first, then API with the same input/output and ASR
settings so it reuses source; use acceptance-only `chunk_chars = 120` for the longer
boundary check. Preserve baseline and subsequent runs. See
[manual GPU acceptance](GPU_ACCEPTANCE.md) for separate, unexecuted RTX 3080 commands
and [runtime validation](VALIDATION.md) for regression/installation history.
