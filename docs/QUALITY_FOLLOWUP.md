# Bounded quality follow-up — October 6, 2026

This follow-up preserves the [earlier failures](QUALITY.md) and evaluates source
recognition separately from text translation. It is a small qualitative comparison,
not broad fidelity acceptance. Source ASR and native translation in the application
remain pinned multilingual large-v3 with their supported quality settings. No server,
gateway route, production default or application dependency was changed.

## Reproduced baseline

A fresh real application run used the three permitted human-audio fixtures from
[the manifest](quality-fixtures.json), explicit `zh`, CPU/int8 (actual
`int8_float32`), beam 5, previous-text conditioning and no VAD. All three source
and native-English texts exactly reproduced the saved failed results. This includes
the ecology omissions/invented flying/predators, wrong historical date/roles, and
survey omission/carryover. The new output contains only three source and three
English TXT files; recovery and temporary files use a separate Work directory.
Original evidence and the user's local configuration were preserved.

Judgments compare propositions with the exact saved source and the published
human-read references separately. Decoded-audio/reference association was established
in the earlier investigation. This follow-up does not add an independent listening
review, a blinded human panel or a conversational/noisy/quiet-speech benchmark.

## Isolated Qwen3-ASR experiment

The [official runtime](https://github.com/QwenLM/Qwen3-ASR) supports Transformers
and vLLM; it is a separate stack from faster-whisper/CTranslate2. We installed
`qwen-asr==0.0.6`, `transformers==4.57.6`, `accelerate==1.12.0` and
`torch==2.9.1+cpu` in an ignored isolated Python 3.11 environment. The application
venv and installer pins were untouched. No vLLM, FlashAttention, forced aligner or
CUDA runtime was added to the application.

The [official 1.7B checkpoint](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) was fixed
at revision `7278e1e70fe206f11671096ffdd38061171dd6e5`; weight hashes and the full
experimental dependency lock are retained privately. Inference used CPU/float32,
SDPA, four threads, batch one, empty context, greedy generation and a 4096-token
ceiling. No reference hints, source corrections or audio filters were supplied.
All four calls ended at EOS with 27, 28, 181 and 30 new tokens; none reached that
ceiling. The runtime processed the complete 66.72-second compilation. The ecology
clip was also run with automatic language detection, which returned Chinese and
the same text as explicit Chinese.

| Fixture | Whisper int8 / Whisper float32 / Qwen float32 normalized edits | Observed Qwen result |
| --- | --- | --- |
| Warning, 18.38 s | 0 / 0 / 0 of 44 characters | Same wording; punctuation still merges the warning's list |
| Ecology, 7.44 s | 3 / 1 / 2 of 35 | Corrects mammals; still misrecognizes the island name and repeats `主` |
| History/survey, 66.72 s | 5 / 4 / 2 of 228 | Corrects Carter/Camp David and queen; introduces `集市` → `吉士` for market women |

Times were 12.70, 10.38 and 76.66 seconds, plus 10.90 seconds for the automatic
short-clip control. Process peak RSS was 12.44 GiB. Those are CPU observations,
not GPU memory or execution acceptance. Character edit counts omit punctuation
and are diagnostics: the zero-edit warning still loses a list relationship.
A preselected source-only control then used the same pinned large-v3 and decoding
on CPU/float32 for all three WAVs: 22.01, 34.80 and 86.26 seconds. It also corrected
mammals, without Qwen's repeated word, but retained the island, queen and historical
name errors. Thus the ecology gain is not unique to Qwen; higher precision did not
recover all facts. Qwen's longer-sample word/role gains remain relative to both
Whisper precisions, while its market-women error is new. Different backends remain
confounded, and these numbers do not establish universal checkpoint superiority.
Earlier float32 native-translation controls still failed, as preserved in the
original investigation; this new control is source-only.

**Decision:** retain large-v3 in the application. The alternative improves some
important source words in this bounded experiment, but still fails complete
reference fidelity. Three read-speech inputs, no French audio evaluation, a new
runtime with different language/metadata behavior, and untested RTX 3080 installation
and memory use do not justify release integration. Half-precision parameter storage
alone is not a peak-VRAM estimate; activations, audio length, caches and runtime
allocations must be measured on the 10 GB device. A future integration would need
sequential model lifetimes, compatible CPU/CUDA packaging and distinct processing
signatures. No experimental source is substituted into existing checkpoints.
Replacing ASR would not change the separate Whisper audio-to-English pass.

## Deployed text-model comparison

Authenticated discovery inventoried all 18 exposed aliases and read-only deployed
model/profile metadata. Context/budget aliases sharing physical weights were
identified; differing conversions of one family were not treated as identical
weights. Exact aliases, URLs, manifests, server profiles, requests, answer text,
usage and timings are retained only in the ignored private handoff/evidence.

Four distinct general-purpose families were selected for their language/instruction
capabilities, including the previously tested Qwen 27B route. Coding-specialized,
uncensored and duplicate-context routes were not prioritized solely for size or
speed. Primary model documentation informed this shortlist:
[Qwen 27B](https://huggingface.co/Qwen/Qwen3.8-27B),
[Qwen Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
[GLM Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), and
[Gemma 4](https://ai.google.dev/gemma/docs/core/model_card_4).
Their published benchmark claims do not certify faithful transcript translation.

Each route received the **same five saved inputs**, existing `faithful-json-v2`
instructions and chunk/context spans: the three unchanged Whisper source texts,
a 1776-character authored French contextual account and a 225-character authored
mixed-language/negation check. The [authored fixtures](translation-evaluation-fixtures.json)
are text checks, not French speech recognition tests. The history source is 251
characters and fits one default request. The French case exercises two adjacent
chunks, `0..1200` and `1200..1776`, with 160 characters of context. Ecology's disclosed
120-character bound still produces one chunk; other cases use 1200. All source
characters are covered exactly once.

Requests contain only model/messages. Effective deployed temperature 1.0 and
top-p 0.95 were common; Qwen uses top-k 20 and xhigh thinking, GLM its high-thinking
preset, and Gemma top-k 64 with thinking disabled. Tested thinking profiles reserve
16384 output tokens; the Gemma route retains its deployed output defaults. Input
budgets exceed these small requests. Client timeout was 600 seconds and attempts
one for this comparison. Endpoint-specific reasoning/conversion behavior differs:
this compares supported deployed routes, not a controlled architecture ranking.
No production defaults were overridden.

| Deployed variant | Fidelity observations |
| --- | --- |
| Qwen3.8-27B, 8-bit conversion | Most conservative on uncertain saved source: retains unfamiliar names with uncertainty, translates misrecognized `往后` as “followed by,” and retains the unclear animals term. Selected. |
| Qwen3.8-Flash-Next, 6-bit conversion | Preserves the ecology term and main facts, but conjectures a queen role from `往后`, even with an uncertainty marker. |
| GLM-5.3-Flash, mixed 4/8-bit conversion | Preserves main numbers/negation, but restores the queen role and familiar names from historical context rather than leaving source uncertainty. |
| Gemma 4 31B, 16-bit conversion | Fluent and retains main facts, but drops the unclear animals modifier and restores familiar historical names. |

All four preserved the checked French roles, dates, payment/negation distinctions,
1400 respondents, 8% versus 6%, two percentage points versus two percent, uncertain
supplier and no promised validation. The original English quotations and accented
product name survive the mixed-language check. No material summarization or
neighbor-context duplication was found in those authored cases. These judgments
are qualitative proposition review, not exact-string or another LLM's scoring.

All **24 responses** ended with `stop` and passed the unchanged strict JSON parser;
there was no observed truncation. Maximum reported completion counts were 7060,
7033, 779 and 277 respectively, including reasoning where the server accounts for
it. Output headroom and valid JSON do not prove semantic completeness. The selected
route's history request took 214.69 seconds, exceeding the generic 180-second
client default. The prepared private configuration therefore uses the already
supported **600-second timeout**, 1200-character chunks, 160-character context
and three bounded transient attempts. Increasing timeout is execution robustness,
not evidence of improved meaning. Generic local installation/defaults are unchanged.

**Selection:** Qwen 27B is the strongest demonstrated choice here for fidelity to
uncertain saved source. It is the previously tested route, now evaluated more
broadly; no superiority over its earlier run or universal best-model claim follows.
Other variants can be preferable for other tasks. The application prompt/parser,
ASR decoding and native translation remain unchanged because no new application
assembly defect was reproduced. A model's plausible repair of an ASR error is not
a reliable substitute for recovering speech accurately.

The selected API retains more facts than the failed native English baseline, but
still fails end-to-end reference fidelity: the island name and missing queen role
in Whisper source persist, and its merged warning list is not recovered. Better
text translation cannot reconstruct speech that is missing or misrecognized.
No released processing path has passed full unreviewed fidelity acceptance.

## Experimental two-stage result

The three Qwen-ASR source texts were also translated through the selected API with
the same existing prompt and 1200/160 text bounds. All three completed. This is an
isolated two-stage experiment, not a released alternative backend or a replacement
of existing source/checkpoints. Corrected source enabled faithful English mammals,
Carter/Camp David and queen, while dates, ages, king's sister and survey facts were
retained. The island remained “Clon”; the newly wrong market-women source became
“Jiushi women [uncertain].” The warning's original list relationship remained lost.

Those are measured improvements in particular end-to-end facts versus the earlier
Whisper-source/API path, but **complete reference fidelity still fails**. Native
Whisper translation was neither replaced nor improved by this experiment. Residual
errors are preserved rather than hidden by a translator's familiar-story repair.

## Real folder and failure acceptance

The chosen private configuration passed the actual `--api-models` and `--api-check`
CLI paths. The latter remains a short authored Mandarin connectivity check, not a
French speech or end-to-end quality test. Real API folder processing reused the
fresh baseline's exact source checkpoints, with ASR/native methods blocked.
Its first batch completed two files and rejected one answer containing an
unescaped JSON control character. The source and previous valid native target
survived; no malformed/incomplete API text was published. That failure is retained
in private evidence. One bounded rerun completed the remaining file, skipped the
other two and repeated no ASR. Parser requirements were not weakened.

A further rerun rebuilt a missing selected API TXT with both ASR and translation
methods blocked: 0 completed, 3 skipped, 0 failed. All six source/target TXT digests
and all Work-state bytes matched before/after; all six historical native/API
records were retained. This validates real recovery/publication, not model accuracy.
The route's occasional malformed answer remains an operational limitation; the
24 successful comparison responses do not establish universal JSON reliability.
