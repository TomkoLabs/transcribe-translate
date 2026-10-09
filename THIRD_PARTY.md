# Attribution and third-party conditions

This application is MIT licensed. Its small fresh implementation was informed by
the MIT-licensed `transcribe-media` reference's install/run workflow, folder
convention, CTranslate2 pin and atomic-recovery approach. No diarization, speaker
review, emotion or analysis implementation was imported. The upstream copyright
is retained in this project's MIT notice.

Dependencies and model weights retain their own licenses; the application license
does not replace them. Consult the exact upstream distributions/model cards before
redistributing binaries or weights:

| Component | Upstream license/reference |
| --- | --- |
| faster-whisper | [MIT](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE) |
| CTranslate2 | [MIT](https://github.com/OpenNMT/CTranslate2/blob/master/LICENSE) |
| PyAV | [BSD-3-Clause](https://github.com/PyAV-Org/PyAV/blob/main/LICENSE.txt); bundled FFmpeg build has its own licenses/notices |
| FFmpeg | [LGPL/GPL depending on build](https://ffmpeg.org/legal.html); no FFmpeg executables/binaries are committed here |
| Whisper large-v3 converted weights | [Systran model card](https://huggingface.co/Systran/faster-whisper-large-v3), MIT metadata; [OpenAI Whisper license](https://github.com/openai/whisper/blob/main/LICENSE) |
| HTTPX | [BSD-3-Clause](https://github.com/encode/httpx/blob/master/LICENSE.md) |
| Hugging Face Hub | [Apache-2.0](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE) |
| NVIDIA optional cuBLAS/cuDNN | NVIDIA package/runtime terms; not covered by this application's MIT license |
| Translation endpoint/model | Your provider/checkpoint terms; configure an authorized endpoint and review its data handling |

Bounded human Mandarin source transcription, native speech translation and API
acceptance used the Google FLEURS `cmn_hans_cn` validation
subset under [CC-BY-4.0](https://huggingface.co/datasets/google/fleurs). Attribution:
Conneau et al., *FLEURS: Few-shot Learning Evaluation of Universal Representations
of Speech* (2022), Google. Audio/reference files and generated transcripts remain
outside tracked source; no dataset is bundled. The fixture audio was placed in
video containers for testing, without synthesizing the speech. That test does not
grant redistribution rights for unrelated personal recordings.

The October 6 quality closeout also used three additional dev clips, concatenated
without trimming/filtering into a disclosed 66.72-second compilation. Exact pinned
IDs, filenames, sample counts and hashes are in
[quality-fixtures.json](docs/quality-fixtures.json); findings are in
[QUALITY.md](docs/QUALITY.md). This is an edited collection of human reads, not a
natural continuous recording. No test media is bundled with the release.
