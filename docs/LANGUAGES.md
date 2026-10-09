# Language reference

Automatic source detection (`source_language = "auto"`) is the generic default.
For known speech, use an explicit supported Whisper language code or accepted
name. The native translation backend produces English (`target_language = "en"`);
the API backend accepts supported target codes, subject to its model's abilities.

| Language | Code / accepted aliases |
| --- | --- |
| French | `fr`, French |
| English | `en`, English |
| Mandarin / Chinese | `zh`, Mandarin, Chinese, `cmn`, `zh-CN`, `zh-TW` |
| Spanish | `es`, Spanish |
| Japanese | `ja`, Japanese |

Additional accepted codes and name aliases are listed in the application's
[language table](../src/transcribe_translate/languages.py). Codes follow Whisper;
not every full language name is an accepted alias.
Mandarin/Chinese aliases normalize to `zh`; they do not request Simplified or
Traditional script conversion. Recognized characters are retained, although ASR
can choose a different spelling/script than a reference. Automatic detection uses
the library's initial-window detection; short or mixed-language speech needs review.
Configuration upgrades preserve existing language overrides.

The Mandarin fixtures in [quality findings](QUALITY.md) remain Mandarin evidence;
the README's French example does not relabel those tests as French acceptance.
