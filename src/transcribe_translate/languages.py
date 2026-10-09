"""Language normalization without loading an inference runtime."""

# Whisper's language identifiers. Language labels do not request script conversion.
CODES = frozenset("en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr bg lt la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su yue".split())
ALIASES = {
    "mandarin": "zh", "chinese": "zh", "mandarin chinese": "zh", "cmn": "zh",
    "zho": "zh", "chi": "zh", "中文": "zh", "普通话": "zh", "普通話": "zh",
    "english": "en", "eng": "en", "french": "fr", "fra": "fr", "fre": "fr",
    "spanish": "es", "spa": "es", "german": "de", "deu": "de", "ger": "de",
    "japanese": "ja", "jpn": "ja", "korean": "ko", "kor": "ko",
    "italian": "it", "ita": "it", "portuguese": "pt", "por": "pt",
    "russian": "ru", "rus": "ru", "arabic": "ar", "ara": "ar",
    "cantonese": "yue", "hebrew": "he", "iw": "he", "indonesian": "id",
}


def normalize_language(value: str, *, auto: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError("Language must be a code or language name")
    code = value.strip().lower().replace("_", "-")
    code = ALIASES.get(code, code)
    if code == "auto" and auto:
        return code
    # Region/script aliases indicate spoken language, never forced conversion.
    if "-" in code:
        parts = code.split("-")
        if any(not p.isalpha() or len(p) not in (2, 4) for p in parts[1:]):
            raise ValueError(f"Unsupported language: {value}")
        code = ALIASES.get(parts[0], parts[0])
    if code not in CODES:
        raise ValueError(f"Unsupported language: {value}; use a Whisper language code (e.g. zh, en, fr)")
    return code
