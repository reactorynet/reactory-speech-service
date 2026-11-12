"""SPEECH-010: multilingual G2P correctness.

These tests pin the three defects the ticket measured, and the policy chosen for
each:

1. **Locale threading** — a caller tag ``zh`` must reach espeak-ng as ``cmn``,
   the locale the voice registry already carries. No parallel mapping table.
2. **espeak language-switch markers** — never spoken. A marker is metadata; a
   *repeated* marker pattern is espeak spelling a script out, and is refused with
   ``UnsupportedLanguageError`` (HTTP 422) instead of synthesising nonsense.
3. **Lexicon scope** — the built-in English respellings apply to English only; a
   French utterance gets French phonemes for the brand name.

They are deliberately unit-level: they need espeak-ng (already required by the
suite) but no Kokoro or Whisper model.
"""

from __future__ import annotations

import pytest

from app.core.contracts import EngineRequest
from app.engines.kokoro import KokoroEngine
from app.services.g2p_capability import G2PCapability
from app.services.phonetic_converter import (
    LANGUAGE_SWITCH_MARKER,
    SPELLING_FALLBACK_MARKER_PAIRS,
    PhoneticConversionError,
    UnsupportedLanguageError,
    marks_spelling_fallback,
    phonetic_converter,
    strip_language_markers,
)
from app.services.tts_service import TTSService
from app.services.voice_registry import canonical_language, engine_locale

#: English phonemes the old global lexicon spliced into every language: the
#: ``(en)ɹˈiːˈaktəɹi(fr)`` the ticket measured.
ENGLISH_BRAND_PHONEMES = ("ɹˈiː", "ɹɪˈak", "ɹ̩ˈi", "ɹiˈækt")


# ---------------------------------------------------------------------------
# 1. Locale resolution reads the registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tag", "locale", "canonical"),
    [
        ("zh", "cmn", "zh"),
        ("zh-CN", "cmn", "zh"),
        ("cmn", "cmn", "zh"),
        ("pt", "pt-br", "pt-BR"),
        ("pt-BR", "pt-br", "pt-BR"),
        ("fr-FR", "fr-fr", "fr-FR"),
        ("fr", "fr-fr", "fr-FR"),
        ("en", "en-us", "en-US"),
        ("en-GB", "en-gb", "en-GB"),
        ("ja", "ja", "ja"),
        ("hi", "hi", "hi"),
        ("it", "it", "it"),
    ],
)
def test_language_tag_resolves_to_the_registry_locale(tag, locale, canonical):
    assert engine_locale(tag) == locale
    assert canonical_language(locale) == canonical


def test_unknown_language_passes_through_lower_cased():
    assert engine_locale("XX-YY") == "xx-yy"
    assert canonical_language("xx-yy") is None


def test_chinese_phonemizes_with_the_voice_locale():
    """``zh`` must not reach espeak-ng verbatim: the backend has ``cmn``."""
    phonemes = phonetic_converter.convert_to_phonetic(
        "你好，今天我能帮你做什么？", language="zh"
    )
    assert phonemes
    assert not LANGUAGE_SWITCH_MARKER.search(phonemes)


def test_phonemize_service_reports_the_locale_actually_used():
    result = TTSService().phonemize("你好，今天我能帮你做什么？", language="zh")
    assert result["language"] == "cmn"
    assert result["phonemes"]


def test_phonemize_service_uses_the_voices_locale():
    """A voice id is authoritative: its registry locale drives the phonemizer."""
    result = TTSService().phonemize("你好，今天我能帮你做什么？", voice="zf_xiaoxiao")
    assert result["language"] == "cmn"


# ---------------------------------------------------------------------------
# 2. Language-switch markers are metadata, never phonemes
# ---------------------------------------------------------------------------


def test_markers_are_stripped_and_reported():
    clean, markers = strip_language_markers("lə sɛʁvˈis (en)ɹɪˈaktəɹi(fr) fɔ̃ksjˈɔn")
    assert markers == ["en", "fr"]
    assert "(en)" not in clean and "(fr)" not in clean
    assert "lə sɛʁvˈis" in clean


def test_a_marker_never_reaches_the_vocoder_in_conversion():
    """The measured Japanese sample is refused, not returned with markers."""
    with pytest.raises(UnsupportedLanguageError):
        phonetic_converter.convert_to_phonetic(
            "リアクトリーの音声サービスはとても便利です", language="ja"
        )


def test_spelling_fallback_is_refused(monkeypatch):
    """Three or more switch pairs is spelling, not a loanword: refuse it."""

    class _SpellingTokenizer:
        def phonemize(self, text, lang="en-us", norm=True):  # noqa: ARG002
            return "(en)dʒˈapəniːz(ja)lˈetə " * SPELLING_FALLBACK_MARKER_PAIRS

    monkeypatch.setattr(phonetic_converter, "_tokenizer", _SpellingTokenizer())
    with pytest.raises(UnsupportedLanguageError) as excinfo:
        phonetic_converter.convert_to_phonetic("日本語", language="ja")
    assert "spelling" in str(excinfo.value).lower()


def test_loanword_switches_are_not_mistaken_for_spelling():
    """The fallback guard must scale with the input, or it refuses tech sentences.

    Measured: a 66-character French sentence naming three English products emits
    *one* switch pair; the 21-character Japanese sentence emits 21. Counting pairs
    alone would have made the first sentence a 422.
    """
    french_tech = "Le service Reactory et GraphQL et ONNX fonctionnent bien ensemble."
    assert not marks_spelling_fallback(french_tech, 3)

    japanese = "リアクトリーの音声サービスはとても便利です"
    assert marks_spelling_fallback(japanese, 21)

    # A short string is still caught by the fixed floor.
    assert marks_spelling_fallback("日本語", SPELLING_FALLBACK_MARKER_PAIRS)


def test_a_single_loanword_marker_is_stripped_not_refused(monkeypatch):
    class _LoanwordTokenizer:
        def phonemize(self, text, lang="en-us", norm=True):  # noqa: ARG002
            return "lə sɛʁvˈis (en)ɹɪˈaktəɹi(fr)"

    monkeypatch.setattr(phonetic_converter, "_tokenizer", _LoanwordTokenizer())
    phonemes = phonetic_converter.convert_to_phonetic("Le service Reactory", language="fr-fr")
    assert "(en)" not in phonemes and "(fr)" not in phonemes


def test_unsupported_backend_language_is_a_422_class_error():
    with pytest.raises(UnsupportedLanguageError):
        phonetic_converter.convert_to_phonetic("hello", language="zz")


def test_probe_marks_japanese_unsupported_and_chinese_supported():
    capability = G2PCapability()
    japanese = capability.probe("ja")
    chinese = capability.probe("zh")
    assert japanese.supported is False
    assert japanese.reason and "spells" in japanese.reason
    assert chinese.supported is True
    assert chinese.locale == "cmn"


# ---------------------------------------------------------------------------
# 3. The lexicon is language-scoped
# ---------------------------------------------------------------------------


def test_french_reactory_contains_no_english_phonemes():
    """The ticket's acceptance test, literally."""
    phonemes = phonetic_converter.convert_to_phonetic(
        "Le service Reactory fonctionne bien", language="fr-fr"
    )
    assert "(en)" not in phonemes
    assert not LANGUAGE_SWITCH_MARKER.search(phonemes)
    for english in ENGLISH_BRAND_PHONEMES:
        assert english not in phonemes, f"English phoneme {english!r} leaked into {phonemes!r}"
    # French uses a uvular ʁ; an English ɹ would mean the switch survived.
    assert "ɹ" not in phonemes


def test_english_still_gets_the_builtin_respelling():
    assert "Ree-actory" in phonetic_converter.normalize_text(
        "Reactory uses GraphQL", language="en-us"
    )


def test_builtin_lexicon_does_not_apply_to_other_languages():
    normalised = phonetic_converter.normalize_text(
        "Le service Reactory", language="fr-fr"
    )
    assert normalised == "Le service Reactory"
    assert "Ree-actory" not in normalised


def test_unscoped_custom_entry_is_english_only():
    """A shipped respelling applied to French is the defect being fixed."""
    entry = {"replacement": "Ree-actory", "ipa": "ɹiˈæktɚɹi"}
    assert phonetic_converter.entry_for_language(entry, "en-us") is entry
    assert phonetic_converter.entry_for_language(entry, "fr-fr") is None


def test_scoped_custom_entry_applies_to_its_language_only():
    entry = {"ipa": "ʁeaktoʁi", "languages": ["fr-FR"]}
    assert phonetic_converter.entry_for_language(entry, "fr-fr") is entry
    assert phonetic_converter.entry_for_language(entry, "es") is None


# ---------------------------------------------------------------------------
# 4. Capabilities and synthesis refuse honestly
# ---------------------------------------------------------------------------


def test_capabilities_exclude_japanese_and_name_the_reason():
    capabilities = KokoroEngine().capabilities()
    assert "ja" not in capabilities.languages
    assert "zh" in capabilities.languages
    assert capabilities.unsupported_languages == ["ja"]
    japanese = next(row for row in capabilities.language_support if row["language"] == "ja")
    assert japanese["supported"] is False
    assert japanese["reason"]


def test_phonemize_of_an_unsupported_language_raises():
    with pytest.raises(UnsupportedLanguageError):
        TTSService().phonemize("リアクトリー", language="ja")


def test_engine_refuses_synthesis_for_an_unsupported_language():
    """The guard runs before any render, so a stub model is never touched."""

    class _StubModel:
        voices = {"jf_alpha": object()}

        def create(self, *args, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("the engine must refuse before rendering")

    engine = KokoroEngine()
    engine.model = _StubModel()
    engine.set_ready(True)

    with pytest.raises(UnsupportedLanguageError):
        engine.synthesize(EngineRequest(text="リアクトリーの音声", voice="jf_alpha"))


def test_engine_allows_caller_supplied_phonemes_for_an_unsupported_language():
    """``is_phonemes=True`` does no G2P, so there is nothing to refuse."""

    class _StubModel:
        voices = {"jf_alpha": object()}

        def create(self, text, voice=None, speed=1.0, lang=None, is_phonemes=False):
            assert text == "ɽi"
            return ([0.0] * 2400, 24000)

    engine = KokoroEngine()
    engine.model = _StubModel()
    engine.set_ready(True)

    result = engine.synthesize(
        EngineRequest(text="ɽi", voice="jf_alpha", is_phonemes=True)
    )
    assert result.duration == pytest.approx(0.1, rel=0.01)


def test_streaming_path_uses_the_voice_locale_and_never_leaks_markers():
    """The streaming call site must carry the same locale and marker policy.

    SPEECH-010 scope: "check every call site, including streaming". ``stream``
    splits the text and renders each span through ``synthesize_span``; a Chinese
    voice must reach the phonemizer as ``cmn``, and no ``(en)``/``(ja)`` marker may
    reach ``create``.
    """

    class _StubModel:
        voices = {"zf_xiaoxiao": object(), "jf_alpha": object()}

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def create(self, text, voice=None, speed=1.0, lang=None, is_phonemes=False):
            self.calls.append(
                {"text": text, "lang": lang, "is_phonemes": is_phonemes, "voice": voice}
            )
            return ([0.0] * 2400, 24000)

    engine = KokoroEngine()
    engine.model = _StubModel()
    engine.set_ready(True)

    chunks = list(engine.stream(EngineRequest(text="你好。", voice="zf_xiaoxiao")))
    assert chunks
    chinese_call = engine.model.calls[-1]
    assert chinese_call["lang"] == "cmn"
    assert chinese_call["is_phonemes"] is True
    assert "(" not in chinese_call["text"]

    with pytest.raises(UnsupportedLanguageError):
        list(engine.stream(EngineRequest(text="こんにちは。", voice="jf_alpha")))


def test_phonetic_conversion_error_is_still_reported():
    """A genuine backend failure is not silently labelled 'unsupported'."""

    class _BrokenTokenizer:
        def phonemize(self, text, lang="en-us", norm=True):  # noqa: ARG002
            raise RuntimeError("espeak exploded")

    converter = type(phonetic_converter)()
    converter._tokenizer = _BrokenTokenizer()
    converter._ready = True
    with pytest.raises(PhoneticConversionError):
        converter.convert_to_phonetic("hello", language="en-us")

# ---------------------------------------------------------------------------
# F1: a script the voice cannot read is refused on *both* paths
# ---------------------------------------------------------------------------
#
# The marker guard alone was not enough. A Japanese sentence sent to an English
# voice makes espeak-ng transliterate every character without emitting a single
# marker (measured: `dʒˈæpəniːzlˌɛɾɚ` x 21, 20.9 s of audio, HTTP 200), so the
# review reproduced the original ticket against `af_heart`. Two independent
# guards close it:
#
#   * deterministic, before phonemisation: the text's script must be one the
#     voice's locale can read;
#   * statistical, after phonemisation: one syllable per character is spelling.
#
# Both are asserted here, and every refusal asserts the model was never reached.

JAPANESE_SENTENCE = "リアクトリーの音声サービスはとても便利です"
CHINESE_SENTENCE = "你好，今天我能帮你做什么？"


class _RecordingModel:
    """A Kokoro double that fails loudly if a render is attempted."""

    def __init__(
        self, voice_ids: tuple[str, ...] = ("af_heart", "bm_george", "jf_alpha")
    ) -> None:
        self.voices = {voice_id: object() for voice_id in voice_ids}
        self.calls: list[dict] = []

    def create(self, text, voice=None, speed=1.0, lang=None, is_phonemes=False):
        self.calls.append({"text": text, "lang": lang, "is_phonemes": is_phonemes})
        return ([0.0] * 2400, 24000)


def _engine_with_recording_model() -> tuple[KokoroEngine, _RecordingModel]:
    engine = KokoroEngine()
    model = _RecordingModel()
    engine.model = model
    engine.set_ready(True)
    return engine, model


@pytest.mark.parametrize(
    ("label", "text", "voice"),
    [
        ("japanese text, explicit English voice", JAPANESE_SENTENCE, "af_heart"),
        ("japanese text, default voice", JAPANESE_SENTENCE, None),
        ("mixed english + japanese", f"Hello Reactory, {JAPANESE_SENTENCE}", "af_heart"),
        ("chinese text on an English voice", CHINESE_SENTENCE, "af_heart"),
        # A second English family for the same case: British English must refuse
        # Japanese text too, and it proves the guard is locale-driven rather than
        # hard-coded to the American locale.
        ("japanese text on a British English voice", JAPANESE_SENTENCE, "bm_george"),
    ],
)
def test_wrong_script_for_the_voice_is_refused_before_any_render(label, text, voice):
    """No ~20 s WAV: the engine raises and the model is never called.

    ``calls == []`` is the load-bearing assertion. A 200 with valid audio that
    says the wrong thing is exactly the failure this ticket is about, so the test
    proves nothing reached the vocoder rather than trusting the status code.
    """
    engine, model = _engine_with_recording_model()

    with pytest.raises(UnsupportedLanguageError) as excinfo:
        engine.synthesize(EngineRequest(text=text, voice=voice))

    assert model.calls == [], f"{label} reached the model"
    assert "script" in str(excinfo.value).lower()


@pytest.mark.parametrize(
    ("text", "language"),
    [
        (JAPANESE_SENTENCE, "en-us"),
        (JAPANESE_SENTENCE, "es"),
        (JAPANESE_SENTENCE, "it"),
        (JAPANESE_SENTENCE, "pt-br"),
        (JAPANESE_SENTENCE, "fr-fr"),
        (CHINESE_SENTENCE, "en-us"),
        (f"Hello Reactory, {JAPANESE_SENTENCE}", "en-us"),
    ],
)
def test_converter_refuses_a_script_the_locale_cannot_read(text, language):
    """The same guard at the converter, for every locale espeak would 'read'."""
    with pytest.raises(UnsupportedLanguageError) as excinfo:
        phonetic_converter.convert_to_phonetic(text, language=language)

    assert "script" in str(excinfo.value).lower()


@pytest.mark.parametrize(
    ("text", "language"),
    [
        # Every supported language's own script must still pass. A guard that
        # refuses these would be worse than the bug it fixes.
        ("Hi Werner, how can I help?", "en-us"),
        ("Hola Werner, ¿en qué puedo ayudarte hoy?", "es"),
        ("Olá Werner, como posso ajudar você hoje?", "pt-br"),
        ("Bonjour Werner, comment puis-je vous aider ?", "fr-fr"),
        ("Ciao Werner, come posso aiutarti oggi?", "it"),
        ("नमस्ते वर्नर, मैं आपकी कैसे मदद कर सकता हूँ?", "hi"),
        (CHINESE_SENTENCE, "zh"),
        ("Reactory 平台把支持工单转换成可以审查的代码变更。", "zh"),
    ],
)
def test_the_script_guard_does_not_refuse_readable_text(text, language):
    assert phonetic_converter.convert_to_phonetic(text, language=language)


def test_the_spelling_output_guard_catches_a_script_the_table_does_not_know(monkeypatch):
    """The output guard is the backstop for a script not enumerated above.

    The converter is given a tokenizer that emits the measured spelling signature
    for text the table considers readable, so only the statistical check can
    produce the refusal.
    """

    class _SpellingTokenizer:
        def phonemize(self, text, lang="en-us", norm=True):  # noqa: ARG002
            return " ".join(["dʒˈæpəniːzlˌɛɾɚ"] * 21)

    converter = type(phonetic_converter)()
    converter._tokenizer = _SpellingTokenizer()
    converter._ready = True

    with pytest.raises(UnsupportedLanguageError) as excinfo:
        converter.convert_to_phonetic("Solaris und Luna", language="de")

    assert "one syllable per character" in str(excinfo.value)


def test_the_spelling_output_guard_ignores_ordinary_repetition(monkeypatch):
    """A repeated word is not spelling: the per-character condition must hold too."""

    class _RepetitiveTokenizer:
        def phonemize(self, text, lang="en-us", norm=True):  # noqa: ARG002
            return " ".join(["nˈoʊ"] * 6)

    converter = type(phonetic_converter)()
    converter._tokenizer = _RepetitiveTokenizer()
    converter._ready = True

    # 12 non-space characters, 6 tokens -> 0.5 tokens/char, below the threshold.
    assert converter.convert_to_phonetic("no no no no no no", language="en-us")


# ---------------------------------------------------------------------------
# N1: the brand name is native in es / pt / it (no models needed)
# ---------------------------------------------------------------------------


def test_spanish_brand_name_uses_spanish_phonemes():
    phonemes = phonetic_converter.convert_to_phonetic(
        "El servicio de voz de Reactory funciona bien.", language="es"
    )

    assert "rˌeaktˈoɾi" in phonemes
    assert "ɹ" not in phonemes
    assert not LANGUAGE_SWITCH_MARKER.search(phonemes)


def test_portuguese_brand_name_uses_portuguese_phonemes():
    phonemes = phonetic_converter.convert_to_phonetic(
        "O serviço de voz da Reactory funciona bem.", language="pt-br"
    )

    assert "xˌeaktˈoɾi" in phonemes
    assert "ɹ" not in phonemes
    assert not LANGUAGE_SWITCH_MARKER.search(phonemes)


def test_italian_brand_name_uses_italian_phonemes():
    phonemes = phonetic_converter.convert_to_phonetic(
        "Il servizio vocale di Reactory funziona bene.", language="it"
    )

    assert "reaktˈorɪ" in phonemes
    assert "ɹ" not in phonemes
    assert not LANGUAGE_SWITCH_MARKER.search(phonemes)