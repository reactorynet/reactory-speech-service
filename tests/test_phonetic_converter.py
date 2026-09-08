"""Unit tests for the phonetic converter, pronunciation lexicon, and markup."""

from app.services.phonetic_converter import phonetic_converter


def test_phonetic_converter_availability():
    """Test that the phonetic converter is properly initialized."""
    assert phonetic_converter is not None
    assert phonetic_converter.is_available is True


def test_text_normalization():
    """Test text normalization expansions."""
    # Test abbreviations & units
    text = "Dr. Smith said etc. and approx. 10 km."
    normalized = phonetic_converter.normalize_text(text)
    assert "Doctor" in normalized
    assert "et cetera" in normalized
    assert "approximately" in normalized
    assert "kilometers" in normalized

    # Test builtin lexicon
    text = "Reactory uses GraphQL and REST APIs with ONNX."
    normalized = phonetic_converter.normalize_text(text)
    assert "Ree-actory" in normalized
    assert "Graph Q L" in normalized
    assert "A P Is" in normalized
    assert "on-ix" in normalized


def test_phonetic_conversion():
    """Test converting text to IPA phonemes."""
    phonemes = phonetic_converter.convert_to_phonetic("Hello world!")
    assert isinstance(phonemes, str)
    assert len(phonemes) > 0
    assert "həlˈoʊ" in phonemes or "wˈɜːld" in phonemes


def test_custom_lexicon_crud():
    """Test adding, retrieving, and deleting custom pronunciation entries."""
    # Add an entry
    res = phonetic_converter.set_entry("Kubernetes", replacement="koo-ber-net-eez", ipa="kˈuːbɚnˌɛtiːz")
    assert res["word"] == "Kubernetes"
    assert res["replacement"] == "koo-ber-net-eez"
    assert res["ipa"] == "kˈuːbɚnˌɛtiːz"

    # Get entry
    entry = phonetic_converter.get_entry("Kubernetes")
    assert entry is not None
    assert entry["replacement"] == "koo-ber-net-eez"

    # Delete entry
    deleted = phonetic_converter.delete_entry("Kubernetes")
    assert deleted is True
    assert phonetic_converter.get_entry("Kubernetes") is None


def test_per_request_custom_lexicon():
    """Test per-request lexicon overrides without global persistence."""
    text = "Deploy the container to K8s."
    # With custom override
    phonemes = phonetic_converter.convert_to_phonetic(
        text,
        custom_lexicon={"K8s": "Kubernetes cluster"}
    )
    assert "kˈuːbɚn" in phonemes or "klˈʌstɚ" in phonemes or "kjuːb" in phonemes


def test_inline_bracket_and_ssml_markup():
    """Test inline bracket notation and SSML phoneme tags."""
    # Bracket respelling
    text1 = "Welcome [Werner|Vair-ner] to Reactory!"
    norm1 = phonetic_converter.normalize_text(text1)
    assert "Vair-ner" in norm1

    # Direct IPA bracket notation
    text2 = "Welcome [Werner|/vˈɛərnər/] to Reactory!"
    phonemes2 = phonetic_converter.convert_to_phonetic(text2)
    assert "vˈɛərnər" in phonemes2

    # SSML phoneme tag
    text3 = 'Welcome <phoneme ph="vˈɛərnər">Werner</phoneme> to Reactory!'
    phonemes3 = phonetic_converter.convert_to_phonetic(text3)
    assert "vˈɛərnər" in phonemes3
