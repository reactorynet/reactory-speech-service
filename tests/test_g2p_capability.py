"""SPEECH-010 requirement 4: the G2P support signal, unit-tested.

The capability probe is the difference between "the voice pack advertises nine
languages" and "this deployment can phonemise eight of them". That claim has to be
observable in three places — the probe's own verdict, ``TTSService``, and
``GET /health`` — and it has to be honest *before* it has measured anything, which
is what the ``pending`` list is for.

Everything here runs without models: the probe is driven through converter doubles
so the branches that are hard to reach with a real espeak-ng install (a language
with no probe sentence, a missing backend, a backend that raises) are covered.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app as fastapi_app
from app.services.g2p_capability import PROBE_SENTENCES, G2PCapability
from app.services.phonetic_converter import (
    PhoneticConversionError,
    UnsupportedLanguageError,
)
from app.services.tts_service import TTSService
from app.services.voice_registry import all_canonical_languages


def converter_calls(capability: G2PCapability) -> int:
    """Number of backend calls the probe has made through its converter."""
    backend = capability._converter
    return len(getattr(backend, "calls", []))


class _StubConverter:
    """A converter double with a fixed phonemisation and a call counter."""

    def __init__(self, phonemes: str = "həlˈoʊ wˈɜːld", available: bool = True) -> None:
        self.phonemes = phonemes
        self.is_available = available
        self.calls: list[tuple[str, str]] = []
        self.raises: Exception | None = None

    def phonemize_raw(self, text: str, language: str = "en-us") -> str:
        self.calls.append((text, language))
        if self.raises is not None:
            raise self.raises
        return self.phonemes


# ---------------------------------------------------------------------------
# The probe's own verdict and its unprobed branches
# ---------------------------------------------------------------------------


def test_a_language_without_a_probe_sentence_is_unsupported_not_blessed():
    """An unmeasured language must never look supported."""
    capability = G2PCapability(converter=_StubConverter())

    support = capability.probe("sv")  # known ISO code, no probe sentence

    assert support.supported is False
    assert support.reason is not None and "no G2P probe sentence" in support.reason
    assert support.language == "sv"


def test_a_missing_backend_is_reported_as_unsupported():
    capability = G2PCapability(converter=_StubConverter(available=False))

    support = capability.probe("en-US")

    assert support.supported is False
    assert support.reason is not None and "not available" in support.reason


def test_a_backend_that_raises_unsupported_is_reported_with_its_reason():
    converter = _StubConverter()
    converter.raises = UnsupportedLanguageError("cmn", "'cmn' is not supported")
    capability = G2PCapability(converter=converter)

    support = capability.probe("zh")

    assert support.supported is False
    assert support.reason is not None and "not supported" in support.reason


def test_a_backend_failure_is_reported_as_a_backend_failure():
    converter = _StubConverter()
    converter.raises = PhoneticConversionError("espeak exploded")
    capability = G2PCapability(converter=converter)

    support = capability.probe("es")

    assert support.supported is False
    assert support.reason is not None and "backend raised" in support.reason


def test_a_language_whose_probe_sentence_is_unreadable_is_unsupported(monkeypatch):
    """The probe applies the same script guard the render path applies.

    A language is only "supported" if the probe sentence can actually be read in
    that language's own script. The stub below returns perfectly good phonemes, so
    only the script guard can produce this refusal.
    """
    monkeypatch.setitem(
        PROBE_SENTENCES, "es", "リアクトリーの音声サービスはとても便利です"
    )
    capability = G2PCapability(converter=_StubConverter())

    support = capability.probe("es")

    assert support.supported is False
    assert support.reason is not None and "script" in support.reason
    assert converter_calls(capability) == 0, "the guard must run before the backend"


def test_spelling_output_marks_the_language_unsupported():
    """One syllable per character is the signature the probe must refuse."""
    # 21 tokens, 2 distinct values for a 30-character Japanese probe sentence:
    # the measured shape of espeak spelling a script out.
    capability = G2PCapability(
        converter=_StubConverter(phonemes=" ".join(["dʒˈapəniːzlˌɛɾɚ"] * 21))
    )

    support = capability.probe("ja")

    assert support.supported is False
    assert support.reason is not None
    assert "one syllable per character" in support.reason


def test_refresh_drops_the_cache_and_reprobes(monkeypatch):
    """``refresh()`` is the operational escape hatch after an espeak pack change."""
    converter = _StubConverter()
    capability = G2PCapability(converter=converter)

    first = capability.probe("en-US")
    calls_after_first = len(converter.calls)
    assert first.supported is True
    assert calls_after_first == 1

    # Cached: a second probe for the same language must not hit the backend.
    capability.probe("en-US")
    assert len(converter.calls) == calls_after_first

    refreshed = capability.refresh()

    assert set(refreshed) == set(all_canonical_languages())
    assert len(converter.calls) > calls_after_first
    assert all(row.supported for row in refreshed.values())


def test_every_canonical_language_has_a_probe_sentence():
    """A language the registry knows but the probe cannot measure is a gap."""
    assert set(PROBE_SENTENCES) == set(all_canonical_languages())


# ---------------------------------------------------------------------------
# The pre-probe state is pending, not supported
# ---------------------------------------------------------------------------


def test_snapshot_before_any_probe_reports_pending_not_supported():
    """A health caller must never read "supported" for something unmeasured."""
    capability = G2PCapability()  # nothing probed yet

    snapshot = capability.snapshot()

    assert snapshot["supported"] == []
    assert snapshot["unsupported"] == []
    assert set(snapshot["pending"]) == set(all_canonical_languages())
    assert snapshot["languages"] == {}
    # The two views never overlap.
    assert not set(snapshot["pending"]) & set(snapshot["supported"])


def test_snapshot_after_probing_moves_every_language_out_of_pending():
    capability = G2PCapability(converter=_StubConverter())
    capability.report()

    snapshot = capability.snapshot()

    assert snapshot["pending"] == []
    assert set(snapshot["languages"]) == set(all_canonical_languages())
    # The stub returns readable phonemes for every probe sentence, so the verdict
    # is "supported" — what matters here is that nothing is left unmeasured.
    assert set(snapshot["supported"]) == set(all_canonical_languages())
    assert snapshot["unsupported"] == []


# ---------------------------------------------------------------------------
# TTSService and GET /health expose it
# ---------------------------------------------------------------------------


def test_tts_service_g2p_capabilities_has_the_documented_shape():
    payload = TTSService().g2p_capabilities()

    assert set(payload) == {"supported", "unsupported", "pending", "languages"}
    assert isinstance(payload["supported"], list)
    assert isinstance(payload["unsupported"], list)
    assert isinstance(payload["pending"], list)
    assert isinstance(payload["languages"], dict)


@pytest.fixture
def real_service_client():
    """Client backed by a real TTSService, so /health reads the real probe."""
    service = TTSService()
    fastapi_app.state.tts_service = service
    fastapi_app.state.stt_service = MagicMock(is_ready=False)
    yield TestClient(fastapi_app)
    fastapi_app.state.tts_service = TTSService()


def test_health_reports_japanese_unsupported_and_chinese_supported(real_service_client):
    """The runtime half of the ticket's requirement 4."""
    response = real_service_client.get("/health")

    assert response.status_code == 200
    g2p = response.json()["g2p"]
    assert g2p is not None

    japanese = g2p["languages"]["ja"]
    assert japanese["supported"] is False
    assert japanese["reason"], "an unsupported language must say why"
    assert "ja" in [row["language"] for row in g2p["unsupported"]]
    assert "ja" not in g2p["supported"]

    chinese = g2p["languages"]["zh"]
    assert chinese["supported"] is True
    assert chinese["locale"] == "cmn"
    assert "zh" in g2p["supported"]


def test_health_g2p_is_absent_when_the_service_cannot_report_it(client):
    """A service double without the probe must not fail /health."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["g2p"] is None
