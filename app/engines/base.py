"""Shared engine-adapter plumbing: errors and direction lowering.

Lowering is the fixed degradation order from ADR-0002:

1. use the engine's native control when it exists;
2. otherwise post-process;
3. otherwise ignore it and report it in ``ignored``.

Post-processing (time-stretch, pitch-shift, gain) is deliberately **not**
implemented in Phase 1 — it belongs with the expressive work in a later ticket.
So a control the engine cannot render natively is reported as ignored with a
human-readable warning.  We never claim a control was applied when it was not.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.contracts import Direction, DirectionPlan, EngineCapabilities

logger = logging.getLogger(__name__)


class TTSEngineError(Exception):
    """Base class for engine-level failures."""


class EngineUnavailableError(TTSEngineError, RuntimeError):
    """The engine is registered but cannot render right now (model not loaded).

    The router treats this as a fallback trigger: a later engine in the ordered
    list may still be able to serve the request.

    Also a :class:`RuntimeError` so callers written against the pre-adapter
    ``TTSService`` (which raised ``RuntimeError("TTS service not initialized")``)
    keep working unchanged.
    """


class EngineCapacityError(EngineUnavailableError):
    """``max_loaded_engines`` is reached and no loaded engine can be evicted.

    Subclasses :class:`EngineUnavailableError` so the router keeps its fallback
    behaviour: another engine may still serve the request.  It is deliberately
    distinct so the reason is visible in logs and in tests.
    """


class UnknownVoiceError(ValueError):
    """The requested voice is not exposed by the resolved engine.

    Subclasses :class:`ValueError` on purpose: the HTTP layer has always mapped
    synthesis ``ValueError`` to 422, and that contract is unchanged.
    """


class UnknownEngineError(ValueError):
    """An explicit engine override names an engine that is not registered."""


def _format_number(value: float) -> str:
    return f"{value:+g}"


def lower_direction(direction: Direction, capabilities: EngineCapabilities) -> DirectionPlan:
    """Lower a provider-neutral direction into one engine's native controls.

    Only controls the caller actually expressed (pydantic ``model_fields_set``)
    are considered, so a default :class:`Direction` reports nothing and a legacy
    request without ``speed`` stays silent.
    """
    applied: dict[str, Any] = {}
    ignored: list[str] = []
    warnings: list[str] = []

    engine = capabilities.engine
    prosody = direction.prosody
    prosody_fields = set(prosody.model_fields_set)
    direction_fields = set(direction.model_fields_set)

    if "rate" in prosody_fields and prosody.rate is not None:
        if capabilities.native_rate:
            applied["rate"] = prosody.rate
        else:
            ignored.append("rate")
            warnings.append(
                f"{engine} has no native rate control and post-processing is not enabled; "
                f"rate {prosody.rate} was not applied"
            )

    # A zero pitch/volume is the neutral value, so it needs no application and no
    # warning: reporting it as "ignored" would be noise, not honesty.
    if "pitch" in prosody_fields and prosody.pitch not in (None, 0.0):
        if capabilities.native_pitch:
            applied["pitch"] = prosody.pitch
        else:
            ignored.append("pitch")
            warnings.append(
                f"{engine} has no pitch control and post-processing is not enabled; "
                f"pitch {_format_number(float(prosody.pitch))} semitones was not applied"
            )

    if "volume" in prosody_fields and prosody.volume not in (None, 0.0):
        if capabilities.native_volume:
            applied["volume"] = prosody.volume
        else:
            ignored.append("volume")
            warnings.append(
                f"{engine} has no volume control and post-processing is not enabled; "
                f"volume {_format_number(float(prosody.volume))} dB was not applied"
            )

    if "pauses" in prosody_fields:
        if prosody.pauses == "natural":
            # Punctuation-driven pausing is what the engine does by default.
            applied["pauses"] = "natural"
        else:
            ignored.append("pauses")
            warnings.append(
                f"{engine} only pauses at punctuation; '{prosody.pauses}' pause shaping "
                "requires silence insertion, which is not enabled"
            )

    style = direction.style
    if "style" in direction_fields and style not in (None, "neutral"):
        if style in capabilities.styles:
            applied["style"] = style
        else:
            ignored.append("style")
            warnings.append(
                f"{engine} does not support emotion; style '{style}' was not applied"
            )

    if (
        "intensity" in direction_fields
        and direction.intensity is not None
        and "style" not in applied
    ):
        ignored.append("intensity")
        if style in (None, "neutral"):
            warnings.append(
                f"intensity is only meaningful with a style {engine} can express; "
                "it was not applied"
            )

    return DirectionPlan(
        engine=engine,
        applied=applied,
        ignored=ignored,
        warnings=warnings,
    )
