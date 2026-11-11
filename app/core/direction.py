"""Merging of the legacy ``speed`` alias into the provider-neutral direction.

``speed`` predates :class:`app.core.contracts.Direction` and remains accepted on
``POST /api/tts/synthesize``.  It is an alias for ``direction.prosody.rate``:

* ``speed`` only            → ``direction.prosody.rate = speed``;
* ``direction`` only        → used as supplied;
* both, with an explicit rate → **direction wins** and the response says so in
  ``direction_warnings``.

The function is pure and transport-free so the router, the orchestrator and the
tests all agree on one implementation.
"""

from __future__ import annotations

from app.core.contracts import Direction, Prosody


def resolve_direction(
    speed: float | None,
    direction: Direction | None,
) -> tuple[Direction, list[str]]:
    """Return the effective direction and any alias warnings.

    The returned direction is always a fresh copy; the caller's model is never
    mutated.
    """
    if direction is None:
        if speed is None:
            return Direction(), []
        return Direction(prosody=Prosody(rate=float(speed))), []

    merged = direction.model_copy(deep=True)
    if speed is None:
        return merged, []

    warnings: list[str] = []
    if "rate" in direction.prosody.model_fields_set and direction.prosody.rate is not None:
        warnings.append(
            f"speed is an alias for direction.prosody.rate; "
            f"direction.prosody.rate={direction.prosody.rate} took precedence over speed={speed}"
        )
    else:
        merged.prosody.rate = float(speed)
    return merged, warnings
