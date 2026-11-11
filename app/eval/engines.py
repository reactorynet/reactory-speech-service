"""Engine discovery for the evaluation harness.

Adding an engine must not require a harness code change (ticket acceptance
criterion). Three registration paths exist, in increasing order of coupling:

1. **Built in** — :data:`BUILTIN_ENGINES` maps the engine ids shipped in this
   repository to their adapter classes. Kokoro is the only entry today.
2. **Entry point** — any installed distribution can advertise a factory under the
   ``reactory.speech.eval_engines`` group::

       [project.entry-points."reactory.speech.eval_engines"]
       chatterbox = "app.engines.chatterbox:ChatterboxEngine"

3. **Command line** — ``--register module:attr`` for an adapter that is not
   packaged yet (a local experiment, a vendor checkout).

All three yield the same thing: an object implementing
:class:`app.core.contracts.TTSEngine`. The harness only ever calls ``capabilities``,
``voices``, ``synthesize`` and (optionally) ``stream``, so a new engine is an
adapter plus a corpus run.

A factory is either a class or a zero-argument callable. The engine id is taken
from the object's ``id`` attribute, so an adapter cannot disagree with itself
about its name.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from app.engines.kokoro import KokoroEngine

#: Entry-point group an out-of-tree engine adapter advertises itself under.
ENTRY_POINT_GROUP = "reactory.speech.eval_engines"

#: Engines that ship in this repository.
BUILTIN_ENGINES: dict[str, Callable[[], Any]] = {
    "kokoro": KokoroEngine,
}


class EvalEngineError(RuntimeError):
    """An engine cannot be constructed, registered or loaded."""


#: Process-wide additions from :func:`register_engine` (tests, plugins).
_REGISTERED: dict[str, Callable[[], Any]] = {}


def register_engine(
    engine_id: str,
    factory: Callable[[], Any],
    *,
    replace: bool = False,
) -> None:
    """Register an engine factory for this process.

    Re-registering an id raises unless ``replace`` is set, so two plugins cannot
    quietly shadow each other and a test cannot leak a stub into a later test
    without saying so.
    """
    if not engine_id:
        raise EvalEngineError("an engine id is required")
    if engine_id in BUILTIN_ENGINES or (engine_id in _REGISTERED and not replace):
        if not replace:
            raise EvalEngineError(
                f"engine '{engine_id}' is already registered; pass replace=True to override"
            )
    _REGISTERED[engine_id] = factory


def unregister_engine(engine_id: str) -> None:
    """Remove a process-local registration (no-op for built-in/entry-point ids)."""
    _REGISTERED.pop(engine_id, None)


def _entry_point_factories() -> dict[str, Callable[[], Any]]:
    """Factories advertised by installed distributions, keyed by entry-point name."""
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - Python < 3.8
        return {}
    found: dict[str, Callable[[], Any]] = {}
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        try:
            found[str(entry_point.name)] = entry_point.load()
        except Exception as exc:  # pragma: no cover - a broken plugin must not kill a run
            raise EvalEngineError(
                f"engine entry point '{entry_point.name}' failed to load: {exc}"
            ) from exc
    return found


def catalogue() -> dict[str, Callable[[], Any]]:
    """All known engine factories: built in, registered, then entry points."""
    resolved: dict[str, Callable[[], Any]] = dict(BUILTIN_ENGINES)
    resolved.update(_REGISTERED)
    for engine_id, factory in _entry_point_factories().items():
        resolved.setdefault(engine_id, factory)
    return resolved


def available_engines() -> list[str]:
    """Ids of every engine this deployment can construct, sorted."""
    return sorted(catalogue())


def _load_spec(spec: str) -> tuple[str, Any]:
    """Construct one engine from ``engine_id`` or ``module:attr``."""
    registry = catalogue()
    if spec in registry:
        engine = registry[spec]()
        engine_id = str(getattr(engine, "id", spec))
        return engine_id, engine

    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise EvalEngineError(
            f"unknown engine {spec!r}. Known engines: {', '.join(available_engines()) or 'none'}. "
            "Use 'module:attr' to register an adapter that is not packaged."
        )
    module = importlib.import_module(module_name)
    target = getattr(module, attribute)
    # A class is instantiated; a factory callable is called; an adapter instance is
    # used as given. Checking ``isinstance(type)`` first matters because a class
    # *has* a ``synthesize`` attribute (the unbound function) and must not be
    # mistaken for an instance.
    if isinstance(target, type):
        engine = target()
    elif callable(target) and not hasattr(target, "synthesize"):
        engine = target()
    else:
        engine = target
    engine_id = str(getattr(engine, "id", "") or attribute)
    return engine_id, engine


def _initialise(engine: Any) -> None:
    """Bring an engine to a ready state without an event loop of its own.

    Kokoro exposes a synchronous ``load()``; an adapter that only has the async
    lifecycle hook is initialised with ``asyncio.run``. A factory that returns an
    already-ready adapter (the test doubles) is left alone.
    """
    if getattr(engine, "is_ready", False):
        return
    load = getattr(engine, "load", None)
    if callable(load):
        load()
    elif callable(getattr(engine, "initialize", None)):
        asyncio.run(engine.initialize())
    if hasattr(engine, "is_ready") and not getattr(engine, "is_ready"):
        raise EvalEngineError(
            f"engine '{getattr(engine, 'id', '?')}' did not become ready; "
            "mount its model files (see scripts/download_models.py) before evaluating it"
        )


def resolve_engines(
    specs: Sequence[str],
    *,
    register: Iterable[str] = (),
    initialise: bool = True,
) -> list[Any]:
    """Build the engine instances a run should measure.

    ``specs`` may contain engine ids, ``module:attr`` specs, or the literal
    ``all`` (every constructible engine). ``register`` is applied first, so
    ``resolve_engines(["chatterbox"], register=["app.engines.chatterbox:ChatterboxEngine"])``
    works without touching this module.
    """
    for spec in register:
        engine_id, engine = _load_spec(spec)
        register_engine(engine_id, lambda engine=engine: engine, replace=True)

    requested: list[str] = []
    for spec in specs:
        if spec == "all":
            requested.extend(available_engines())
        else:
            requested.append(spec)

    engines: list[Any] = []
    seen: set[str] = set()
    for spec in requested:
        engine_id, engine = _load_spec(spec)
        if engine_id in seen:
            continue
        seen.add(engine_id)
        if initialise:
            _initialise(engine)
        engines.append(engine)
    if not engines:
        raise EvalEngineError("no engines selected; pass --engines <id>[,<id>]")
    return engines


def engine_identity(engine: Any) -> dict[str, Any]:
    """Stable description of an engine for the report header.

    Deliberately small and serialisable: id, name, licence, attribution and the
    native sample rates. It is part of what makes a report comparable, so it must
    not include a timestamp or a memory address.
    """
    capabilities = None
    describe = getattr(engine, "capabilities", None)
    if callable(describe):
        capabilities = describe()
    return {
        "id": str(getattr(engine, "id", "unknown")),
        "name": getattr(engine, "name", None),
        "license": getattr(engine, "license", None),
        "attribution": getattr(engine, "attribution", None),
        "sample_rates": list(getattr(capabilities, "sample_rates", []) or []),
        "native_rate": bool(getattr(capabilities, "native_rate", False)),
        "styles": list(getattr(capabilities, "styles", []) or []),
        "supports_streaming": bool(getattr(capabilities, "supports_streaming", False)),
    }


__all__ = [
    "BUILTIN_ENGINES",
    "ENTRY_POINT_GROUP",
    "EvalEngineError",
    "available_engines",
    "catalogue",
    "engine_identity",
    "register_engine",
    "resolve_engines",
    "unregister_engine",
]
