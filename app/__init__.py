"""Reactory Speech Service package.

``__version__`` is the application version.  It is folded into
:func:`app.services.synthesis_cache.renderer_fingerprint`, so a deploy that
changes the built-in lexicon or the normalisation code invalidates persisted
cache entries instead of serving audio rendered by the previous release.
"""

__version__ = "1.0.0"
