"""TTS engine adapters and the engine router.

Importing this package must stay cheap: ``app.engines.kokoro`` imports the
registry and the phonetic converter, but no model runtime is loaded until
:meth:`app.engines.kokoro.KokoroEngine.initialize` runs.
"""
