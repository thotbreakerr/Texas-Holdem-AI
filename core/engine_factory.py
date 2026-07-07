"""Selects the poker rules engine: the Poker-Engine-backed adapter (default)
or the legacy in-repo engine.

Importing this module is cheap and pulls in NEITHER engine — both are imported
lazily inside :func:`make_table`. Callers pick an engine by passing
``engine_impl`` or setting the ``THAI_ENGINE_IMPL`` environment variable; the
default is ``pe`` (Poker-Engine) as of P6 of the engine migration.

The legacy engine (``core.engine.Table``) is DEPRECATED: it stays reachable via
``engine_impl="legacy"`` / ``THAI_ENGINE_IMPL=legacy`` so the ~50 ``sanity_*.py``
regression tests keep exercising it, but new work — training runs, evals — runs
on ``pe``. See ENGINE_MIGRATION_PLAN.md and PROGRESS.md.
"""

import os
import random
from typing import Optional

# Poker-Engine adapter is the default as of P6 (2026-07-06). "legacy" remains
# selectable for the legacy regression suite; see the module docstring.
DEFAULT_ENGINE_IMPL = "pe"


def resolve_engine_impl(engine_impl: Optional[str] = None) -> str:
    """Resolve the effective engine implementation name.

    Priority: explicit argument > ``THAI_ENGINE_IMPL`` env var > default.
    """
    return (engine_impl or os.environ.get("THAI_ENGINE_IMPL")
            or DEFAULT_ENGINE_IMPL).lower()


def make_table(rng: Optional[random.Random] = None,
               engine_impl: Optional[str] = None):
    """Return a rules-engine table for the selected implementation.

    ``engine_impl`` is "pe" (default, Poker-Engine adapter) or "legacy". The
    legacy engine is imported only when actually requested, so nothing here
    forces its (or the sibling repo's) import cost on the other path.
    """
    impl = resolve_engine_impl(engine_impl)
    if impl == "legacy":
        from core.engine import Table
        return Table(rng=rng)
    if impl == "pe":
        from core.pe_engine import PokerEngineTable
        return PokerEngineTable(rng=rng)
    raise ValueError(
        f"unknown engine_impl {impl!r} (expected 'legacy' or 'pe')")
