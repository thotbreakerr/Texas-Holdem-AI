"""Selects the poker rules engine: the legacy in-repo engine or the
Poker-Engine-backed adapter.

Importing this module is cheap and pulls in NEITHER engine — both are imported
lazily inside :func:`make_table`, so the default ``legacy`` path never touches
the sibling Poker-Engine repo (or its import cost). Callers pick an engine by
passing ``engine_impl`` or setting the ``THAI_ENGINE_IMPL`` environment
variable; the default is ``legacy`` so behavior is unchanged until opted in.
"""

import os
import random
from typing import Optional

DEFAULT_ENGINE_IMPL = "legacy"


def resolve_engine_impl(engine_impl: Optional[str] = None) -> str:
    """Resolve the effective engine implementation name.

    Priority: explicit argument > ``THAI_ENGINE_IMPL`` env var > default.
    """
    return (engine_impl or os.environ.get("THAI_ENGINE_IMPL")
            or DEFAULT_ENGINE_IMPL).lower()


def make_table(rng: Optional[random.Random] = None,
               engine_impl: Optional[str] = None):
    """Return a rules-engine table for the selected implementation.

    ``engine_impl`` is "legacy" (default) or "pe". The Poker-Engine adapter is
    imported only when actually requested, so nothing here forces the sibling
    repo to be importable on the legacy path.
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
