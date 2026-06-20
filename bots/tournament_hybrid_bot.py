"""Deterministic tournament hybrid bot.

Phase 2 adds a real preflop strategy while preserving the Phase 1 safety
contract: all aggressive sizing goes through legal-action specs and every
decision is sanitized before it leaves ``act``.

Phase 3 adds a postflop module: Monte Carlo equity (``core.equity.equity``)
compared against pot odds, with per-profile thresholds, value-bet sizing, and
aggro semi-bluff / bluff lines.  Phase 4 adds tournament-specific stack-rank
context, small chipEV-first future-edge tax, short-stack desperation, chip-
leader pressure, and heads-up aggression.  Equity is seeded per decision so
play is reproducible, and every action still passes through the same legal-
action sanitizer.
"""

from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass
from typing import Any, Optional

from core.bot_api import Action, PlayerView
from core.equity import equity as _mc_equity


DecisionTrace = dict[str, Any]

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"
_RANK_INDEX = {rank: i for i, rank in enumerate(_RANKS)}
_POSITION_TAGS = {"BTN", "SB", "BB", "UTG", "UTG+1", "UTG1", "MP", "LJ", "HJ", "CO"}


def _hand_key(r1: str, r2: str, suited: bool) -> str:
    """Canonical hand key: high card first, suited/off-suit suffix for non-pairs."""
    i1, i2 = _RANK_INDEX[r1], _RANK_INDEX[r2]
    if i1 < i2:
        r1, r2 = r2, r1
    if r1 == r2:
        return f"{r1}{r2}"
    return f"{r1}{r2}{'s' if suited else 'o'}"


def _literal_hand(token: str) -> str:
    token = token.strip()
    if len(token) == 2 and token[0] == token[1]:
        return token
    if len(token) == 3 and token[2] in ("s", "o"):
        return _hand_key(token[0], token[1], token[2] == "s")
    raise ValueError(f"unsupported hand token: {token!r}")


def _expand_token(token: str) -> frozenset[str]:
    """Expand compact poker notation into canonical hand keys."""
    token = token.strip()
    if not token:
        return frozenset()
    if token.endswith("+"):
        base = token[:-1]
        if len(base) == 2 and base[0] == base[1]:
            start = _RANK_INDEX[base[0]]
            return frozenset(f"{rank}{rank}" for rank in _RANKS[start:])
        if len(base) == 3 and base[2] in ("s", "o"):
            high, low, suited = base[0], base[1], base[2] == "s"
            high_i = _RANK_INDEX[high]
            low_i = _RANK_INDEX[low]
            return frozenset(
                _hand_key(high, _RANKS[i], suited)
                for i in range(low_i, high_i)
            )
    if "-" in token:
        start, end = token.split("-", 1)
        if len(start) == len(end) == 2 and start[0] == start[1] and end[0] == end[1]:
            start_i = _RANK_INDEX[start[0]]
            end_i = _RANK_INDEX[end[0]]
            step = -1 if start_i >= end_i else 1
            return frozenset(
                f"{_RANKS[i]}{_RANKS[i]}"
                for i in range(start_i, end_i + step, step)
            )
        if len(start) == len(end) == 3 and start[2] == end[2] and start[2] in ("s", "o"):
            suited = start[2] == "s"
            if start[0] == end[0]:
                high = start[0]
                start_i = _RANK_INDEX[start[1]]
                end_i = _RANK_INDEX[end[1]]
                step = -1 if start_i >= end_i else 1
                return frozenset(
                    _hand_key(high, _RANKS[i], suited)
                    for i in range(start_i, end_i + step, step)
                )
            start_high = _RANK_INDEX[start[0]]
            end_high = _RANK_INDEX[end[0]]
            gap = start_high - _RANK_INDEX[start[1]]
            step = -1 if start_high >= end_high else 1
            hands = []
            for high_i in range(start_high, end_high + step, step):
                low_i = high_i - gap
                if 0 <= low_i < high_i:
                    hands.append(_hand_key(_RANKS[high_i], _RANKS[low_i], suited))
            return frozenset(hands)
    return frozenset({_literal_hand(token)})


def _hands(*tokens: str) -> frozenset[str]:
    result: set[str] = set()
    for token in tokens:
        result.update(_expand_token(token))
    return frozenset(result)


def _clamp(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return lo
    if f < lo:
        return lo
    if f > hi:
        return hi
    return f


_PREMIUM_3BET = _hands("QQ+", "AKs", "AKo")
_LATE_VALUE_3BET = _PREMIUM_3BET | _hands("JJ", "AQs")
_AGGRO_LATE_VALUE_3BET = _LATE_VALUE_3BET | _hands("TT", "AJs", "AQo", "KQs")

_SURVIVAL_RFI_EARLY = _hands(
    "55+", "ATs+", "KJs+", "QJs", "JTs", "AJo+", "KQo", "A5s-A4s"
)
_SURVIVAL_RFI_MIDDLE = _hands(
    "44+", "A7s+", "KTs+", "QTs+", "JTs", "T9s", "98s",
    "A9o+", "KJo+", "QJo"
)
_SURVIVAL_RFI_LATE = _hands(
    "22+", "A2s+", "K9s+", "Q9s+", "J9s+", "T8s+",
    "98s-65s", "A8o+", "KTo+", "QTo+", "JTo"
)
_SURVIVAL_RFI_SB = _hands(
    "22+", "A2s+", "K8s+", "Q9s+", "J9s+", "T8s+",
    "98s-54s", "A7o+", "KTo+", "QTo+", "JTo"
)

_AGGRO_RFI_EARLY = _SURVIVAL_RFI_EARLY
_AGGRO_RFI_MIDDLE = _SURVIVAL_RFI_MIDDLE | _hands(
    "22+", "A2s-A6s", "K9s", "Q9s", "J9s", "T8s", "A8o", "KTo", "QTo"
)
_AGGRO_RFI_LATE = _SURVIVAL_RFI_LATE | _hands(
    "K2s+", "Q5s+", "J8s+", "T7s+", "97s+", "86s+", "75s+",
    "64s+", "A2o+", "K8o+", "Q9o+", "J9o+", "T9o", "K8s"
)
_AGGRO_RFI_SB = _SURVIVAL_RFI_SB | _hands(
    "K2s+", "Q5s+", "J7s+", "T7s+", "96s+", "85s+", "74s+",
    "A2o+", "K7o+", "Q9o+", "J9o+", "T9o"
)

_RFI_RANGES = {
    "survival": {
        "early": _SURVIVAL_RFI_EARLY,
        "middle": _SURVIVAL_RFI_MIDDLE,
        "late": _SURVIVAL_RFI_LATE,
        "sb": _SURVIVAL_RFI_SB,
        "bb": _SURVIVAL_RFI_LATE,
        "heads_up": _SURVIVAL_RFI_SB,
    },
    "aggro": {
        "early": _AGGRO_RFI_EARLY,
        "middle": _AGGRO_RFI_MIDDLE,
        "late": _AGGRO_RFI_LATE,
        "sb": _AGGRO_RFI_SB,
        "bb": _AGGRO_RFI_LATE,
        "heads_up": _AGGRO_RFI_SB,
    },
}

_SHORT_SHOVE_SURVIVAL = {
    "early": _hands("44+", "A8s+", "ATo+", "KJs+", "KQo", "QJs"),
    "middle": _hands("33+", "A2s+", "A9o+", "KTs+", "KQo", "QTs+", "JTs"),
    "late": _hands(
        "22+", "A2s+", "A8o+", "K9s+", "KTo+", "Q9s+", "QTo+",
        "J9s+", "JTo", "T9s", "98s"
    ),
    "sb": _hands(
        "22+", "A2s+", "A2o+", "K8s+", "K9o+", "Q8s+", "QTo+",
        "J8s+", "JTo", "T8s+", "98s-76s"
    ),
    "bb": _hands(
        "22+", "A2s+", "A8o+", "K9s+", "KTo+", "Q9s+", "QTo+",
        "J9s+", "JTo", "T9s"
    ),
    "heads_up": _hands(
        "22+", "A2s+", "A2o+", "K7s+", "K9o+", "Q8s+", "QTo+",
        "J8s+", "JTo", "T8s+", "98s-76s"
    ),
}
_SHORT_SHOVE_AGGRO = {
    cat: hands | _hands("K6s+", "Q7s+", "J8s+", "T8s+", "A7o+", "K9o+")
    for cat, hands in _SHORT_SHOVE_SURVIVAL.items()
}
_SHORT_SHOVE_RANGES = {
    "survival": _SHORT_SHOVE_SURVIVAL,
    "aggro": _SHORT_SHOVE_AGGRO,
}

_CALL_SHOVE_SURVIVAL = {
    "early": _hands("TT+", "AQs+", "AQo+", "AKo"),
    "middle": _hands("77+", "AJs+", "AQo+", "KQs"),
    "late": _hands("44+", "A8s+", "ATo+", "KTs+", "KQo", "QJs"),
    "sb": _hands("44+", "A8s+", "ATo+", "KTs+", "KQo", "QJs"),
    "bb": _hands(
        "22+", "A2s+", "A8o+", "K8s+", "KTo+", "QTs+", "QJo",
        "JTs", "T9s"
    ),
    "heads_up": _hands(
        "22+", "A2s+", "A7o+", "K8s+", "KTo+", "QTs+", "QJo",
        "JTs", "T9s"
    ),
}
_CALL_SHOVE_AGGRO = {
    cat: hands | _hands("A9o", "K9s", "QTs", "JTs")
    for cat, hands in _CALL_SHOVE_SURVIVAL.items()
}
_CALL_SHOVE_RANGES = {
    "survival": _CALL_SHOVE_SURVIVAL,
    "aggro": _CALL_SHOVE_AGGRO,
}

_RESTEAL_SURVIVAL = {
    "early": _hands("TT+", "AQs+", "AKo"),
    "middle": _hands("77+", "ATs+", "AQo+", "KQs"),
    "late": _hands("55+", "A8s+", "ATo+", "KTs+", "KQo", "QJs", "JTs"),
    "sb": _hands("55+", "A8s+", "ATo+", "KTs+", "KQo", "QJs", "JTs"),
    "bb": _hands("44+", "A7s+", "A9o+", "KTs+", "KQo", "QJs", "JTs"),
    "heads_up": _hands("44+", "A2s+", "A8o+", "K9s+", "KTo+", "QTs+", "JTs"),
}
_RESTEAL_AGGRO = {
    cat: hands | _hands("44", "A5s-A2s", "K9s", "QTs", "T9s")
    for cat, hands in _RESTEAL_SURVIVAL.items()
}
_RESTEAL_RANGES = {
    "survival": _RESTEAL_SURVIVAL,
    "aggro": _RESTEAL_AGGRO,
}

_MEDIUM_OPEN_JAM = {
    "survival": {
        "late": _hands("22-66", "A2s-A5s", "A9s+", "ATo+", "KTs+", "QJs"),
        "sb": _hands("22-66", "A2s-A5s", "A8s+", "A9o+", "KTs+", "QJs", "JTs"),
        "heads_up": _hands("22-77", "A2s+", "A8o+", "KTs+", "QJs", "JTs"),
    },
    "aggro": {
        "late": _hands("22-77", "A2s-A5s", "A7s+", "A9o+", "K9s+", "KTo+", "QTs+", "JTs"),
        "sb": _hands("22-77", "A2s+", "A8o+", "K9s+", "KTo+", "QTs+", "JTs"),
        "heads_up": _hands("22-88", "A2s+", "A7o+", "K8s+", "KTo+", "QTs+", "JTs"),
    },
}

_VALUE_3BET_RANGES = {
    "survival": {
        "early": _PREMIUM_3BET,
        "middle": _PREMIUM_3BET,
        "late": _LATE_VALUE_3BET,
        "sb": _LATE_VALUE_3BET,
        "bb": _LATE_VALUE_3BET,
        "heads_up": _LATE_VALUE_3BET,
    },
    "aggro": {
        "early": _PREMIUM_3BET,
        "middle": _LATE_VALUE_3BET,
        "late": _AGGRO_LATE_VALUE_3BET,
        "sb": _AGGRO_LATE_VALUE_3BET,
        "bb": _AGGRO_LATE_VALUE_3BET,
        "heads_up": _AGGRO_LATE_VALUE_3BET,
    },
}

_FLAT_VS_OPEN_SURVIVAL = {
    "early": _hands("JJ-88", "AQs", "AJs", "KQs"),
    "middle": _hands("JJ-77", "AQs", "AJs", "ATs", "KQs", "KJs", "QJs", "JTs", "AQo", "KQo"),
    "late": _hands(
        "TT-22", "A2s+", "KTs+", "QTs+", "JTs", "T9s", "98s-65s",
        "AQo", "AJo", "KQo", "KJo", "QJo", "JTo"
    ),
    "sb": _hands("JJ-88", "AQs", "AJs", "KQs", "KJs", "QJs", "JTs"),
    "bb": _hands(
        "TT-22", "A2s+", "K8s+", "Q9s+", "J9s+", "T8s+", "98s-54s",
        "A8o+", "KTo+", "QTo+", "JTo"
    ),
    "heads_up": _hands(
        "TT-22", "A2s+", "K8s+", "Q9s+", "J9s+", "T8s+", "98s-54s",
        "A8o+", "KTo+", "QTo+", "JTo"
    ),
}
_FLAT_VS_OPEN_AGGRO = {
    cat: hands | _hands("A2o-A7o", "K9o", "Q9o", "J9o", "T9o")
    for cat, hands in _FLAT_VS_OPEN_SURVIVAL.items()
}
_FLAT_VS_OPEN_RANGES = {
    "survival": _FLAT_VS_OPEN_SURVIVAL,
    "aggro": _FLAT_VS_OPEN_AGGRO,
}

_VS_3BET_VALUE = _hands("QQ+", "AKs", "AKo")
_VS_3BET_FLAT = _hands("JJ", "AQs")
_VS_4BET_CORE = _hands("AA", "KK", "AKs")
_VS_4BET_CONDITIONAL = _hands("QQ", "AKo")

_ISO_VALUE = _hands("88+", "ATs+", "AJo+", "KQs", "KJs+", "QJs")
_ISO_VALUE_SHORT = _hands("TT+", "AQs+", "AQo+", "AKo")
_OVERLIMP_SPECULATIVE = _hands(
    "22-77", "A2s-A5s", "KTs+", "QTs+", "JTs", "T9s", "98s-65s"
)
_BB_DEFEND_LATE_SMALL = _hands(
    "22+", "A2s+", "K2s+", "Q6s+", "J7s+", "T7s+", "97s+",
    "86s+", "75s+", "64s+", "A2o+", "K9o+", "QTo+", "JTo", "T9o"
)


# ---------------------------------------------------------------------------
# Phase 4 tournament adjustment tables.
#
# The base BB band still chooses the mechanical mode.  These tables only nudge
# membership/frequencies by at most one tactical gear.  Widths are represented
# as fractions of the 169 canonical starting-hand keys, not combo-weighted.
# ---------------------------------------------------------------------------
_RANGE_DENOM = 169.0

_RANK_BUCKET_WEIGHT = {
    "chip_leader": 1.00,
    "top_2": 0.55,
    "middle": 0.0,
    "short_stack": 0.0,
    "hu_leader": 0.0,
    "hu_trailer": 0.0,
    "hu_even": 0.0,
}

_CHIPLEADER_PREFLOP = {
    "survival": {
        "early_open": (0.03, 0.010),
        "middle_open": (0.05, 0.020),
        "late_open": (0.08, 0.030),
        "btn_steal": (0.14, 0.050),
        "sb_steal": (0.16, 0.060),
        "iso": (0.10, 0.040),
        "threebet": (0.08, 0.020),
        "resteal": (0.12, 0.040),
        "open_jam": (0.08, 0.030),
    },
    "aggro": {
        "early_open": (0.05, 0.020),
        "middle_open": (0.08, 0.030),
        "late_open": (0.14, 0.050),
        "btn_steal": (0.22, 0.080),
        "sb_steal": (0.25, 0.090),
        "iso": (0.16, 0.060),
        "threebet": (0.14, 0.040),
        "resteal": (0.20, 0.070),
        "open_jam": (0.12, 0.050),
    },
}

_PRESSURE_RFI_EXPANSION = {
    "early": _hands("44-22", "A9s-A2s", "KTs", "QTs", "JTs", "AJo", "KQo"),
    "middle": _hands("33-22", "A6s-A2s", "K9s+", "Q9s+", "J9s+", "T9s", "A8o+", "KTo+", "QTo+"),
    "late": _hands(
        "A2s+", "K2s+", "Q5s+", "J7s+", "T7s+", "97s+", "86s+",
        "75s+", "64s+", "A2o+", "K8o+", "Q8o+", "J8o+", "T8o+"
    ),
    "sb": _hands(
        "A2s+", "K2s+", "Q4s+", "J6s+", "T6s+", "96s+", "85s+",
        "74s+", "63s+", "A2o+", "K7o+", "Q8o+", "J8o+", "T8o+"
    ),
    "bb": frozenset(),
    "heads_up": _hands(
        "A2s+", "K2s+", "Q4s+", "J6s+", "T6s+", "96s+", "85s+",
        "74s+", "A2o+", "K7o+", "Q8o+", "J8o+", "T8o+"
    ),
}

_PRESSURE_3BET_EXPANSION = {
    "early": _hands("JJ", "AQs", "AKo"),
    "middle": _hands("TT", "AJs+", "AQo+", "KQs"),
    "late": _hands("99+", "ATs+", "AJo+", "KJs+", "KQo", "QJs", "JTs"),
    "sb": _hands("88+", "A9s+", "ATo+", "KTs+", "KQo", "QTs+", "JTs"),
    "bb": _hands("77+", "A8s+", "ATo+", "KTs+", "KQo", "QTs+", "JTs", "T9s"),
    "heads_up": _hands("66+", "A2s+", "A8o+", "K9s+", "KTo+", "QTs+", "JTs", "T9s"),
}

_PRESSURE_ISO_EXPANSION = _hands(
    "77-22", "A2s+", "A9o+", "K9s+", "KTo+", "Q9s+", "QTo+",
    "J9s+", "JTo", "T9s", "98s-76s"
)

_SHORT_SHOVE_EXPANSION = {
    "early": _hands("22+", "A2s+", "A8o+", "KTs+", "KQo", "QTs+", "JTs"),
    "middle": _hands("22+", "A2s+", "A7o+", "K9s+", "KTo+", "Q9s+", "QTo+", "J9s+", "T9s"),
    "late": _hands(
        "22+", "A2s+", "A2o+", "K5s+", "K8o+", "Q7s+", "Q9o+",
        "J7s+", "J9o+", "T7s+", "T9o", "97s+", "86s+", "75s+"
    ),
    "sb": _hands(
        "22+", "A2s+", "A2o+", "K2s+", "K6o+", "Q5s+", "Q8o+",
        "J6s+", "J8o+", "T6s+", "T8o+", "96s+", "85s+", "74s+"
    ),
    "bb": _hands(
        "22+", "A2s+", "A2o+", "K5s+", "K8o+", "Q7s+", "Q9o+",
        "J7s+", "J9o+", "T7s+", "T9o", "97s+", "86s+"
    ),
    "heads_up": _hands(
        "22+", "A2s+", "A2o+", "K2s+", "K6o+", "Q5s+", "Q8o+",
        "J6s+", "J8o+", "T6s+", "T8o+", "96s+", "85s+", "74s+"
    ),
}

_SHORT_DESPERATION = {
    "survival": (
        (4.0, 0.22, 0.090),
        (6.0, 0.15, 0.060),
        (8.0, 0.09, 0.035),
        (10.0, 0.08, 0.030),
        (12.0, 0.05, 0.020),
        (16.0, 0.025, 0.010),
    ),
    "aggro": (
        (4.0, 0.32, 0.120),
        (6.0, 0.22, 0.080),
        (8.0, 0.14, 0.050),
        (10.0, 0.12, 0.040),
        (12.0, 0.08, 0.030),
        (16.0, 0.04, 0.015),
    ),
}

_SHOVE_CEILINGS = {
    "survival": {"early": 0.35, "middle": 0.42, "late": 0.75, "sb": 0.90, "bb": 0.75, "heads_up": 0.90},
    "aggro": {"early": 0.45, "middle": 0.52, "late": 0.90, "sb": 1.00, "bb": 0.90, "heads_up": 1.00},
}

_RANGE_GEAR_CAPS = {
    "survival": {"default": (0.25, 0.080), "threebet": (0.25, 0.050), "shove": (0.25, 0.080)},
    "aggro": {"default": (0.35, 0.120), "threebet": (0.40, 0.080), "shove": (0.35, 0.120)},
}

_CHIPLEADER_POSTFLOP_MULT = {
    "survival": {
        "hu": {"cbet": (1.08, 1.05, 1.00), "bluff": (1.12, 1.08, 1.04), "semi": (1.10, 1.08)},
        "3way": {"cbet": (1.04, 1.025, 1.00), "bluff": (1.06, 1.04, 1.02), "semi": (1.05, 1.04)},
        "4plus": {"cbet": (1.02, 1.01, 1.00), "bluff": (1.03, 1.02, 1.01), "semi": (1.02, 1.02)},
    },
    "aggro": {
        "hu": {"cbet": (1.15, 1.10, 1.03), "bluff": (1.25, 1.18, 1.10), "semi": (1.18, 1.14)},
        "3way": {"cbet": (1.075, 1.05, 1.015), "bluff": (1.125, 1.09, 1.05), "semi": (1.09, 1.07)},
        "4plus": {"cbet": (1.03, 1.02, 1.01), "bluff": (1.05, 1.04, 1.02), "semi": (1.04, 1.03)},
    },
}

_HU_POSTFLOP = {
    "survival": {
        "cbet": (1.12, 1.08, 1.03),
        "bluff": (1.18, 1.12, 1.08),
        "semi": (1.15, 1.12),
        "call_cushion": (-0.015, -0.020, -0.025),
        "value_bet": (-0.010, -0.010, -0.010),
        "value_raise": (-0.008, -0.008, -0.008),
    },
    "aggro": {
        "cbet": (1.20, 1.15, 1.08),
        "bluff": (1.30, 1.22, 1.15),
        "semi": (1.25, 1.18),
        "call_cushion": (-0.025, -0.030, -0.035),
        "value_bet": (-0.015, -0.015, -0.020),
        "value_raise": (-0.012, -0.012, -0.015),
    },
}

_FREQ_GEAR_CAPS = {
    "survival": {"mult": 1.25, "abs": 0.15},
    "aggro": {"mult": 1.40, "abs": 0.22},
}

_FREQ_CEILINGS = {
    "survival": {"cbet": 0.92, "bluff": 0.45, "semi": 0.70},
    "aggro": {"cbet": 0.96, "bluff": 0.55, "semi": 0.80},
}

_FUTURE_EDGE_TAX_CAPS = {
    "survival": {6: 0.020, 5: 0.015, 4: 0.008, 3: 0.003, 2: 0.007},
    "aggro": {6: 0.006, 5: 0.004, 4: 0.002, 3: 0.0, 2: 0.003},
}


# ---------------------------------------------------------------------------
# Phase 3 postflop strategy tables.
#
# Tournament-tuned defaults.  ``way3`` keys: "hu" (1 live opponent), "3way" (2),
# "4plus" (3+).
# ``way2`` keys (sizing): "hu" vs "mw" (multiway).  Street tuples are indexed
# flop=0, turn=1, river=2.  All equity/odds values are fractions in [0, 1].
# ---------------------------------------------------------------------------
_STREET_IDX = {"flop": 0, "turn": 1, "river": 2}

# Equity cushion required ABOVE raw pot odds before calling a bet.
_CALL_CUSHION = {
    "aggro": {
        "hu": (0.04, 0.03, 0.02),
        "3way": (0.07, 0.06, 0.05),
        "4plus": (0.10, 0.09, 0.08),
    },
    "survival": {
        "hu": (0.07, 0.06, 0.05),
        "3way": (0.11, 0.10, 0.09),
        "4plus": (0.15, 0.14, 0.13),
    },
}

# Equity needed to raise a bet (or build a large value line).  "4plus" adds +3pp.
_VALUE_RAISE = {
    "aggro": {"hu": (0.65, 0.67, 0.70), "mw": (0.72, 0.75, 0.78)},
    "survival": {"hu": (0.69, 0.71, 0.74), "mw": (0.76, 0.79, 0.82)},
}

# Equity needed to value-bet when checked to (to_call == 0).  "4plus" adds +3pp.
_VALUE_BET = {
    "aggro": {"hu": (0.57, 0.60, 0.56), "mw": (0.66, 0.68, 0.64)},
    "survival": {"hu": (0.61, 0.64, 0.60), "mw": (0.70, 0.72, 0.68)},
}

# Continuation-bet frequency when hero was the preflop aggressor and is checked to.
_CBET_FREQ = {
    "aggro": {
        "hu": (0.70, 0.45, 0.30),
        "3way": (0.45, 0.28, 0.18),
        "4plus": (0.30, 0.16, 0.08),
    },
    "survival": {
        "hu": (0.55, 0.30, 0.18),
        "3way": (0.30, 0.18, 0.10),
        "4plus": (0.20, 0.10, 0.05),
    },
}

# Pure-bluff frequency for non-value, non-semi-bluff hands when checked to.
_BLUFF_FREQ = {
    "aggro": {
        "hu": (0.18, 0.12, 0.22),
        "3way": (0.08, 0.05, 0.10),
        "4plus": (0.03, 0.02, 0.04),
    },
    "survival": {
        "hu": (0.06, 0.04, 0.06),
        "3way": (0.02, 0.01, 0.03),
        "4plus": (0.0, 0.0, 0.01),
    },
}

# Semi-bluff candidate equity band per way, flop (0) and turn (1) only.
_SEMI_BAND = {
    "hu": ((0.28, 0.52), (0.18, 0.40)),
    "3way": ((0.25, 0.48), (0.16, 0.35)),
    "4plus": ((0.22, 0.44), (0.14, 0.30)),
}

# Frequency of betting/raising semi-bluff candidates, flop (0) and turn (1).
_SEMI_FREQ = {
    "aggro": {"hu": (0.65, 0.50), "3way": (0.45, 0.32), "4plus": (0.25, 0.18)},
    "survival": {"hu": (0.35, 0.22), "3way": (0.22, 0.12), "4plus": (0.12, 0.06)},
}

# Bet sizing as a fraction of pot: profile -> category -> way2 -> (flop, turn, river).
_SIZING = {
    "aggro": {
        "value": {"hu": (0.55, 0.70, 0.75), "mw": (0.70, 0.85, 0.85)},
        "semibluff": {"hu": (0.50, 0.65, 0.65), "mw": (0.60, 0.75, 0.75)},
        "bluff": {"hu": (0.35, 0.55, 0.75), "mw": (0.45, 0.65, 0.85)},
    },
    "survival": {
        "value": {"hu": (0.45, 0.60, 0.65), "mw": (0.60, 0.75, 0.75)},
        "semibluff": {"hu": (0.40, 0.50, 0.50), "mw": (0.50, 0.60, 0.60)},
        "bluff": {"hu": (0.25, 0.35, 0.45), "mw": (0.25, 0.40, 0.50)},
    },
}


class OpponentProfiles:
    """Deterministic Phase 5 opponent action counters.

    The engine only exposes per-hand action prefixes, so ingest rebuilds the
    prefix every time and dedups only the counter increment by ``(hand_id, idx)``.
    """

    _ACTIONS = {"fold", "check", "call", "bet", "raise"}
    _PRESSURE_RESPONSES = {"fold", "call", "raise"}
    _POSTFLOP = {"flop", "turn", "river"}

    _DEFAULT_COUNTERS = {
        "preflop_action_seen": 0,
        "vpip_seen": 0,
        "preflop_raise_seen": 0,
        "postflop_bet_raise": 0,
        "postflop_call": 0,
        "postflop_check": 0,
        "pressure_fold": 0,
        "pressure_call": 0,
        "pressure_raise": 0,
        "postflop_pressure_fold": 0,
        "postflop_pressure_call": 0,
        "postflop_pressure_raise": 0,
        "large_bet_count": 0,
        "jam_like_count": 0,
        "short_jam_like_count": 0,
        "jam_opportunity_n": 0,
        "blind_steal_response_n": 0,
        "blind_fold_to_steal": 0,
        "fold_to_cbet_n": 0,
        "fold_to_cbet": 0,
        "threebet_count": 0,
        "fourbet_count": 0,
        "preflop_pressure_action_n": 0,
    }

    _PRIORS = {
        "vpip": 0.32,
        "preflop_aggression_rate": 0.16,
        "postflop_aggression_freq": 0.33,
        "fold_to_pressure": 0.45,
        "station_score": 0.45,
        "blind_fold_to_steal": 0.45,
        "fold_to_cbet": 0.45,
        "threebet_rate": 0.08,
        "fourbet_rate": 0.03,
    }

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.profiles: dict[Any, dict[str, int]] = {}
        self.seen: set[tuple[Any, int]] = set()
        self.once_flags: set[tuple[Any, Any, str]] = set()
        self.last_hand_id: int | None = None

    def ingest(self, view: PlayerView) -> None:
        hand_id = getattr(view, "hand_id", None)
        hand_int = self._maybe_int(hand_id)
        if hand_int is not None:
            if self.last_hand_id is not None and hand_int < self.last_hand_id:
                self.reset()
            self.last_hand_id = hand_int

        hero = getattr(view, "me", None)
        state = self._new_hand_state(view)
        all_in_pids = set(getattr(view, "all_in_opponents", None) or [])
        bb = self._infer_big_blind(view)
        history = list(getattr(view, "history", None) or [])
        for idx, entry in enumerate(history):
            if not isinstance(entry, dict):
                continue
            classified = self._classify_entry(entry, state, view, all_in_pids, bb)
            key = (hand_id, idx)
            if key not in self.seen:
                self.seen.add(key)
                pid = classified.get("pid")
                if pid is not None and pid != hero:
                    self._update_profile(hand_id, pid, classified)
            self._apply_entry(entry, state)

    def raw(self, pid: Any) -> dict[str, int]:
        return dict(self.profiles.get(pid, self._blank_profile()))

    def all_raw(self) -> dict[Any, dict[str, int]]:
        return {pid: dict(counters) for pid, counters in self.profiles.items()}

    def stat_summary(self, pid: Any, confidence_w: float = 10.0) -> dict[str, Any]:
        counters = self.profiles.get(pid, self._blank_profile())
        w = max(0.0, float(confidence_w or 0.0))

        preflop_n = counters["preflop_action_seen"]
        postflop_aggr_n = (
            counters["postflop_bet_raise"]
            + counters["postflop_call"]
            + counters["postflop_check"]
        )
        fold_pressure_n = (
            counters["pressure_fold"]
            + counters["pressure_call"]
            + counters["pressure_raise"]
        )
        station_n = (
            counters["postflop_pressure_call"]
            + counters["postflop_pressure_raise"]
        )
        preflop_pressure_n = counters["preflop_pressure_action_n"]

        summary = {
            "pid": pid,
            "preflop_action_seen": preflop_n,
            "vpip_seen": counters["vpip_seen"],
            "preflop_raise_seen": counters["preflop_raise_seen"],
            "postflop_aggression_n": postflop_aggr_n,
            "pressure_fold_n": fold_pressure_n,
            "station_response_n": station_n,
            "pressure_response_n": station_n,
            "blind_steal_response_n": counters["blind_steal_response_n"],
            "fold_to_cbet_n": counters["fold_to_cbet_n"],
            "preflop_pressure_action_n": preflop_pressure_n,
            "jam_opportunity_n": counters["jam_opportunity_n"],
            "jam_like_count": counters["jam_like_count"],
            "short_jam_like_count": counters["short_jam_like_count"],
            "large_bet_count": counters["large_bet_count"],
            "vpip_hat": self._p_hat(counters["vpip_seen"], preflop_n, self._PRIORS["vpip"], w),
            "preflop_aggression_rate_hat": self._p_hat(
                counters["preflop_raise_seen"],
                preflop_n,
                self._PRIORS["preflop_aggression_rate"],
                w,
            ),
            "postflop_aggression_freq_hat": self._p_hat(
                counters["postflop_bet_raise"],
                postflop_aggr_n,
                self._PRIORS["postflop_aggression_freq"],
                w,
            ),
            "fold_to_pressure_hat": self._p_hat(
                counters["pressure_fold"],
                fold_pressure_n,
                self._PRIORS["fold_to_pressure"],
                w,
            ),
            "station_score_hat": self._p_hat(
                counters["postflop_pressure_call"],
                station_n,
                self._PRIORS["station_score"],
                w,
            ),
            "blind_fold_to_steal_hat": self._p_hat(
                counters["blind_fold_to_steal"],
                counters["blind_steal_response_n"],
                self._PRIORS["blind_fold_to_steal"],
                w,
            ),
            "fold_to_cbet_hat": self._p_hat(
                counters["fold_to_cbet"],
                counters["fold_to_cbet_n"],
                self._PRIORS["fold_to_cbet"],
                w,
            ),
            "threebet_rate_hat": self._p_hat(
                counters["threebet_count"],
                preflop_pressure_n,
                self._PRIORS["threebet_rate"],
                w,
            ),
            "fourbet_rate_hat": self._p_hat(
                counters["fourbet_count"],
                preflop_pressure_n,
                self._PRIORS["fourbet_rate"],
                w,
            ),
        }
        return summary

    def read_strength(
        self,
        pid: Any,
        stat_name: str,
        *,
        threshold: float,
        band: float,
        confidence_w: float = 10.0,
    ) -> float:
        if not self._finite_positive(band):
            return 0.0
        summary = self.stat_summary(pid, confidence_w)
        sample_key = {
            "vpip": "preflop_action_seen",
            "preflop_aggression_rate": "preflop_action_seen",
            "postflop_aggression_freq": "postflop_aggression_n",
            "fold_to_pressure": "pressure_fold_n",
            "station_score": "station_response_n",
            "blind_fold_to_steal": "blind_steal_response_n",
            "fold_to_cbet": "fold_to_cbet_n",
            "threebet_rate": "preflop_pressure_action_n",
            "fourbet_rate": "preflop_pressure_action_n",
        }.get(stat_name)
        if sample_key and self._safe_int(summary.get(sample_key)) <= 0:
            return 0.0
        p_hat = summary.get(f"{stat_name}_hat")
        if not self._is_finite(p_hat) or not self._is_finite(threshold):
            return 0.0
        return max(0.0, min(1.0, (float(p_hat) - float(threshold)) / float(band)))

    @classmethod
    def _blank_profile(cls) -> dict[str, int]:
        return dict(cls._DEFAULT_COUNTERS)

    def _profile_for(self, pid: Any) -> dict[str, int]:
        if pid not in self.profiles:
            self.profiles[pid] = self._blank_profile()
        return self.profiles[pid]

    def _bump_once(self, hand_id: Any, pid: Any, flag: str, counters: dict[str, int], key: str) -> None:
        once = (hand_id, pid, flag)
        if once in self.once_flags:
            return
        self.once_flags.add(once)
        counters[key] += 1

    def _update_profile(self, hand_id: Any, pid: Any, data: dict[str, Any]) -> None:
        counters = self._profile_for(pid)
        if data.get("preflop_action"):
            self._bump_once(hand_id, pid, "preflop_action", counters, "preflop_action_seen")
        if data.get("vpip"):
            self._bump_once(hand_id, pid, "vpip", counters, "vpip_seen")
        if data.get("preflop_raise"):
            self._bump_once(hand_id, pid, "preflop_raise", counters, "preflop_raise_seen")

        if data.get("postflop_bet_raise"):
            counters["postflop_bet_raise"] += 1
            counters["jam_opportunity_n"] += 1
            if data.get("jam_like"):
                counters["jam_like_count"] += 1
            elif data.get("short_jam_like"):
                counters["short_jam_like_count"] += 1
            elif data.get("large_bet"):
                counters["large_bet_count"] += 1
        if data.get("postflop_call"):
            counters["postflop_call"] += 1
        if data.get("postflop_check"):
            counters["postflop_check"] += 1

        if data.get("facing_pressure"):
            action_type = data.get("type")
            if action_type in self._PRESSURE_RESPONSES:
                counters[f"pressure_{action_type}"] += 1
                if data.get("street") in self._POSTFLOP:
                    counters[f"postflop_pressure_{action_type}"] += 1
                if data.get("street") == "preflop":
                    counters["preflop_pressure_action_n"] += 1

        if data.get("blind_steal_response"):
            counters["blind_steal_response_n"] += 1
            if data.get("type") == "fold":
                counters["blind_fold_to_steal"] += 1

        if data.get("cbet_response"):
            counters["fold_to_cbet_n"] += 1
            if data.get("type") == "fold":
                counters["fold_to_cbet"] += 1

        if data.get("threebet"):
            counters["threebet_count"] += 1
        if data.get("fourbet"):
            counters["fourbet_count"] += 1

    def _new_hand_state(self, view: PlayerView) -> dict[str, Any]:
        return {
            "contrib": {},
            "preflop_raise_count": 0,
            "preflop_raiser_pid": None,
            "last_preflop_raiser_pid": None,
            "first_preflop_raiser_steal": False,
            "postflop_first_aggressor_pid": None,
            "postflop_first_aggressor_is_pfr": False,
            "current_postflop_aggressor_pid": None,
            "seat_indices": getattr(view, "seat_indices", None) or {},
        }

    def _classify_entry(
        self,
        entry: dict[str, Any],
        state: dict[str, Any],
        view: PlayerView,
        all_in_pids: set[Any],
        bb: int,
    ) -> dict[str, Any]:
        street = str(entry.get("street") or "")
        action_type = str(entry.get("type") or "")
        pid = entry.get("pid")
        to_call_before = self._safe_int(entry.get("to_call_before"))
        amount = self._safe_int(entry.get("amount"))
        pot_before = self._safe_int(entry.get("pot_before"))
        prior_contrib = self._street_contrib(state, street).get(pid, 0)
        incremental = self._incremental_amount(action_type, amount, prior_contrib)

        preflop = street == "preflop"
        postflop = street in self._POSTFLOP
        response = action_type in self._PRESSURE_RESPONSES
        prior_preflop_raise = state["preflop_raise_count"] > 0
        facing_pressure = response and to_call_before > 0 and (
            (postflop) or (preflop and prior_preflop_raise)
        )
        postflop_bet_raise = postflop and action_type in {"bet", "raise"}
        jam_like = False
        short_jam_like = False
        if postflop_bet_raise and pid in all_in_pids:
            risk_bb = incremental / max(1, bb)
            if risk_bb < 12.0:
                short_jam_like = True
            else:
                jam_like = True

        large_bet = (
            postflop_bet_raise
            and not jam_like
            and not short_jam_like
            and incremental >= max(8 * max(1, bb), int(round(0.75 * max(1, pot_before))))
        )

        is_steal_response = (
            preflop
            and facing_pressure
            and state.get("first_preflop_raiser_steal")
            and action_type in self._PRESSURE_RESPONSES
        )
        cbet_response = (
            postflop
            and facing_pressure
            and state.get("postflop_first_aggressor_is_pfr")
            and action_type in self._PRESSURE_RESPONSES
        )

        return {
            "pid": pid,
            "street": street,
            "type": action_type,
            "preflop_action": preflop and action_type in self._ACTIONS,
            "vpip": preflop and (action_type == "raise" or (action_type == "call" and to_call_before > 0)),
            "preflop_raise": preflop and action_type == "raise",
            "facing_pressure": facing_pressure,
            "postflop_bet_raise": postflop_bet_raise,
            "postflop_call": postflop and action_type == "call",
            "postflop_check": postflop and action_type == "check",
            "large_bet": large_bet,
            "jam_like": jam_like,
            "short_jam_like": short_jam_like,
            "blind_steal_response": is_steal_response,
            "cbet_response": cbet_response,
            "threebet": preflop and action_type == "raise" and state["preflop_raise_count"] >= 1,
            "fourbet": preflop and action_type == "raise" and state["preflop_raise_count"] >= 2,
        }

    def _apply_entry(self, entry: dict[str, Any], state: dict[str, Any]) -> None:
        street = str(entry.get("street") or "")
        action_type = str(entry.get("type") or "")
        pid = entry.get("pid")
        if pid is None or action_type not in self._ACTIONS:
            return
        amount = self._safe_int(entry.get("amount"))
        contrib = self._street_contrib(state, street)
        prior = contrib.get(pid, 0)
        if action_type == "call":
            contrib[pid] = prior + max(0, amount)
        elif action_type in {"bet", "raise"}:
            contrib[pid] = max(prior, amount)

        if street == "preflop" and action_type == "raise":
            state["preflop_raise_count"] += 1
            if state["preflop_raiser_pid"] is None:
                state["preflop_raiser_pid"] = pid
                state["first_preflop_raiser_steal"] = self._is_steal_position(state, pid)
            state["last_preflop_raiser_pid"] = pid

        if street in self._POSTFLOP and action_type in {"bet", "raise"}:
            if state["postflop_first_aggressor_pid"] is None:
                state["postflop_first_aggressor_pid"] = pid
                state["postflop_first_aggressor_is_pfr"] = pid == state.get("last_preflop_raiser_pid")
            state["current_postflop_aggressor_pid"] = pid

    def _street_contrib(self, state: dict[str, Any], street: str) -> dict[Any, int]:
        return state["contrib"].setdefault(street, {})

    @staticmethod
    def _incremental_amount(action_type: str, amount: int, prior_contrib: int) -> int:
        if action_type == "call":
            return max(0, amount)
        if action_type in {"bet", "raise"}:
            return max(0, amount - max(0, prior_contrib))
        return 0

    @staticmethod
    def _is_steal_position(state: dict[str, Any], pid: Any) -> bool:
        seats = state.get("seat_indices") or {}
        if not isinstance(seats, dict) or pid not in seats or len(seats) < 2:
            return False
        # Six-max order in this codebase is BTN, SB, BB, UTG, UTG+1, MP.  A
        # late-position steal sample is enough for log-only telemetry.
        try:
            idx = int(seats.get(pid))
        except (TypeError, ValueError):
            return False
        return idx in {0, max(0, len(seats) - 1)}

    @classmethod
    def _p_hat(cls, k: int, n: int, p0: float, w: float) -> float:
        k = max(0, cls._safe_int(k))
        n = max(0, cls._safe_int(n))
        if not cls._is_finite(p0) or not cls._is_finite(w):
            return 0.0
        p0 = max(0.0, min(1.0, float(p0)))
        w = max(0.0, float(w))
        denom = n + w
        if denom <= 0.0:
            return p0
        value = (k + p0 * w) / denom
        return value if cls._is_finite(value) else 0.0

    @classmethod
    def _infer_big_blind(cls, view: PlayerView) -> int:
        min_raise = cls._safe_int(getattr(view, "min_raise", 0))
        if min_raise > 0:
            return max(1, min_raise)
        pot = cls._safe_int(getattr(view, "pot", 0))
        if pot > 0:
            return max(1, int(round(pot / 1.5)))
        for entry in getattr(view, "history", None) or []:
            if not isinstance(entry, dict):
                continue
            to_call = cls._safe_int(entry.get("to_call_before"))
            if to_call > 0:
                return max(1, to_call)
        return 10

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _maybe_int(cls, value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_finite(value: Any) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @classmethod
    def _finite_positive(cls, value: Any) -> bool:
        return cls._is_finite(value) and float(value) > 0.0


@dataclass(frozen=True)
class ProfileConfig:
    name: str

    # Preflop module knobs.
    preflop_range_key: Optional[str] = None
    preflop_open_adjustment: float = 0.0
    preflop_call_adjustment: float = 0.0

    # Postflop module stubs.  Phase 3 will consume equity and draw margins.
    postflop_equity_margin: float = 0.0
    postflop_draw_bias: float = 0.0
    postflop_probe_frequency: Optional[float] = None

    # Sizing module stubs.  Later phases will translate intent into totals.
    value_bet_fraction: Optional[float] = None
    bluff_bet_fraction: Optional[float] = None
    max_risk_fraction: Optional[float] = None

    # Tournament module knobs.
    survival_pressure: float = 0.0
    short_stack_bb_thresholds: Optional[dict[str, int]] = None
    short_stack_bb_threshold: Optional[int] = None
    medium_band_top_bb: int = 25
    bubble_factor: Optional[float] = None
    future_edge_tax_scale: float = 1.0
    chipleader_pressure_scale: float = 1.0
    desperation_scale: float = 1.0
    hu_aggression_scale: float = 1.0

    # Opponent module stubs.  Phase 5 will add memory and exploitation hooks.
    opponent_model_key: Optional[str] = None
    exploit_adjustment: float = 0.0
    memory_hands: Optional[int] = None
    p5_enabled: bool = False
    p5_log_only: bool = False
    p5_station_enabled: bool = True
    p5_r2_enabled: bool = True
    station_suppression_scale: float = 0.55
    spewy_tax_relief: float = 0.50
    p5_confidence_w: float = 10.0


SURVIVAL_CONFIG = ProfileConfig(
    name="survival",
    max_risk_fraction=0.35,
    survival_pressure=1.0,
    short_stack_bb_thresholds={
        "early": 10,
        "middle": 11,
        "late": 12,
        "sb": 14,
        "bb": 12,
        "heads_up": 15,
    },
    medium_band_top_bb=25,
    future_edge_tax_scale=1.0,
    chipleader_pressure_scale=0.60,
    desperation_scale=1.0,
    hu_aggression_scale=1.0,
)

AGGRO_CONFIG = ProfileConfig(
    name="aggro",
    preflop_open_adjustment=0.10,
    postflop_probe_frequency=0.10,
    max_risk_fraction=0.50,
    survival_pressure=0.5,
    short_stack_bb_thresholds={
        "early": 9,
        "middle": 10,
        "late": 11,
        "sb": 13,
        "bb": 11,
        "heads_up": 14,
    },
    medium_band_top_bb=25,
    exploit_adjustment=0.10,
    future_edge_tax_scale=0.70,
    chipleader_pressure_scale=1.0,
    desperation_scale=1.15,
    hu_aggression_scale=1.10,
)

_PROFILE_CONFIGS = {
    "survival": SURVIVAL_CONFIG,
    "aggro": AGGRO_CONFIG,
}


class TournamentHybridBot:
    """Legal-by-construction tournament bot with deterministic preflop play."""

    def __init__(self, profile: str = "survival"):
        key = str(profile).strip().lower()
        if key not in _PROFILE_CONFIGS:
            names = ", ".join(sorted(_PROFILE_CONFIGS))
            raise ValueError(f"unknown TournamentHybridBot profile {profile!r}; "
                             f"expected one of: {names}")
        self.config = _PROFILE_CONFIGS[key]
        self.profile = self.config.name
        self.last_decision: DecisionTrace | None = None
        self._bb_cache: dict[Any, int] = {}
        self._rank_cache: dict[Any, DecisionTrace] = {}
        self._pending_preflop_trace: DecisionTrace = {}
        self._profiles = OpponentProfiles()
        self.p5_enabled = bool(self.config.p5_enabled)
        self.p5_log_only = bool(self.config.p5_log_only)
        self.p5_station_enabled = bool(self.config.p5_station_enabled)
        self.p5_r2_enabled = bool(self.config.p5_r2_enabled)
        self.station_suppression_scale = float(self.config.station_suppression_scale)
        self.spewy_tax_relief = float(self.config.spewy_tax_relief)
        self.p5_confidence_w = float(self.config.p5_confidence_w)
        self.p5_error_count = 0
        self.p5_decision_count = 0
        self.p5_station_read_fire_count = 0
        self.p5_r2_read_fire_count = 0
        self._p5_trace_updates: DecisionTrace = {}

        # Adaptive Monte Carlo budget for postflop equity.  Defaults are sized
        # for fast iterated evaluation; raise ``postflop_sim_cap`` (and the
        # big/all-in tiers) on match day where heavier compute is affordable.
        self.postflop_base_sims = 200
        self.postflop_big_sims = 1200
        self.postflop_allin_sims = 2500
        self.postflop_sim_cap = 2500

        # A/B override for the future-edge tax (Phase 4 plan §12 R7). When set it
        # is a hard ceiling on the per-decision tax cap; 0.0 disables the tax
        # entirely so eval can isolate whether the tax earns its keep against a
        # pure-chipEV arm.  ``None`` keeps the built-in per-player caps.
        self.future_edge_tax_cap_override: float | None = None

    def reset_memory(self):
        """Tournament boundary hook for cached state and Phase 5 profiles."""
        self.last_decision = None
        self._bb_cache.clear()
        self._rank_cache.clear()
        self._pending_preflop_trace = {}
        self._profiles.reset()
        self.p5_error_count = 0
        self.p5_decision_count = 0
        self.p5_station_read_fire_count = 0
        self.p5_r2_read_fire_count = 0
        self._p5_trace_updates = {}
        return None

    def act(self, view: PlayerView) -> Action:
        """Return a safe action and populate ``last_decision``."""
        self._p5_trace_updates = self._p5_base_trace()
        try:
            self._p5_ingest(view)
        except Exception as exc:
            self._p5_record_error("ingest", exc)

        context: dict[str, Any]
        context_failed = False
        try:
            context = self._context(view)
        except Exception as exc:  # Defensive boundary for malformed views.
            context_failed = True
            context = {
                "players_remaining": 1,
                "position": getattr(view, "position", None),
                "effective_stack": 0,
                "context_error": str(exc),
            }

        legal_types = self._legal_types(view)
        proposed: Action | None = None
        path = "passive"
        reason = "postflop passive default"
        self._pending_preflop_trace = self._empty_preflop_trace()

        if getattr(view, "street", None) == "preflop":
            if context_failed:
                proposed = self._fold_or_check(view)
                self._pending_preflop_trace.update({
                    "branch": "context_error",
                    "reason": context.get("context_error", "context error"),
                })
            else:
                try:
                    proposed = self._preflop_action(view, context)
                except Exception as exc:
                    proposed = self._fold_or_check(view)
                    self._pending_preflop_trace.update({
                        "branch": "preflop_exception",
                        "reason": str(exc),
                    })
            if proposed is not None:
                path = "preflop"
                reason = self._pending_preflop_trace.get("reason", "preflop strategy")
            else:
                reason = self._pending_preflop_trace.get("reason", "preflop passive fallback")
        elif getattr(view, "street", None) in _STREET_IDX and not context_failed:
            try:
                proposed = self._postflop_action(view, context)
            except Exception as exc:  # Defensive boundary mirrors preflop.
                proposed = self._fold_or_check(view)
                self._pending_preflop_trace.update({
                    "branch": "postflop_exception",
                    "reason": str(exc),
                })
            if proposed is not None:
                path = "postflop"
                reason = self._pending_preflop_trace.get("reason", "postflop strategy")

        if proposed is None:
            proposed = self._fold_or_check(view) if context_failed else self._safe_passive(view)

        action = self._sanitize(view, proposed)
        if legal_types and action.type not in legal_types:
            path = "fallback"
            reason = "degenerate legal_actions had no usable proposed action"
        elif action.type != proposed.type or action.amount != proposed.amount:
            path = "fallback"
            reason = "final sanitizer adjusted proposed action"

        trace = dict(self._pending_preflop_trace)
        trace.update({
            "profile": self.config.name,
            "legal_types": tuple(sorted(legal_types)),
            "path": path,
            "target_total": trace.get("target_total"),
            "clamped_amount": action.amount,
            "all_in_detected": self._action_is_all_in(view, action),
            "reason": reason,
            "context": context,
        })
        self._p5_trace_log_only_reads(view)
        self._p5_trace_updates["p5_error_count"] = self.p5_error_count
        trace.update(self._p5_trace_updates)
        self._p5_update_telemetry(trace)
        self.last_decision = trace
        return action

    @staticmethod
    def _empty_preflop_trace() -> DecisionTrace:
        return {
            "position_category": None,
            "inferred_bb": None,
            "eff_bb": None,
            "hero_bb": None,
            "band": None,
            "raises_faced": None,
            "branch": None,
            "range_hit": False,
            "base_range_hit": False,
            "spot_type": None,
            "pure_chipEV_action": None,
            "phase4_action": None,
            "future_edge_tax": 0.0,
        }

    def _trace_preflop(self, **updates: Any) -> None:
        self._pending_preflop_trace.update(updates)

    # ------------------------------------------------------------------
    # Phase 5: opponent tendencies substrate and guarded exploits.
    # ------------------------------------------------------------------
    def _p5_base_trace(self) -> DecisionTrace:
        return {
            "p5_enabled": bool(self.p5_enabled),
            "p5_log_only": bool(self.p5_log_only),
            "p5_active": self._p5_active(),
            "p5_station_enabled": bool(getattr(self, "p5_station_enabled", True)),
            "p5_r2_enabled": bool(getattr(self, "p5_r2_enabled", True)),
            "p5_error_count": self.p5_error_count,
            "p5_station_applied": False,
            "p5_station_pid": None,
            "p5_station_strength": 0.0,
            "p5_station_delta": 0.0,
            "p5_r2_relief_applied": False,
            "p5_r2_confirmed_spewy": False,
            "p5_r2_aggressor": None,
        }

    def _p5_ingest(self, view: PlayerView) -> None:
        self._profiles.ingest(view)

    def _p5_record_error(self, label: str, exc: Exception | str) -> None:
        self.p5_error_count += 1
        self._p5_trace_updates.update({
            "p5_error_count": self.p5_error_count,
            "p5_last_error": f"{label}: {exc}",
        })

    def _p5_active(self) -> bool:
        return bool(getattr(self, "p5_enabled", False)) and not bool(getattr(self, "p5_log_only", False))

    def _p5_confidence_w(self) -> float:
        try:
            w = float(getattr(self, "p5_confidence_w", self.config.p5_confidence_w))
        except (TypeError, ValueError):
            self._p5_record_error("confidence_w", "non-finite confidence weight")
            return 10.0
        if not math.isfinite(w) or w < 0.0:
            self._p5_record_error("confidence_w", "non-finite confidence weight")
            return 10.0
        return w

    def _p5_sorted_pids(self, view: PlayerView, pids: list[Any] | tuple[Any, ...] | None = None) -> list[Any]:
        if pids is None:
            pids = list(getattr(view, "acting_opponents", None) or getattr(view, "opponents", None) or [])
        seats = getattr(view, "seat_indices", None) or {}

        def sort_key(pid: Any) -> tuple[int, str]:
            try:
                seat = int(seats.get(pid))
            except (AttributeError, TypeError, ValueError):
                seat = 10**9
            return (seat, str(pid))

        return sorted([pid for pid in pids if pid is not None and pid != getattr(view, "me", None)], key=sort_key)

    @staticmethod
    def _p5_finite(value: Any) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @classmethod
    def _p5_clamp(cls, value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
        if not cls._p5_finite(value):
            return lo
        f = float(value)
        if f < lo:
            return lo
        if f > hi:
            return hi
        return f

    def _p5_read_strength(
        self,
        pid: Any,
        stat_name: str,
        *,
        threshold: float,
        band: float,
    ) -> float:
        if not self._p5_finite(band) or float(band) <= 0.0:
            self._p5_record_error(f"{stat_name}_band", "non-positive band")
            return 0.0
        strength = self._profiles.read_strength(
            pid,
            stat_name,
            threshold=threshold,
            band=band,
            confidence_w=self._p5_confidence_w(),
        )
        if not self._p5_finite(strength):
            self._p5_record_error(f"{stat_name}_strength", "non-finite strength")
            return 0.0
        return self._p5_clamp(strength)

    def _p5_station_read(self, view: PlayerView) -> dict[str, Any]:
        best: dict[str, Any] = {
            "pid": None,
            "strength": 0.0,
            "station_score_hat": OpponentProfiles._PRIORS["station_score"],
            "pressure_response_n": 0,
            "guard": False,
        }
        for pid in self._p5_sorted_pids(view):
            stats = self._profiles.stat_summary(pid, self._p5_confidence_w())
            n = self._safe_int(stats.get("pressure_response_n"))
            strength = self._p5_read_strength(pid, "station_score", threshold=0.60, band=0.25)
            guard = n >= 4 and strength > 0.0
            score = float(stats.get("station_score_hat", 0.0) or 0.0)
            if guard and (strength, score, str(pid)) > (
                best["strength"],
                float(best.get("station_score_hat", 0.0) or 0.0),
                str(best.get("pid")),
            ):
                best = {
                    "pid": pid,
                    "strength": strength,
                    "station_score_hat": score,
                    "pressure_response_n": n,
                    "guard": True,
                }
        return best

    def _p5_station_frequency(self, view: PlayerView, knob_name: str, freq: float) -> float:
        try:
            read = self._p5_station_read(view)
            strength = float(read.get("strength", 0.0) or 0.0)
            scale = self._p5_clamp(getattr(self, "station_suppression_scale", 0.55), 0.0, 1.0)
            if knob_name == "cbet":
                target_scale = 1.0 - (1.0 - scale) * 0.50
            elif knob_name == "bluff":
                target_scale = scale
            else:
                target_scale = 1.0
            mult = 1.0 - (1.0 - target_scale) * strength
            proposed = self._p5_clamp(float(freq) * mult, 0.0, 1.0)
            delta = proposed - float(freq)
            applied = (
                bool(read.get("guard"))
                and self._p5_active()
                and bool(getattr(self, "p5_station_enabled", True))
                and knob_name in {"cbet", "bluff"}
            )
            would_apply = bool(read.get("guard")) and knob_name in {"cbet", "bluff"}
            self._p5_trace_updates.update({
                "p5_station_pid": read.get("pid"),
                "p5_station_score_hat": round(float(read.get("station_score_hat", 0.0) or 0.0), 4),
                "p5_station_pressure_response_n": self._safe_int(read.get("pressure_response_n")),
                "p5_station_strength": round(strength, 4),
                "p5_station_knob": knob_name,
                "p5_station_scale": round(target_scale, 4),
                "p5_station_delta": round(delta, 4),
                "p5_station_proposed_freq": round(proposed, 4),
                "p5_station_would_apply": would_apply,
                "p5_station_applied": applied,
            })
            return proposed if applied else freq
        except Exception as exc:
            self._p5_record_error("station_suppression", exc)
            return freq

    def _p5_current_postflop_aggressor(self, view: PlayerView) -> Any:
        aggressor = None
        for entry in getattr(view, "history", None) or []:
            if (
                isinstance(entry, dict)
                and entry.get("street") in _STREET_IDX
                and entry.get("type") in {"bet", "raise"}
            ):
                aggressor = entry.get("pid")
        return aggressor

    def _p5_confirmed_spewy(self, pid: Any) -> tuple[bool, DecisionTrace]:
        stats = self._profiles.stat_summary(pid, self._p5_confidence_w())
        vpip_ok = float(stats.get("vpip_hat", 0.0) or 0.0) >= 0.36
        pfr_ok = float(stats.get("preflop_aggression_rate_hat", 0.0) or 0.0) >= 0.18
        post_ok = float(stats.get("postflop_aggression_freq_hat", 0.0) or 0.0) >= 0.38
        looseness = bool(vpip_ok and pfr_ok and post_ok)
        jam_n = self._safe_int(stats.get("jam_opportunity_n"))
        jam_count = self._safe_int(stats.get("jam_like_count"))
        large_count = self._safe_int(stats.get("large_bet_count"))
        short_jam = self._safe_int(stats.get("short_jam_like_count"))
        confirmed = (
            jam_n >= 3
            and looseness
            and (jam_count >= 2 or large_count >= 3)
            and short_jam == 0
        )
        trace = {
            "p5_r2_vpip_hat": round(float(stats.get("vpip_hat", 0.0) or 0.0), 4),
            "p5_r2_preflop_aggr_hat": round(float(stats.get("preflop_aggression_rate_hat", 0.0) or 0.0), 4),
            "p5_r2_postflop_aggr_hat": round(float(stats.get("postflop_aggression_freq_hat", 0.0) or 0.0), 4),
            "p5_r2_looseness_conjunction": looseness,
            "p5_r2_jam_opportunity_n": jam_n,
            "p5_r2_jam_like_count": jam_count,
            "p5_r2_large_bet_count": large_count,
            "p5_r2_short_jam_like_count": short_jam,
            "p5_r2_confirmed_spewy": confirmed,
        }
        return confirmed, trace

    def _p5_relieve_future_tax(
        self,
        view: PlayerView,
        tax: float,
        *,
        eq: float,
        pot_odds: float,
        all_in_call: bool,
    ) -> float:
        try:
            if tax <= 0.0 or not all_in_call or getattr(view, "street", None) not in _STREET_IDX:
                return tax
            aggressor = self._p5_current_postflop_aggressor(view)
            all_in = set(getattr(view, "all_in_opponents", None) or [])
            live_villains = self._p5_sorted_pids(
                view,
                [
                    pid for pid in (getattr(view, "opponents", None) or [])
                    if pid != getattr(view, "me", None)
                ],
            )
            heads_up_shover = len(live_villains) == 1 and aggressor in live_villains and aggressor in all_in
            chip_ev_ok = self._p5_finite(eq) and self._p5_finite(pot_odds) and float(eq) >= float(pot_odds)
            confirmed = False
            spewy_trace: DecisionTrace = {}
            if aggressor is not None:
                confirmed, spewy_trace = self._p5_confirmed_spewy(aggressor)
            relief = self._p5_clamp(getattr(self, "spewy_tax_relief", 0.50), 0.0, 1.0)
            would_apply = bool(heads_up_shover and chip_ev_ok and confirmed)
            applied = bool(
                heads_up_shover
                and chip_ev_ok
                and confirmed
                and self._p5_active()
                and bool(getattr(self, "p5_r2_enabled", True))
            )
            proposed_tax = round(max(0.0, float(tax) * (1.0 - relief)), 4)
            self._p5_trace_updates.update({
                "p5_r2_aggressor": aggressor,
                "p5_r2_heads_up_shover": heads_up_shover,
                "p5_r2_chip_ev_ok": bool(chip_ev_ok),
                "p5_r2_relief": round(relief, 4),
                "p5_r2_tax_before": round(float(tax), 4),
                "p5_r2_tax_after_proposed": proposed_tax,
                "p5_r2_would_apply": would_apply,
                "p5_r2_relief_applied": applied,
                **spewy_trace,
            })
            return proposed_tax if applied else tax
        except Exception as exc:
            self._p5_record_error("r2_tax_relief", exc)
            return tax

    def _p5_trace_log_only_reads(self, view: PlayerView) -> None:
        try:
            pids = self._p5_sorted_pids(view)
            if not pids:
                self._p5_trace_updates.update({
                    "p5_overfolder_delta_log_only": 0.0,
                    "p5_passive_tight_delta_log_only": 0.0,
                    "p5_station_value_delta_log_only": 0.0,
                    "p5_aggro_second_gear_delta_log_only": 0.0,
                })
                return

            best_over = (0.0, None)
            best_passive = (0.0, None)
            best_value = (0.0, None)
            best_aggro = (0.0, None)
            for pid in pids:
                over = 0.15 * self._p5_read_strength(pid, "fold_to_pressure", threshold=0.58, band=0.17)
                station = 0.020 * self._p5_read_strength(pid, "station_score", threshold=0.60, band=0.25)
                aggro = 0.12 * self._p5_read_strength(pid, "postflop_aggression_freq", threshold=0.52, band=0.20)
                stats = self._profiles.stat_summary(pid, self._p5_confidence_w())
                vpip_hat = float(stats.get("vpip_hat", 0.0) or 0.0)
                post_hat = float(stats.get("postflop_aggression_freq_hat", 0.0) or 0.0)
                passive = 0.010 if (
                    stats.get("preflop_action_seen", 0) > 0
                    and stats.get("postflop_aggression_n", 0) > 0
                    and vpip_hat <= 0.24
                    and post_hat <= 0.24
                ) else 0.0
                if over > best_over[0]:
                    best_over = (over, pid)
                if passive > best_passive[0]:
                    best_passive = (passive, pid)
                if station > best_value[0]:
                    best_value = (station, pid)
                if aggro > best_aggro[0]:
                    best_aggro = (aggro, pid)

            self._p5_trace_updates.update({
                "p5_overfolder_pid": best_over[1],
                "p5_overfolder_delta_log_only": round(best_over[0], 4),
                "p5_passive_tight_pid": best_passive[1],
                "p5_passive_tight_delta_log_only": round(best_passive[0], 4),
                "p5_station_value_pid": best_value[1],
                "p5_station_value_delta_log_only": round(best_value[0], 4),
                "p5_aggro_second_gear_pid": best_aggro[1],
                "p5_aggro_second_gear_delta_log_only": round(best_aggro[0], 4),
            })
        except Exception as exc:
            self._p5_record_error("log_only_reads", exc)

    def _p5_update_telemetry(self, trace: DecisionTrace) -> None:
        self.p5_decision_count += 1
        if trace.get("p5_station_would_apply"):
            self.p5_station_read_fire_count += 1
        if trace.get("p5_r2_would_apply"):
            self.p5_r2_read_fire_count += 1

    def p5_telemetry_summary(self) -> DecisionTrace:
        villains = {}
        for pid in self._p5_sorted_pids(
            type("_P5View", (), {"opponents": list(self._profiles.profiles), "me": None, "seat_indices": {}})()
        ):
            stats = self._profiles.stat_summary(pid, self._p5_confidence_w())
            villains[pid] = {
                "preflop_action_seen": stats.get("preflop_action_seen", 0),
                "postflop_aggression_n": stats.get("postflop_aggression_n", 0),
                "station_response_n": stats.get("station_response_n", 0),
                "pressure_fold_n": stats.get("pressure_fold_n", 0),
                "jam_opportunity_n": stats.get("jam_opportunity_n", 0),
                "jam_like_count": stats.get("jam_like_count", 0),
                "large_bet_count": stats.get("large_bet_count", 0),
                "short_jam_like_count": stats.get("short_jam_like_count", 0),
            }
        decisions = max(1, self.p5_decision_count)
        return {
            "p5_decisions": self.p5_decision_count,
            "p5_station_read_fire_count": self.p5_station_read_fire_count,
            "p5_r2_read_fire_count": self.p5_r2_read_fire_count,
            "p5_any_active_read_fire_rate": (
                (self.p5_station_read_fire_count + self.p5_r2_read_fire_count) / decisions
            ),
            "p5_station_read_fire_rate": self.p5_station_read_fire_count / decisions,
            "p5_r2_read_fire_rate": self.p5_r2_read_fire_count / decisions,
            "p5_error_count": self.p5_error_count,
            "p5_villain_samples": villains,
        }

    def _preflop_action(self, view: PlayerView, context: dict[str, Any]) -> Action | None:
        key = self._hand_key_from_view(view)
        bb = self._infer_big_blind(view)
        n_players = self._table_size(view, context)
        position = getattr(view, "position", None)
        category = self._classify_position(position, n_players)
        strategy_category = self._strategy_category(category, n_players)
        eff_bb = self._eff_bb(view, context)
        hero_bb = float(context.get("hero_bb", eff_bb) or eff_bb)
        raises = self._count_raises(view)
        limpers = self._count_limpers(view)
        threshold = self._shove_threshold(category, n_players)
        band = "short" if eff_bb <= threshold else (
            "medium" if eff_bb <= self.config.medium_band_top_bb else "deep"
        )
        self._trace_preflop(
            position_category=category,
            inferred_bb=bb,
            eff_bb=eff_bb,
            hero_bb=round(hero_bb, 3),
            band=band,
            raises_faced=raises,
            branch="no_cards",
            range_hit=False,
            rank_ctx=context.get("rank_ctx"),
            players_left=(context.get("rank_ctx") or {}).get("players_left", n_players),
            future_edge_tax=0.0,
            reason="missing or invalid hole cards",
        )
        if not key:
            return self._fold_or_check(view)

        if band == "short":
            return self._short_stack_action(
                view, key, strategy_category, n_players, raises, limpers, context, hero_bb, eff_bb
            )
        if band == "medium":
            return self._medium_stack_action(
                view, key, strategy_category, category, n_players, raises, limpers, bb, context, hero_bb, eff_bb
            )
        return self._deep_stack_action(
            view, key, strategy_category, category, n_players, raises, limpers, bb, eff_bb, context, hero_bb
        )

    def _short_stack_action(
        self,
        view: PlayerView,
        key: str,
        strategy_category: str,
        n_players: int,
        raises: int,
        limpers: int,
        context: dict[str, Any],
        hero_bb: float,
        eff_bb: int,
    ) -> Action:
        if self._facing_all_in(view, raises):
            all_in_cat = self._all_in_aggressor_category(view, n_players, raises)
            call_range = _CALL_SHOVE_RANGES[self.config.name].get(all_in_cat, frozenset())
            hit = key in call_range
            self._trace_preflop(branch=f"short_call_off_vs_{all_in_cat}", range_hit=hit,
                                base_range_hit=hit,
                                spot_type="all_in_call",
                                pure_chipEV_action="call" if hit else "fold",
                                phase4_action="call" if hit else "fold",
                                future_edge_tax=0.0,
                                **self._stack_state_trace(
                                    view, context, hero_bb, eff_bb,
                                    short_weight=self._short_weight(hero_bb),
                                ),
                                reason="short stack facing all-in")
            return self._call_or_fold(view) if hit else self._fold_or_check(view)

        shove_range = _SHORT_SHOVE_RANGES[self.config.name].get(strategy_category, frozenset())
        short_nudge = self._short_desperation_nudge(view, hero_bb)
        ceiling = {
            "family": "shove",
            "max_pct": _SHOVE_CEILINGS.get(self.config.name, _SHOVE_CEILINGS["survival"]).get(strategy_category, 1.0),
        }
        hit, range_trace = self._range_hit_with_nudges(
            view,
            key,
            shove_range,
            _SHORT_SHOVE_EXPANSION.get(strategy_category, frozenset()),
            "short_shove",
            [short_nudge],
            ceiling,
        )
        branch = "short_shove" if raises or limpers else "short_open_shove"
        self._trace_preflop(
            branch=branch,
            **range_trace,
            target_total=self._jam_total(view) if hit else None,
            spot_type="shove" if raises or limpers else "open_shove",
            pure_chipEV_action="raise" if range_trace.get("base_range_hit") else "fold",
            phase4_action="raise" if hit else "fold",
            future_edge_tax=0.0,
            urgency_bonus=self._urgency_bonus(hero_bb),
            short_desperation_nudge={
                "rel": round(short_nudge.get("rel", 0.0), 4),
                "abs": round(short_nudge.get("abs", 0.0), 4),
                "weight": round(short_nudge.get("weight", 0.0), 4),
            },
            **self._stack_state_trace(
                view, context, hero_bb, eff_bb,
                short_weight=self._short_weight(hero_bb),
            ),
            reason="short stack shove/fold",
        )
        if hit:
            return self._safe_aggressive(view, self._jam_total(view))
        return self._fold_or_check(view)

    def _medium_stack_action(
        self,
        view: PlayerView,
        key: str,
        strategy_category: str,
        category: str,
        n_players: int,
        raises: int,
        limpers: int,
        bb: int,
        context: dict[str, Any],
        hero_bb: float,
        eff_bb: int,
    ) -> Action:
        if self._facing_all_in(view, raises):
            all_in_cat = self._all_in_aggressor_category(view, n_players, raises)
            call_range = _CALL_SHOVE_RANGES[self.config.name].get(all_in_cat, frozenset())
            hit = key in call_range
            self._trace_preflop(branch=f"medium_call_off_vs_{all_in_cat}", range_hit=hit,
                                base_range_hit=hit,
                                spot_type="all_in_call",
                                pure_chipEV_action="call" if hit else "fold",
                                phase4_action="call" if hit else "fold",
                                future_edge_tax=0.0,
                                **self._stack_state_trace(view, context, hero_bb, eff_bb),
                                reason="medium stack facing all-in")
            return self._call_or_fold(view) if hit else self._fold_or_check(view)

        if raises >= 2:
            hit = key in _VS_3BET_VALUE
            self._trace_preflop(
                branch="medium_vs_3bet_jam",
                range_hit=hit,
                base_range_hit=hit,
                target_total=self._jam_total(view) if hit else None,
                spot_type="shove",
                pure_chipEV_action="raise" if hit else "fold",
                phase4_action="raise" if hit else "fold",
                future_edge_tax=0.0,
                **self._stack_state_trace(view, context, hero_bb, eff_bb),
                reason="medium stack never 3bet-then-folds",
            )
            return self._safe_aggressive(view, self._jam_total(view)) if hit else self._fold_or_check(view)

        if raises == 1:
            opener_cat = self._opener_category(view, n_players)
            resteal_range = _RESTEAL_RANGES[self.config.name].get(opener_cat, _RESTEAL_RANGES[self.config.name]["middle"])
            if opener_cat in ("early", "middle"):
                resteal_range = _RESTEAL_RANGES[self.config.name][opener_cat]
            pressure = self._pressure_weight(
                view,
                context.get("rank_ctx") or self._rank_context(view),
                hero_bb,
                eff_bb,
                [self._opener_pid(view)],
            )
            nudge = self._pressure_nudge(self.config.name, "resteal", pressure, "threebet")
            hit, range_trace = self._range_hit_with_nudges(
                view,
                key,
                resteal_range,
                _PRESSURE_3BET_EXPANSION.get(opener_cat, frozenset()),
                f"medium_resteal_{opener_cat}",
                [nudge],
                {"family": "threebet", "max_pct": 1.0},
            )
            self._trace_preflop(
                branch=f"medium_resteal_vs_{opener_cat}",
                **range_trace,
                target_total=self._jam_total(view) if hit else None,
                spot_type="resteal",
                pressure_weight=round(pressure, 4),
                pure_chipEV_action="raise" if range_trace.get("base_range_hit") else "fold",
                phase4_action="raise" if hit else "fold",
                future_edge_tax=0.0,
                **self._stack_state_trace(view, context, hero_bb, eff_bb, pressure),
                reason="medium stack resteal jam or fold",
            )
            return self._safe_aggressive(view, self._jam_total(view)) if hit else self._fold_or_check(view)

        if limpers:
            iso_range = _ISO_VALUE_SHORT if eff_bb <= 15 else _ISO_VALUE
            pressure = self._pressure_weight(
                view,
                context.get("rank_ctx") or self._rank_context(view),
                hero_bb,
                eff_bb,
                self._limper_pids(view),
            )
            nudge = self._pressure_nudge(self.config.name, "iso", pressure)
            hit, range_trace = self._range_hit_with_nudges(
                view, key, iso_range, _PRESSURE_ISO_EXPANSION,
                "medium_iso", [nudge], {"family": "default", "max_pct": 1.0},
            )
            self._trace_preflop(
                branch="medium_iso_limpers",
                **range_trace,
                target_total=self._jam_total(view) if hit and eff_bb <= 15 else None,
                spot_type="iso",
                pressure_weight=round(pressure, 4),
                pure_chipEV_action="raise" if range_trace.get("base_range_hit") else "fold",
                phase4_action="raise" if hit else "fold",
                future_edge_tax=0.0,
                **self._stack_state_trace(view, context, hero_bb, eff_bb, pressure),
                reason="medium stack iso against limpers",
            )
            if not hit:
                return self._check_or_fold(view) if category == "bb" else self._fold_or_check(view)
            if eff_bb <= 15:
                return self._safe_aggressive(view, self._jam_total(view))
            target = self._iso_total(bb, eff_bb, category, limpers)
            self._trace_preflop(target_total=target)
            return self._safe_aggressive(view, target)

        open_jam_range = _MEDIUM_OPEN_JAM.get(self.config.name, {}).get(strategy_category, frozenset())
        pressure = self._pressure_weight(
            view,
            context.get("rank_ctx") or self._rank_context(view),
            hero_bb,
            eff_bb,
            getattr(view, "acting_opponents", None) or getattr(view, "opponents", None) or [],
        )
        jam_nudges = []
        if pressure > 0.0:
            jam_nudges.append(self._pressure_nudge(self.config.name, "open_jam", pressure, "shove"))
        if hero_bb <= 16.0:
            jam_nudges.append(self._short_desperation_nudge(view, hero_bb))
        jam_hit, jam_trace = self._range_hit_with_nudges(
            view,
            key,
            open_jam_range,
            _SHORT_SHOVE_EXPANSION.get(strategy_category, frozenset()),
            "medium_open_jam",
            jam_nudges,
            {
                "family": "shove",
                "max_pct": _SHOVE_CEILINGS.get(self.config.name, _SHOVE_CEILINGS["survival"]).get(strategy_category, 1.0),
            },
        )
        can_open_jam = strategy_category in ("late", "sb", "heads_up") and jam_hit
        if can_open_jam:
            self._trace_preflop(
                branch="medium_open_jam",
                **jam_trace,
                target_total=self._jam_total(view),
                spot_type="open_jam",
                pressure_weight=round(pressure, 4),
                pure_chipEV_action="raise" if jam_trace.get("base_range_hit") else "fold",
                phase4_action="raise",
                future_edge_tax=0.0,
                urgency_bonus=self._urgency_bonus(hero_bb),
                **self._stack_state_trace(view, context, hero_bb, eff_bb, pressure),
                reason="medium late-position open jam",
            )
            return self._safe_aggressive(view, self._jam_total(view))

        rfi_range = _RFI_RANGES[self.config.name].get(strategy_category, frozenset())
        rfi_nudges = []
        if pressure > 0.0:
            rfi_nudges.append(self._pressure_nudge(
                self.config.name,
                self._rfi_pressure_spot(view, category, strategy_category),
                pressure,
            ))
        if hero_bb <= 16.0:
            transition = self._short_desperation_nudge(view, hero_bb)
            transition["family"] = "default"
            rfi_nudges.append(transition)
        hit, range_trace = self._range_hit_with_nudges(
            view,
            key,
            rfi_range,
            _PRESSURE_RFI_EXPANSION.get(strategy_category, _PRESSURE_RFI_EXPANSION.get(category, frozenset())),
            "medium_rfi",
            rfi_nudges,
            {"family": "default", "max_pct": 1.0},
        )
        self._trace_preflop(
            branch="medium_rfi",
            **range_trace,
            spot_type="steal" if category in ("late", "sb") else "open",
            pressure_weight=round(pressure, 4),
            pure_chipEV_action="raise" if range_trace.get("base_range_hit") else "fold",
            phase4_action="raise" if hit else "fold",
            future_edge_tax=0.0,
            urgency_bonus=self._urgency_bonus(hero_bb),
            **self._stack_state_trace(view, context, hero_bb, eff_bb, pressure),
            reason="medium stack open/fold",
        )
        if hit and self._can_open_from_here(view, category, n_players, bb):
            target = self._open_total(bb, eff_bb, category, n_players)
            self._trace_preflop(target_total=target)
            return self._safe_aggressive(view, target)
        return self._check_or_fold(view) if category == "bb" else self._fold_or_check(view)

    def _deep_stack_action(
        self,
        view: PlayerView,
        key: str,
        strategy_category: str,
        category: str,
        n_players: int,
        raises: int,
        limpers: int,
        bb: int,
        eff_bb: int,
        context: dict[str, Any],
        hero_bb: float,
    ) -> Action:
        if raises >= 3:
            core_hit = key in _VS_4BET_CORE
            conditional_hit = eff_bb <= 30 and key in _VS_4BET_CONDITIONAL
            hit = core_hit or conditional_hit
            target = self._jam_total(view) if hit and eff_bb <= 50 else self._fourbet_total(view)
            self._trace_preflop(
                branch="deep_vs_4bet",
                range_hit=hit,
                base_range_hit=hit,
                target_total=target if hit else None,
                spot_type="4bet_continue",
                pure_chipEV_action="raise" if hit else "fold",
                phase4_action="raise" if hit else "fold",
                future_edge_tax=0.0,
                **self._stack_state_trace(view, context, hero_bb, eff_bb),
                reason="deep stack 4bet continue",
            )
            return self._safe_aggressive(view, target) if hit else self._fold_or_check(view)

        if raises == 2:
            if key in _VS_3BET_VALUE:
                target = self._fourbet_total(view)
                if eff_bb <= 35 or self._commits_fraction(view, target, 0.30):
                    target = self._jam_total(view)
                self._trace_preflop(
                    branch="deep_vs_3bet_value",
                    range_hit=True,
                    base_range_hit=True,
                    target_total=target,
                    spot_type="4bet_value",
                    pure_chipEV_action="raise",
                    phase4_action="raise",
                    future_edge_tax=0.0,
                    **self._stack_state_trace(view, context, hero_bb, eff_bb),
                    reason="premium 4bet or jam versus 3bet",
                )
                return self._safe_aggressive(view, target)
            flat_hit = key in _VS_3BET_FLAT and eff_bb >= 40 and category != "sb"
            self._trace_preflop(
                branch="deep_vs_3bet_flat" if flat_hit else "deep_vs_3bet_fold",
                range_hit=flat_hit,
                base_range_hit=flat_hit,
                spot_type="vs_3bet",
                pure_chipEV_action="call" if flat_hit else "fold",
                phase4_action="call" if flat_hit else "fold",
                future_edge_tax=0.0,
                **self._stack_state_trace(view, context, hero_bb, eff_bb),
                reason="flat narrow range versus 3bet",
            )
            return self._call_or_fold(view) if flat_hit else self._fold_or_check(view)

        if raises == 1:
            opener_cat = self._opener_category(view, n_players)
            value_range = _VALUE_3BET_RANGES[self.config.name].get(opener_cat, _PREMIUM_3BET)
            pressure = self._pressure_weight(
                view,
                context.get("rank_ctx") or self._rank_context(view),
                hero_bb,
                eff_bb,
                [self._opener_pid(view)],
            )
            nudge = self._pressure_nudge(self.config.name, "threebet", pressure, "threebet")
            threebet_hit, range_trace = self._range_hit_with_nudges(
                view,
                key,
                value_range,
                _PRESSURE_3BET_EXPANSION.get(opener_cat, frozenset()),
                f"deep_threebet_{opener_cat}",
                [nudge],
                {"family": "threebet", "max_pct": 1.0},
            )
            if threebet_hit:
                target = self._threebet_total(view, bb, opener_cat, category)
                if eff_bb <= 30 or self._commits_fraction(view, target, 0.30):
                    target = self._jam_total(view)
                self._trace_preflop(
                    branch=f"deep_value_3bet_vs_{opener_cat}",
                    **range_trace,
                    target_total=target,
                    spot_type="3bet",
                    pressure_weight=round(pressure, 4),
                    pure_chipEV_action="raise" if range_trace.get("base_range_hit") else "fold",
                    phase4_action="raise",
                    future_edge_tax=0.0,
                    **self._stack_state_trace(view, context, hero_bb, eff_bb, pressure),
                    reason="value 3bet versus open",
                )
                return self._safe_aggressive(view, target)

            flat_range = _FLAT_VS_OPEN_RANGES[self.config.name].get(opener_cat, frozenset())
            flat_hit = key in flat_range and category != "sb"
            if category == "bb" and opener_cat in ("late", "sb", "heads_up") and self._is_small_open(view, bb):
                flat_hit = flat_hit or key in _BB_DEFEND_LATE_SMALL
            if key in _hands("22-66") and eff_bb < 12:
                flat_hit = False
            self._trace_preflop(
                branch=f"deep_flat_vs_{opener_cat}" if flat_hit else f"deep_fold_vs_{opener_cat}",
                range_hit=flat_hit,
                base_range_hit=flat_hit,
                spot_type="flat_vs_open",
                pressure_weight=round(pressure, 4),
                pure_chipEV_action="call" if flat_hit else "fold",
                phase4_action="call" if flat_hit else "fold",
                future_edge_tax=0.0,
                **self._stack_state_trace(view, context, hero_bb, eff_bb, pressure),
                reason="flat call-set versus open",
            )
            return self._call_or_fold(view) if flat_hit else self._fold_or_check(view)

        if limpers:
            return self._limped_pot_action(view, key, category, bb, limpers, eff_bb, context, hero_bb)

        rfi_range = _RFI_RANGES[self.config.name].get(strategy_category, frozenset())
        pressure = self._pressure_weight(
            view,
            context.get("rank_ctx") or self._rank_context(view),
            hero_bb,
            eff_bb,
            getattr(view, "acting_opponents", None) or getattr(view, "opponents", None) or [],
        )
        nudge = self._pressure_nudge(
            self.config.name,
            self._rfi_pressure_spot(view, category, strategy_category),
            pressure,
        )
        hit, range_trace = self._range_hit_with_nudges(
            view,
            key,
            rfi_range,
            _PRESSURE_RFI_EXPANSION.get(strategy_category, _PRESSURE_RFI_EXPANSION.get(category, frozenset())),
            "deep_rfi",
            [nudge],
            {"family": "default", "max_pct": 1.0},
        )
        self._trace_preflop(
            branch="deep_rfi",
            **range_trace,
            spot_type="steal" if category in ("late", "sb") else "open",
            pressure_weight=round(pressure, 4),
            pure_chipEV_action="raise" if range_trace.get("base_range_hit") else "fold",
            phase4_action="raise" if hit else "fold",
            future_edge_tax=0.0,
            **self._stack_state_trace(view, context, hero_bb, eff_bb, pressure),
            reason="deep stack open/fold",
        )
        if hit and self._can_open_from_here(view, category, n_players, bb):
            target = self._open_total(bb, eff_bb, category, n_players)
            self._trace_preflop(target_total=target)
            return self._safe_aggressive(view, target)
        return self._check_or_fold(view) if category == "bb" else self._fold_or_check(view)

    def _limped_pot_action(
        self,
        view: PlayerView,
        key: str,
        category: str,
        bb: int,
        limpers: int,
        eff_bb: int,
        context: dict[str, Any],
        hero_bb: float,
    ) -> Action:
        pressure = self._pressure_weight(
            view,
            context.get("rank_ctx") or self._rank_context(view),
            hero_bb,
            eff_bb,
            self._limper_pids(view),
        )
        base_iso = _ISO_VALUE
        if eff_bb <= 15:
            base_iso = _ISO_VALUE_SHORT
        nudge = self._pressure_nudge(self.config.name, "iso", pressure)
        iso_hit, range_trace = self._range_hit_with_nudges(
            view,
            key,
            base_iso,
            _PRESSURE_ISO_EXPANSION,
            "deep_iso",
            [nudge],
            {"family": "default", "max_pct": 1.0},
        )
        if iso_hit:
            target = self._jam_total(view) if eff_bb <= 15 else self._iso_total(bb, eff_bb, category, limpers)
            self._trace_preflop(
                branch="deep_iso_limpers",
                **range_trace,
                target_total=target,
                spot_type="iso",
                pressure_weight=round(pressure, 4),
                pure_chipEV_action="raise" if range_trace.get("base_range_hit") else "fold",
                phase4_action="raise",
                future_edge_tax=0.0,
                **self._stack_state_trace(view, context, hero_bb, eff_bb, pressure),
                reason="value iso versus limpers",
            )
            return self._safe_aggressive(view, target)

        overlimp_hit = key in _OVERLIMP_SPECULATIVE and category in ("late", "middle", "sb")
        self._trace_preflop(
            branch="deep_overlimp" if overlimp_hit else "deep_limped_fold_or_check",
            range_hit=overlimp_hit,
            base_range_hit=overlimp_hit,
            spot_type="overlimp",
            pressure_weight=round(pressure, 4),
            pure_chipEV_action="call" if overlimp_hit else "fold",
            phase4_action="call" if overlimp_hit else "fold",
            future_edge_tax=0.0,
            **self._stack_state_trace(view, context, hero_bb, eff_bb, pressure),
            reason="speculative overlimp or blind check",
        )
        if overlimp_hit:
            return self._call_or_check(view)
        if category == "bb":
            return self._check_or_fold(view)
        return self._fold_or_check(view)

    # ------------------------------------------------------------------
    # Phase 3: postflop equity and pot odds.
    # ------------------------------------------------------------------
    def _postflop_action(self, view: PlayerView, context: dict[str, Any]) -> Action | None:
        street = getattr(view, "street", None)
        if street not in _STREET_IDX:
            return None

        hole = getattr(view, "hole_cards", None) or []
        board = getattr(view, "board", None) or []
        legal = self._legal_types(view)
        n_opp = max(1, len(getattr(view, "opponents", None) or []))
        pot = self._safe_int(getattr(view, "pot", 0))
        to_call = self._safe_int(getattr(view, "to_call", 0))
        si = _STREET_IDX[street]
        way3 = "hu" if n_opp == 1 else ("3way" if n_opp == 2 else "4plus")
        way2 = "hu" if n_opp == 1 else "mw"
        profile = self.config.name

        self._trace_preflop(
            branch="postflop", street=street, n_opp=n_opp,
            reason="postflop equity vs pot odds",
        )

        if not self._valid_postflop_cards(street, hole, board):
            self._trace_preflop(branch="postflop_no_cards", reason="missing or invalid postflop cards")
            return self._fold_or_check(view)

        sims = self._sim_budget(view, pot, to_call, n_opp)
        eq = self._postflop_equity(view, hole, board, n_opp, sims)
        if eq is None:
            self._trace_preflop(branch="postflop_equity_error", reason="postflop equity unavailable")
            return self._fold_or_check(view)
        self._trace_preflop(equity=round(eq, 4), sims=sims)

        if to_call > 0:
            return self._postflop_vs_bet(
                view, eq, pot, to_call, profile, way3, way2, si, n_opp, context, hole, board, legal
            )
        return self._postflop_open(
            view, eq, pot, profile, way3, way2, si, n_opp, context, hole, board, legal
        )

    def _postflop_vs_bet(
        self,
        view: PlayerView,
        eq: float,
        pot: int,
        to_call: int,
        profile: str,
        way3: str,
        way2: str,
        si: int,
        n_opp: int,
        context: dict[str, Any],
        hole: list,
        board: list,
        legal: set[str],
    ) -> Action:
        stacks = getattr(view, "stacks", None) or {}
        hero_stack = self._safe_int(stacks.get(getattr(view, "me", None), 0))
        call_cost = min(max(0, to_call), max(0, hero_stack)) if hero_stack > 0 else max(0, to_call)
        denom = pot + call_cost
        pot_odds = (call_cost / denom) if denom > 0 else 0.0
        hero_bb = float(context.get("hero_bb", 0.0) or 0.0)
        if hero_bb <= 0.0:
            hero_bb = hero_stack / max(1, self._infer_big_blind(view))
        eff_bb = self._eff_bb(view, context)
        rank_ctx = context.get("rank_ctx") or self._rank_context(view)
        players_left = max(2, self._safe_int(rank_ctx.get("players_left")) or self._table_size(view, context))
        all_in_call = hero_stack > 0 and to_call >= hero_stack
        commit_frac = (call_cost / max(1, hero_stack)) if hero_stack > 0 else 0.0
        future_tax = self._future_edge_tax(
            view,
            context,
            hero_bb,
            eff_bb,
            commit_frac,
            players_left,
            rank_ctx,
            is_all_in_call=all_in_call,
        )
        future_tax = self._p5_relieve_future_tax(
            view,
            future_tax,
            eq=eq,
            pot_odds=pot_odds,
            all_in_call=all_in_call,
        )
        base_cushion = _CALL_CUSHION.get(profile, _CALL_CUSHION["survival"])[way3][si]
        hu_weight = self._heads_up_weight(eff_bb) if players_left == 2 else 0.0
        cushion_deltas: list[dict[str, float]] = []
        if players_left == 2 and not all_in_call:
            cushion_deltas.append({
                "delta": _HU_POSTFLOP.get(profile, _HU_POSTFLOP["survival"])["call_cushion"][si],
                "weight": hu_weight * float(self.config.hu_aggression_scale or 1.0),
            })
        cushion = self._apply_call_cushion(base_cushion, cushion_deltas, profile)
        base_call_req = pot_odds + base_cushion
        call_req = pot_odds + cushion + future_tax
        pressure = self._pressure_weight(
            view,
            rank_ctx,
            hero_bb,
            eff_bb,
            getattr(view, "opponents", None) or [],
        )
        self._trace_preflop(
            pot_odds=round(pot_odds, 4),
            pure_chipEV_threshold=round(pot_odds, 4),
            threshold_before=round(base_call_req, 4),
            threshold_after=round(call_req, 4),
            call_cushion_before=round(base_cushion, 4),
            call_cushion_after=round(cushion, 4),
            call_req=round(call_req, 4),
            future_edge_tax=round(future_tax, 4),
            commit_tax=round(future_tax, 4),
            spot_type="all_in_call" if all_in_call else "postflop_call",
            pure_chipEV_action="call" if eq >= pot_odds else "fold",
            phase3_action="call" if eq >= base_call_req else "fold",
            pressure_weight=round(pressure, 4),
            heads_up_weight=round(hu_weight, 4),
            players_left=players_left,
            **self._stack_state_trace(view, context, hero_bb, eff_bb, pressure, hu_weight=hu_weight, commit_frac=commit_frac),
        )

        can_raise = "raise" in legal and self._first_aggressive_spec(view) is not None
        base_value_raise = self._value_raise_thr(profile, way3, si)
        value_deltas: list[dict[str, float]] = []
        if players_left == 2:
            value_deltas.append({
                "delta": _HU_POSTFLOP.get(profile, _HU_POSTFLOP["survival"])["value_raise"][si],
                "weight": hu_weight * float(self.config.hu_aggression_scale or 1.0),
            })
        value_raise_thr = self._apply_value_threshold(base_value_raise, value_deltas, profile, True)
        self._trace_preflop(
            value_threshold_before=round(base_value_raise, 4),
            value_threshold_after=round(value_raise_thr, 4),
        )

        if can_raise and eq >= value_raise_thr:
            target = self._raise_total(pot, to_call, self._sizing(profile, "value", way2, si))
            self._trace_preflop(branch="postflop_value_raise", target_total=target,
                                phase4_action="raise",
                                reason="raise made hand for value")
            return self._safe_aggressive(view, target)

        if can_raise and si <= 1:
            lo, hi = _SEMI_BAND[way3][si]
            if lo <= eq <= hi:
                base_freq = _SEMI_FREQ.get(profile, _SEMI_FREQ["survival"])[way3][si] \
                    * self._draw_mult(hole, board)
                mults = []
                mults.append({
                    "mult": _CHIPLEADER_POSTFLOP_MULT.get(profile, _CHIPLEADER_POSTFLOP_MULT["survival"])[way3]["semi"][si],
                    "weight": pressure,
                })
                if players_left == 2:
                    mults.append({
                        "mult": _HU_POSTFLOP.get(profile, _HU_POSTFLOP["survival"])["semi"][si],
                        "weight": hu_weight * float(self.config.hu_aggression_scale or 1.0),
                    })
                freq = self._apply_freq_nudges(base_freq, mults, profile, "semi")
                self._trace_preflop(freq_before=round(base_freq, 4), freq_after=round(freq, 4))
                if self._freq_gate(view, "semibluff_raise", freq):
                    target = self._raise_total(pot, to_call,
                                               self._sizing(profile, "semibluff", way2, si))
                    self._trace_preflop(branch="postflop_semibluff_raise", target_total=target,
                                        phase4_action="raise",
                                        reason="semi-bluff raise with draw")
                    return self._safe_aggressive(view, target)

        if eq >= call_req:
            self._trace_preflop(branch="postflop_call", phase4_action="call",
                                reason="equity meets the price")
            return self._call_or_fold(view)

        self._trace_preflop(branch="postflop_fold", phase4_action="fold",
                            reason="equity below the price")
        return self._fold_or_check(view)

    def _postflop_open(
        self,
        view: PlayerView,
        eq: float,
        pot: int,
        profile: str,
        way3: str,
        way2: str,
        si: int,
        n_opp: int,
        context: dict[str, Any],
        hole: list,
        board: list,
        legal: set[str],
    ) -> Action:
        can_bet = self._first_aggressive_spec(view) is not None
        hero_stack = self._safe_int((getattr(view, "stacks", None) or {}).get(getattr(view, "me", None), 0))
        hero_bb = float(context.get("hero_bb", 0.0) or 0.0)
        if hero_bb <= 0.0:
            hero_bb = hero_stack / max(1, self._infer_big_blind(view))
        eff_bb = self._eff_bb(view, context)
        rank_ctx = context.get("rank_ctx") or self._rank_context(view)
        players_left = max(2, self._safe_int(rank_ctx.get("players_left")) or self._table_size(view, context))
        pressure = self._pressure_weight(
            view,
            rank_ctx,
            hero_bb,
            eff_bb,
            getattr(view, "opponents", None) or [],
        )
        hu_weight = self._heads_up_weight(eff_bb) if players_left == 2 else 0.0
        base_value_thr = self._value_bet_thr(profile, way3, si)
        value_deltas: list[dict[str, float]] = []
        if players_left == 2:
            value_deltas.append({
                "delta": _HU_POSTFLOP.get(profile, _HU_POSTFLOP["survival"])["value_bet"][si],
                "weight": hu_weight * float(self.config.hu_aggression_scale or 1.0),
            })
        value_thr = self._apply_value_threshold(base_value_thr, value_deltas, profile, False)
        self._trace_preflop(
            players_left=players_left,
            pressure_weight=round(pressure, 4),
            heads_up_weight=round(hu_weight, 4),
            value_threshold_before=round(base_value_thr, 4),
            value_threshold_after=round(value_thr, 4),
            threshold_before=round(base_value_thr, 4),
            threshold_after=round(value_thr, 4),
            future_edge_tax=0.0,
            **self._stack_state_trace(view, context, hero_bb, eff_bb, pressure, hu_weight=hu_weight),
        )

        if can_bet and eq >= value_thr:
            target = self._bet_total(pot, self._sizing(profile, "value", way2, si))
            self._trace_preflop(branch="postflop_value_bet", target_total=target,
                                spot_type="value_bet",
                                pure_chipEV_action="bet" if eq >= base_value_thr else "check",
                                phase4_action="bet",
                                reason="value bet made hand")
            return self._safe_aggressive(view, target)

        if can_bet and si <= 1:
            lo, hi = _SEMI_BAND[way3][si]
            if lo <= eq <= hi:
                base_freq = _SEMI_FREQ.get(profile, _SEMI_FREQ["survival"])[way3][si] \
                    * self._draw_mult(hole, board)
                mults = [{
                    "mult": _CHIPLEADER_POSTFLOP_MULT.get(profile, _CHIPLEADER_POSTFLOP_MULT["survival"])[way3]["semi"][si],
                    "weight": pressure,
                }]
                if players_left == 2:
                    mults.append({
                        "mult": _HU_POSTFLOP.get(profile, _HU_POSTFLOP["survival"])["semi"][si],
                        "weight": hu_weight * float(self.config.hu_aggression_scale or 1.0),
                    })
                freq = self._apply_freq_nudges(base_freq, mults, profile, "semi")
                self._trace_preflop(
                    spot_type="semibluff",
                    pure_chipEV_action="bet" if self._freq_gate(view, "semibluff_bet", base_freq) else "check",
                    freq_before=round(base_freq, 4),
                    freq_after=round(freq, 4),
                )
                if self._freq_gate(view, "semibluff_bet", freq):
                    target = self._bet_total(pot, self._sizing(profile, "semibluff", way2, si))
                    self._trace_preflop(branch="postflop_semibluff_bet", target_total=target,
                                        phase4_action="bet",
                                        reason="semi-bluff bet with draw")
                    return self._safe_aggressive(view, target)

        if can_bet and eq < value_thr:
            is_pfr = self._last_preflop_raiser(view) == getattr(view, "me", None)
            if is_pfr:
                base_freq = _CBET_FREQ.get(profile, _CBET_FREQ["survival"])[way3][si]
                knob = "cbet"
            else:
                base_freq = _BLUFF_FREQ.get(profile, _BLUFF_FREQ["survival"])[way3][si]
                knob = "bluff"
            mults = [{
                "mult": _CHIPLEADER_POSTFLOP_MULT.get(profile, _CHIPLEADER_POSTFLOP_MULT["survival"])[way3][knob][si],
                "weight": pressure,
            }]
            if players_left == 2:
                mults.append({
                    "mult": _HU_POSTFLOP.get(profile, _HU_POSTFLOP["survival"])[knob][si],
                    "weight": hu_weight * float(self.config.hu_aggression_scale or 1.0),
                })
            adjusted_freq = self._apply_freq_nudges(base_freq, mults, profile, knob)
            board_mult = self._board_bluff_mult(board, n_opp)
            base_final_freq = base_freq * board_mult
            freq = adjusted_freq * board_mult
            freq = self._p5_station_frequency(view, knob, freq)
            base_gate = self._freq_gate(view, "bluff", base_final_freq)
            self._trace_preflop(
                spot_type="cbet" if is_pfr else "bluff",
                pure_chipEV_action="bet" if base_gate else "check",
                freq_before=round(base_final_freq, 4),
                freq_after=round(freq, 4),
            )
            if self._freq_gate(view, "bluff", freq):
                target = self._bet_total(pot, self._sizing(profile, "bluff", way2, si))
                self._trace_preflop(branch="postflop_cbet" if is_pfr else "postflop_bluff",
                                    target_total=target, phase4_action="bet",
                                    reason="bluff / continuation bet")
                return self._safe_aggressive(view, target)

        self._trace_preflop(branch="postflop_check", phase4_action="check",
                            reason="check, no profitable bet")
        return self._check_or_fold(view)

    def _postflop_equity(self, view: PlayerView, hole: list, board: list,
                         n_opp: int, sims: int) -> float | None:
        """Seeded Monte Carlo equity so the same spot is reproducible.

        The global ``random`` state is saved and restored so the engine's own
        RNG stream is never disturbed by our seeding.
        """
        seed = self._stable_hash(view, "equity") & 0x7FFFFFFF
        state = _random.getstate()
        try:
            _random.seed(seed)
            return float(_mc_equity(hole, board, max(1, n_opp), n_sims=max(1, int(sims))))
        except Exception:
            return None
        finally:
            _random.setstate(state)

    @staticmethod
    def _valid_postflop_cards(street: str, hole: list, board: list) -> bool:
        expected_board = {"flop": 3, "turn": 4, "river": 5}.get(street)
        if expected_board is None:
            return False
        if len(hole or []) != 2 or len(board or []) != expected_board:
            return False
        seen: set[tuple[str, str]] = set()
        for card in list(hole or []) + list(board or []):
            try:
                rank, suit = card[0], card[1]
            except (TypeError, IndexError):
                return False
            if rank not in _RANK_INDEX or suit not in _SUITS:
                return False
            key = (rank, suit)
            if key in seen:
                return False
            seen.add(key)
        return True

    def _sim_budget(self, view: PlayerView, pot: int, to_call: int, n_opp: int) -> int:
        stacks = getattr(view, "stacks", None) or {}
        hero_stack = self._safe_int(stacks.get(getattr(view, "me", None), 0))
        cap = self.postflop_sim_cap
        if hero_stack > 0:
            if to_call >= hero_stack or to_call >= 0.35 * hero_stack:
                return min(cap, self.postflop_allin_sims)
            if to_call >= 0.20 * hero_stack:
                return min(cap, self.postflop_big_sims)
        bb = max(1, self._infer_big_blind(view))
        if pot >= 12 * bb:
            return min(cap, self.postflop_big_sims)
        return min(cap, self.postflop_base_sims)

    # ------------------------------------------------------------------
    # Phase 4: tournament stack context and bounded nudges.
    # ------------------------------------------------------------------
    def _rank_context(self, view: PlayerView) -> DecisionTrace:
        hand_id = getattr(view, "hand_id", None)
        if hand_id is not None and hand_id in self._rank_cache:
            return dict(self._rank_cache[hand_id])

        ctx = self._rank_context_uncached(view)
        if hand_id is not None:
            self._rank_cache[hand_id] = dict(ctx)
        return ctx

    def _rank_context_uncached(self, view: PlayerView) -> DecisionTrace:
        stacks = getattr(view, "stacks", None) or {}
        hero = getattr(view, "me", None)
        hero_stack = self._safe_int(stacks.get(hero, 0))
        all_in_opponents = set(getattr(view, "all_in_opponents", None) or [])
        alive: list[tuple[Any, int]] = []
        seen_alive: set[Any] = set()
        for pid, stack in stacks.items():
            stack_value = self._safe_int(stack)
            if stack_value > 0 or pid in all_in_opponents:
                alive.append((pid, stack_value))
                seen_alive.add(pid)
        for pid in all_in_opponents:
            if pid not in seen_alive:
                alive.append((pid, 0))
                seen_alive.add(pid)
        if hero_stack > 0 and hero not in seen_alive:
            alive.append((hero, hero_stack))
        if not alive:
            return {
                "bucket": "middle",
                "rank": 1,
                "raw_rank": 1,
                "players_left": 1,
                "hero_stack": hero_stack,
                "avg_stack": max(1, hero_stack),
                "avg_ratio": 1.0,
                "median_ratio": 1.0,
                "leader_ratio": 1.0,
                "chip_share": 1.0,
                "cover_count": 0,
                "cover_ratio": 0.0,
            }

        amounts = sorted((stack for _, stack in alive), reverse=True)
        players_left = len(amounts)
        total = max(1, sum(amounts))
        avg = total / players_left
        median = amounts[players_left // 2] if players_left % 2 else (
            (amounts[players_left // 2 - 1] + amounts[players_left // 2]) / 2.0
        )
        leader = max(1, amounts[0])
        second = amounts[1] if players_left > 1 else amounts[0]
        tie_tol = 0.05
        raw_rank = 1 + sum(1 for stack in amounts if stack > hero_stack)
        rank = 1 + sum(1 for stack in amounts if stack > hero_stack * (1.0 + tie_tol))
        opponents = [
            (pid, stack)
            for pid, stack in alive
            if pid != hero
        ]
        cover_count = sum(1 for _, stack in opponents if hero_stack > stack)
        opp_count = max(1, players_left - 1)
        avg_ratio = hero_stack / avg if avg > 0 else 1.0
        median_ratio = hero_stack / median if median > 0 else 1.0
        leader_ratio = hero_stack / leader if leader > 0 else 1.0
        chip_share = hero_stack / total if total > 0 else 0.0
        cover_ratio = cover_count / opp_count

        if players_left == 2:
            villain_stack = opponents[0][1] if opponents else hero_stack
            if abs(hero_stack - villain_stack) <= max(hero_stack, villain_stack) * tie_tol:
                bucket = "hu_even"
            elif hero_stack > villain_stack:
                bucket = "hu_leader"
            else:
                bucket = "hu_trailer"
        else:
            lead_avg_cut = 1.10 if players_left <= 3 else 1.20
            ahead_second = hero_stack >= second * 1.05 if players_left > 1 else True
            if rank == 1 and avg_ratio >= lead_avg_cut and ahead_second:
                bucket = "chip_leader"
            elif rank <= 2 and (avg_ratio >= 0.95 or cover_count >= (opp_count + 1) // 2):
                bucket = "top_2"
            else:
                if players_left <= 3:
                    short_rank = rank == players_left and (
                        avg_ratio <= 0.80 or leader_ratio <= 0.65
                    )
                elif players_left == 4:
                    short_rank = rank == 4 and (
                        avg_ratio <= 0.75 or median_ratio <= 0.70 or leader_ratio <= 0.35
                    )
                else:
                    short_rank = rank >= players_left - 1 and (
                        avg_ratio <= 0.75 or median_ratio <= 0.70 or leader_ratio <= 0.35
                    )
                bucket = "short_stack" if short_rank else "middle"

        return {
            "bucket": bucket,
            "rank": rank,
            "raw_rank": raw_rank,
            "players_left": players_left,
            "hero_stack": hero_stack,
            "avg_stack": round(avg, 3),
            "avg_ratio": round(avg_ratio, 4),
            "median_ratio": round(median_ratio, 4),
            "leader_ratio": round(leader_ratio, 4),
            "chip_share": round(chip_share, 4),
            "cover_count": cover_count,
            "cover_ratio": round(cover_ratio, 4),
        }

    @staticmethod
    def _rank_weight(hero_bb: float, eff_bb: float) -> float:
        return _clamp((min(float(hero_bb), float(eff_bb)) - 10.0) / 20.0)

    @staticmethod
    def _cover_weight(hero_stack: int, villain_stacks: list[int]) -> float:
        live = [max(0, int(stack)) for stack in villain_stacks if int(stack) > 0]
        if not live or hero_stack <= 0:
            return 0.0
        covered = sum(1 for stack in live if hero_stack > stack)
        ratio = covered / len(live)
        if ratio >= 1.0:
            return 1.0
        if ratio >= 2.0 / 3.0:
            return 0.70
        if ratio >= 0.50:
            return 0.40
        return 0.0

    def _pressure_weight(
        self,
        view: PlayerView,
        rank_ctx: dict[str, Any],
        hero_bb: float,
        eff_bb: float,
        relevant_villains: list[Any] | tuple[Any, ...] | None = None,
    ) -> float:
        if hero_bb <= 12.0:
            return 0.0
        bucket = str(rank_ctx.get("bucket", "middle"))
        bucket_weight = _RANK_BUCKET_WEIGHT.get(bucket, 0.0)
        if bucket_weight <= 0.0:
            return 0.0

        stacks = getattr(view, "stacks", None) or {}
        hero_stack = self._safe_int(stacks.get(getattr(view, "me", None), 0))
        villains = list(relevant_villains or getattr(view, "acting_opponents", None) or getattr(view, "opponents", None) or [])
        villain_stacks = [
            self._safe_int(stacks.get(pid, 0))
            for pid in villains
            if pid != getattr(view, "me", None) and self._safe_int(stacks.get(pid, 0)) > 0
        ]
        if not villain_stacks:
            villain_stacks = [
                self._safe_int(stack)
                for pid, stack in stacks.items()
                if pid != getattr(view, "me", None) and self._safe_int(stack) > 0
            ]
        cover = self._cover_weight(hero_stack, villain_stacks)
        if cover <= 0.0:
            return 0.0

        players_left = max(2, self._safe_int(rank_ctx.get("players_left")) or self._table_size(view))
        avg_share = 1.0 / players_left
        chip_share = float(rank_ctx.get("chip_share", 0.0) or 0.0)
        chip_share_weight = _clamp((chip_share - avg_share) / max(avg_share, 0.01))
        if bucket == "top_2":
            chip_share_weight = max(chip_share_weight, 0.35)

        bb = max(1, self._infer_big_blind(view))
        covered_short = any(hero_stack > stack and (stack / bb) <= 12.0 for stack in villain_stacks)
        depth_gate = self._rank_weight(hero_bb, eff_bb)
        if covered_short:
            depth_gate = max(depth_gate, 0.65)

        max_villain = max(villain_stacks) if villain_stacks else 0
        cover_margin = _clamp(((hero_stack / max(1, max_villain)) - 1.0) / 0.75)
        cover_quality = max(0.45, cover_margin) if hero_stack > max_villain else cover
        scale = float(self.config.chipleader_pressure_scale or 1.0)
        return _clamp(bucket_weight * cover * cover_quality * depth_gate * max(chip_share_weight, 0.30) * scale)

    @staticmethod
    def _short_weight(hero_bb: float) -> float:
        hero_bb = float(hero_bb)
        if hero_bb <= 4.0:
            return 1.0
        if hero_bb >= 16.0:
            return 0.0
        return _clamp((16.0 - hero_bb) / 12.0)

    @staticmethod
    def _heads_up_weight(eff_bb: float) -> float:
        return _clamp((float(eff_bb) - 8.0) / 17.0)

    def _urgency_bonus(self, hero_bb: float) -> float:
        # Telemetry-only in Phase 4 preflop; range widening is the mechanism.
        max_bonus = 0.020 if self.config.name == "survival" else 0.030
        return round(max_bonus * self._short_weight(hero_bb), 4)

    def _m_ratio(self, view: PlayerView, hero_stack: int | None = None) -> float:
        bb = max(1, self._infer_big_blind(view))
        if hero_stack is None:
            stacks = getattr(view, "stacks", None) or {}
            hero_stack = self._safe_int(stacks.get(getattr(view, "me", None), 0))
        return hero_stack / max(1.0, 1.5 * bb)

    @staticmethod
    def _hands_until_blind_up(view: PlayerView) -> int:
        schedule = getattr(view, "blind_increase_every", None)
        if schedule is None:
            schedule = 50
        try:
            schedule = int(schedule)
        except (TypeError, ValueError):
            schedule = 50
        if schedule <= 0:
            return 10**9
        try:
            hand_id = int(getattr(view, "hand_id", 0) or 0)
        except (TypeError, ValueError):
            return schedule
        remainder = hand_id % schedule
        return schedule if remainder == 0 else schedule - remainder

    def _short_desperation_nudge(self, view: PlayerView, hero_bb: float) -> dict[str, float]:
        rows = _SHORT_DESPERATION.get(self.config.name, _SHORT_DESPERATION["survival"])
        rel = 0.0
        abs_pp = 0.0
        for top, row_rel, row_abs in rows:
            if hero_bb <= top:
                rel = row_rel
                abs_pp = row_abs
                break
        if rel <= 0.0 and abs_pp <= 0.0:
            return {"rel": 0.0, "abs": 0.0, "weight": 0.0, "family": "shove"}

        stacks = getattr(view, "stacks", None) or {}
        hero_stack = self._safe_int(stacks.get(getattr(view, "me", None), 0))
        m_ratio = self._m_ratio(view, hero_stack)
        m_factor = 1.15 if m_ratio <= 4.0 else (1.08 if m_ratio <= 6.0 else 1.0)
        clock = self._hands_until_blind_up(view)
        clock_factor = 1.20 if clock <= 5 else (1.10 if clock <= 10 else 1.0)
        scale = float(self.config.desperation_scale or 1.0)
        weight = _clamp(self._short_weight(hero_bb) * m_factor * clock_factor * scale, 0.0, 1.25)
        return {"rel": rel, "abs": abs_pp, "weight": weight, "family": "shove"}

    def _apply_range_nudges(
        self,
        base_pct: float,
        nudges: list[dict[str, float]] | tuple[dict[str, float], ...],
        profile: str,
        ceiling: Any,
    ) -> float:
        family = "default"
        max_pct = 1.0
        if isinstance(ceiling, dict):
            family = str(ceiling.get("family", family))
            max_pct = float(ceiling.get("max_pct", max_pct))
        elif isinstance(ceiling, str):
            family = ceiling
        elif ceiling is not None:
            try:
                max_pct = float(ceiling)
            except (TypeError, ValueError):
                max_pct = 1.0

        raw_delta = 0.0
        for nudge in nudges or []:
            if not isinstance(nudge, dict):
                continue
            rel = max(0.0, float(nudge.get("rel", 0.0) or 0.0))
            abs_pp = max(0.0, float(nudge.get("abs", 0.0) or 0.0))
            weight = max(0.0, float(nudge.get("weight", 1.0) or 0.0))
            raw_delta += min(max(0.0, base_pct) * rel, abs_pp) * weight
            family = str(nudge.get("family", family))

        cap_rel, cap_abs = _RANGE_GEAR_CAPS.get(profile, _RANGE_GEAR_CAPS["survival"]).get(
            family,
            _RANGE_GEAR_CAPS.get(profile, _RANGE_GEAR_CAPS["survival"])["default"],
        )
        capped_delta = min(raw_delta, max(0.0, base_pct) * cap_rel, cap_abs)
        return _clamp(max(0.0, base_pct) + capped_delta, 0.0, max(0.0, min(1.0, max_pct)))

    def _apply_freq_nudges(
        self,
        base_freq: float,
        mults: list[dict[str, float]] | tuple[dict[str, float], ...],
        profile: str,
        knob_name: str,
    ) -> float:
        freq = _clamp(base_freq)
        for mult in mults or []:
            if not isinstance(mult, dict):
                continue
            target_mult = float(mult.get("mult", 1.0) or 1.0)
            weight = max(0.0, float(mult.get("weight", 1.0) or 0.0))
            freq *= max(0.0, 1.0 + (target_mult - 1.0) * weight)
        caps = _FREQ_GEAR_CAPS.get(profile, _FREQ_GEAR_CAPS["survival"])
        ceiling = _FREQ_CEILINGS.get(profile, _FREQ_CEILINGS["survival"]).get(knob_name, 1.0)
        max_freq = min(base_freq * caps["mult"], base_freq + caps["abs"], ceiling)
        return _clamp(freq, 0.0, max_freq)

    def _apply_call_cushion(
        self,
        base: float,
        deltas: list[dict[str, float]] | tuple[dict[str, float], ...],
        profile: str,
    ) -> float:
        value = float(base)
        for delta in deltas or []:
            if not isinstance(delta, dict):
                continue
            value += float(delta.get("delta", 0.0) or 0.0) * max(0.0, float(delta.get("weight", 1.0) or 0.0))
        max_reduction = 0.035 if profile == "aggro" else 0.025
        max_increase = 0.020
        return _clamp(value, max(0.0, base - max_reduction), min(0.30, base + max_increase))

    def _apply_value_threshold(
        self,
        base: float,
        deltas: list[dict[str, float]] | tuple[dict[str, float], ...],
        profile: str,
        is_raise: bool,
    ) -> float:
        value = float(base)
        for delta in deltas or []:
            if not isinstance(delta, dict):
                continue
            value += float(delta.get("delta", 0.0) or 0.0) * max(0.0, float(delta.get("weight", 1.0) or 0.0))
        max_reduction = 0.030 if profile == "aggro" else 0.020
        if is_raise:
            max_reduction *= 0.80
        return _clamp(value, max(0.35, base - max_reduction), min(0.95, base + 0.020))

    def _future_edge_tax(
        self,
        view: PlayerView,
        ctx: dict[str, Any],
        hero_bb: float,
        eff_bb: float,
        commit_frac: float,
        players_left: int,
        rank: dict[str, Any] | str | None,
        *,
        is_all_in_call: bool = False,
    ) -> float:
        if commit_frac < 0.25 or hero_bb <= 10.0:
            return 0.0
        profile = self.config.name
        cap_table = _FUTURE_EDGE_TAX_CAPS.get(profile, _FUTURE_EDGE_TAX_CAPS["survival"])
        capped_players = max(2, min(6, int(players_left or 2)))
        cap = cap_table.get(capped_players, cap_table.get(6, 0.0))
        if players_left >= 6:
            cap = cap_table.get(6, cap)
        if self.future_edge_tax_cap_override is not None:
            cap = min(cap, max(0.0, float(self.future_edge_tax_cap_override)))
        if is_all_in_call:
            cap = min(cap, 0.010 if profile == "survival" else 0.004)
        if players_left == 2:
            cap = min(cap, 0.008 if profile == "survival" else 0.003)
            depth_factor = _clamp((float(eff_bb) - 12.0) / 28.0)
        else:
            depth_factor = _clamp((min(float(hero_bb), float(eff_bb)) - 12.0) / 28.0)
        if cap <= 0.0 or depth_factor <= 0.0:
            return 0.0

        commit_factor = _clamp((float(commit_frac) - 0.25) / 0.50)
        bust_factor = 1.0 if commit_frac >= 1.0 else _clamp((float(commit_frac) - 0.50) / 0.50)
        if hero_bb <= 16.0:
            depth_factor *= 0.25
        tax = cap * depth_factor * commit_factor * max(0.35, bust_factor)
        tax *= float(self.config.future_edge_tax_scale or 1.0)
        return round(_clamp(tax, 0.0, cap), 4)

    @staticmethod
    def _range_pct(hands: frozenset[str] | set[str]) -> float:
        return _clamp(len(hands or ()) / _RANGE_DENOM)

    def _range_hit_with_nudges(
        self,
        view: PlayerView,
        key: str,
        base_range: frozenset[str],
        expansion_range: frozenset[str],
        spot: str,
        nudges: list[dict[str, float]],
        ceiling: Any,
    ) -> tuple[bool, DecisionTrace]:
        base_hit = key in base_range
        base_pct = self._range_pct(base_range)
        target_pct = self._apply_range_nudges(base_pct, nudges, self.config.name, ceiling)
        candidates = frozenset(expansion_range or frozenset()) - frozenset(base_range or frozenset())
        candidate_pct = self._range_pct(candidates)
        admit_freq = 0.0
        nudge_pp = max(0.0, target_pct - base_pct)
        if candidate_pct > 0.0 and nudge_pp > 0.0:
            admit_freq = _clamp(nudge_pp / candidate_pct)
        expansion_hit = (
            not base_hit
            and key in candidates
            and self._freq_gate(view, f"{spot}_range_expand", admit_freq)
        )
        return base_hit or expansion_hit, {
            "base_range_hit": base_hit,
            "range_hit": base_hit or expansion_hit,
            "range_expansion_hit": expansion_hit,
            "range_base_pct": round(base_pct, 4),
            "range_after_pct": round(target_pct, 4),
            "range_nudge_pp": round(nudge_pp, 4),
            "range_admit_freq": round(admit_freq, 4),
        }

    def _rfi_pressure_spot(self, view: PlayerView, category: str, strategy_category: str) -> str:
        if strategy_category == "heads_up":
            return "btn_steal"
        tag = str(getattr(view, "position", "") or "").strip().upper()
        if category == "sb":
            return "sb_steal"
        if tag == "BTN":
            return "btn_steal"
        if tag == "CO":
            return "late_open"
        if category == "late":
            return "late_open"
        if category == "middle":
            return "middle_open"
        return "early_open"

    @staticmethod
    def _pressure_nudge(profile: str, spot: str, weight: float, family: str = "default") -> dict[str, float]:
        rel, abs_pp = _CHIPLEADER_PREFLOP.get(profile, _CHIPLEADER_PREFLOP["survival"]).get(spot, (0.0, 0.0))
        return {"rel": rel, "abs": abs_pp, "weight": _clamp(weight), "family": family}

    def _limper_pids(self, view: PlayerView) -> list[Any]:
        pids: list[Any] = []
        for entry in getattr(view, "history", None) or []:
            if not isinstance(entry, dict) or entry.get("street") != "preflop":
                continue
            if entry.get("type") == "raise":
                break
            if entry.get("type") == "call":
                pids.append(entry.get("pid"))
        return [pid for pid in pids if pid is not None]

    def _opener_pid(self, view: PlayerView) -> Any:
        for entry in getattr(view, "history", None) or []:
            if isinstance(entry, dict) and entry.get("street") == "preflop" and entry.get("type") == "raise":
                return entry.get("pid")
        return None

    def _stack_state_trace(
        self,
        view: PlayerView,
        context: dict[str, Any],
        hero_bb: float,
        eff_bb: float,
        pressure_weight: float = 0.0,
        short_weight: float | None = None,
        hu_weight: float = 0.0,
        commit_frac: float = 0.0,
    ) -> DecisionTrace:
        rank_ctx = context.get("rank_ctx") or self._rank_context(view)
        stacks = getattr(view, "stacks", None) or {}
        hero_stack = self._safe_int(stacks.get(getattr(view, "me", None), 0))
        return {
            "stack_state": {
                "bucket": rank_ctx.get("bucket"),
                "rank": rank_ctx.get("rank"),
                "players_left": rank_ctx.get("players_left"),
                "hero_stack": hero_stack,
                "hero_bb": round(float(hero_bb), 3),
                "eff_bb": round(float(eff_bb), 3),
                "m_ratio": round(self._m_ratio(view, hero_stack), 3),
                "hands_until_blind_up": self._hands_until_blind_up(view),
                "pressure_weight": round(float(pressure_weight), 4),
                "short_weight": round(self._short_weight(hero_bb) if short_weight is None else short_weight, 4),
                "heads_up_weight": round(float(hu_weight), 4),
                "commit_frac": round(float(commit_frac), 4),
            }
        }

    @staticmethod
    def _value_raise_thr(profile: str, way3: str, si: int) -> float:
        table = _VALUE_RAISE.get(profile, _VALUE_RAISE["survival"])
        base = table["hu"][si] if way3 == "hu" else table["mw"][si]
        return base + (0.03 if way3 == "4plus" else 0.0)

    @staticmethod
    def _value_bet_thr(profile: str, way3: str, si: int) -> float:
        table = _VALUE_BET.get(profile, _VALUE_BET["survival"])
        base = table["hu"][si] if way3 == "hu" else table["mw"][si]
        return base + (0.03 if way3 == "4plus" else 0.0)

    @staticmethod
    def _sizing(profile: str, category: str, way2: str, si: int) -> float:
        try:
            return _SIZING[profile][category][way2][si]
        except (KeyError, IndexError):
            return 0.6

    @staticmethod
    def _bet_total(pot: int, frac: float) -> int:
        return max(1, int(round(frac * max(0, pot))))

    @staticmethod
    def _raise_total(pot: int, to_call: int, frac: float) -> int:
        pot_after_call = max(0, pot) + max(0, to_call)
        return max(1, int(round(max(0, to_call) + frac * pot_after_call)))

    def _freq_gate(self, view: PlayerView, salt: str, freq: float) -> bool:
        """Deterministic frequency gate keyed on the exact spot (no global RNG)."""
        try:
            f = float(freq)
        except (TypeError, ValueError):
            return False
        if f <= 0.0:
            return False
        if f >= 1.0:
            return True
        return (self._stable_hash(view, salt) % 100000) / 100000.0 < f

    def _stable_hash(self, view: PlayerView, salt: str) -> int:
        """FNV-1a over stable spot features; stable across processes."""
        key = "|".join(str(part) for part in (
            getattr(view, "hand_id", None),
            getattr(view, "hole_cards", None) or [],
            getattr(view, "board", None) or [],
            getattr(view, "street", None),
            self._safe_int(getattr(view, "pot", 0)),
            self._safe_int(getattr(view, "to_call", 0)),
            salt,
        ))
        h = 2166136261
        for ch in key:
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    @staticmethod
    def _draw_mult(hole: list, board: list) -> float:
        return 1.25 if TournamentHybridBot._draw_category(hole, board) in ("flush", "oeso") else 1.0

    @staticmethod
    def _draw_category(hole: list, board: list) -> str:
        """Lightweight flush-/straight-draw detector for semi-bluff weighting."""
        try:
            hole = list(hole or [])
            cards = [
                (c[0], c[1]) for c in hole + list(board or [])
                if c and len(c) >= 2 and c[0] in _RANK_INDEX
            ]
            if len(cards) < 4:
                return "none"
            hero_suits = set(c[1] for c in hole if c and len(c) >= 2)
            suit_counts: dict[str, int] = {}
            for _, s in cards:
                suit_counts[s] = suit_counts.get(s, 0) + 1
            for s, cnt in suit_counts.items():
                if cnt == 4 and s in hero_suits:
                    return "flush"
            ranks = set(_RANK_INDEX[r] for r, _ in cards)
            if 12 in ranks:  # Ace also plays low for the wheel.
                ranks.add(-1)
            oeso = False
            gutshot = False
            for low in range(-1, 9):
                present = sum(1 for k in range(5) if (low + k) in ranks)
                if present < 4:
                    continue
                run4 = any(all((start + k) in ranks for k in range(4))
                           for start in (low, low + 1))
                if run4:
                    oeso = True
                else:
                    gutshot = True
            if oeso:
                return "oeso"
            if gutshot:
                return "gutshot"
            return "none"
        except Exception:
            return "none"

    @staticmethod
    def _board_bluff_mult(board: list, n_opp: int) -> float:
        """Cut pure-bluff frequency on wet boards multiway."""
        try:
            if n_opp < 2:
                return 1.0
            cards = [c for c in (board or []) if c and len(c) >= 2 and c[0] in _RANK_INDEX]
            suit_counts: dict[str, int] = {}
            for _, s in cards:
                suit_counts[s] = suit_counts.get(s, 0) + 1
            monotone = any(cnt >= 3 for cnt in suit_counts.values())
            ranks = sorted(set(_RANK_INDEX[r] for r, _ in cards))
            connected = len(ranks) >= 3 and (ranks[-1] - ranks[0]) <= 4
            return 0.5 if (monotone or connected) else 1.0
        except Exception:
            return 1.0

    @staticmethod
    def _legal_specs(view: PlayerView) -> list[dict[str, Any]]:
        legal = getattr(view, "legal_actions", None) or []
        return [spec for spec in legal if isinstance(spec, dict)]

    @classmethod
    def _legal_types(cls, view: PlayerView) -> set[str]:
        return {
            spec["type"]
            for spec in cls._legal_specs(view)
            if isinstance(spec.get("type"), str)
        }

    @classmethod
    def _safe_passive(cls, view: PlayerView) -> Action:
        legal = cls._legal_types(view)
        for action_type in ("check", "call", "fold"):
            if action_type in legal:
                return Action(action_type)
        return Action("fold")

    @classmethod
    def _safe_aggressive(cls, view: PlayerView, target_total: Any) -> Action:
        spec = cls._first_aggressive_spec(view)
        if spec is None:
            return cls._safe_passive(view)
        bounds = cls._bounds(spec)
        if bounds is None:
            return cls._safe_passive(view)

        lo, hi = bounds
        try:
            target = int(target_total)
        except (TypeError, ValueError):
            target = lo
        clamped = max(lo, min(hi, target))
        return Action(str(spec["type"]), clamped)

    @staticmethod
    def _is_all_in(spec: dict[str, Any], amount: Any) -> bool:
        if not isinstance(spec, dict):
            return False
        if bool(spec.get("all_in")):
            return True
        try:
            return int(amount) >= int(spec["max"])
        except (KeyError, TypeError, ValueError):
            return False

    @classmethod
    def _action_is_all_in(cls, view: PlayerView, action: Action) -> bool:
        action_type = getattr(action, "type", None)
        if action_type == "call":
            stacks = getattr(view, "stacks", None) or {}
            hero = getattr(view, "me", None)
            hero_stack = cls._safe_int(stacks.get(hero, 0))
            to_call = cls._safe_int(getattr(view, "to_call", 0))
            return hero_stack > 0 and to_call >= hero_stack
        if action_type in ("bet", "raise"):
            spec = cls._spec_for_type(view, action_type)
            return cls._is_all_in(spec, getattr(action, "amount", None))
        return False

    @classmethod
    def _sanitize(cls, view: PlayerView, action: Action) -> Action:
        """Final guard that mirrors engine legality without relying on it."""
        legal_types = cls._legal_types(view)
        action_type = getattr(action, "type", None)

        if action_type in ("bet", "raise") and action_type in legal_types:
            spec = cls._spec_for_type(view, action_type)
            bounds = cls._bounds(spec)
            if bounds is None:
                return cls._fallback_from_legal(legal_types)

            lo, hi = bounds
            raw_amount = getattr(action, "amount", None)
            if raw_amount is None:
                amount = lo
            else:
                try:
                    amount = int(raw_amount)
                except (TypeError, ValueError):
                    amount = lo
            return Action(action_type, max(lo, min(hi, amount)))

        if action_type in legal_types:
            return Action(str(action_type))

        return cls._fallback_from_legal(legal_types)

    @staticmethod
    def _fallback_from_legal(legal_types: set[str]) -> Action:
        for action_type in ("call", "check", "fold"):
            if action_type in legal_types:
                return Action(action_type)
        return Action("fold")

    def _context(self, view: PlayerView) -> dict[str, Any]:
        stacks = getattr(view, "stacks", None) or {}
        opponents = list(getattr(view, "opponents", None) or [])
        hero = getattr(view, "me", None)
        hero_stack = self._safe_int(stacks.get(hero, 0))
        opponent_stacks = [
            self._safe_int(stacks.get(pid, 0))
            for pid in opponents
            if self._safe_int(stacks.get(pid, 0)) > 0
        ]
        effective_stack = (
            min(hero_stack, max(opponent_stacks))
            if opponent_stacks
            else hero_stack
        )
        rank_ctx = self._rank_context(view)
        bb = max(1, self._infer_big_blind(view))
        return {
            "players_remaining": max(1, self._safe_int(rank_ctx.get("players_left")) or (1 + len(opponents))),
            "position": getattr(view, "position", None),
            "hero_stack": hero_stack,
            "effective_stack": effective_stack,
            "hero_bb": hero_stack / bb,
            "rank_ctx": rank_ctx,
        }

    @classmethod
    def _table_size(cls, view: PlayerView, context: dict[str, Any] | None = None) -> int:
        seat_indices = getattr(view, "seat_indices", None) or {}
        if isinstance(seat_indices, dict) and seat_indices:
            return max(2, len(seat_indices))
        stacks = getattr(view, "stacks", None) or {}
        if isinstance(stacks, dict) and stacks:
            return max(2, len(stacks))
        if context is not None:
            return max(2, cls._safe_int(context.get("players_remaining")) or 2)
        return max(2, 1 + len(getattr(view, "opponents", None) or []))

    @classmethod
    def _first_aggressive_spec(cls, view: PlayerView) -> dict[str, Any] | None:
        for spec in cls._legal_specs(view):
            if spec.get("type") in ("bet", "raise") and cls._bounds(spec) is not None:
                return spec
        return None

    @classmethod
    def _spec_for_type(cls, view: PlayerView, action_type: str) -> dict[str, Any] | None:
        for spec in cls._legal_specs(view):
            if spec.get("type") == action_type:
                return spec
        return None

    @staticmethod
    def _bounds(spec: dict[str, Any] | None) -> tuple[int, int] | None:
        if not isinstance(spec, dict):
            return None
        if spec.get("type") not in ("bet", "raise"):
            return None
        try:
            lo = int(spec["min"])
            hi = int(spec["max"])
        except (KeyError, TypeError, ValueError):
            return None
        if hi < lo:
            return None
        return lo, hi

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _infer_big_blind(self, view: PlayerView) -> int:
        hand_id = getattr(view, "hand_id", None)
        if hand_id is not None and hand_id in self._bb_cache:
            return self._bb_cache[hand_id]

        bb = self._infer_big_blind_uncached(view)
        bb = max(1, int(bb or 10))
        if hand_id is not None:
            self._bb_cache[hand_id] = bb
        return bb

    def _infer_big_blind_uncached(self, view: PlayerView) -> int:
        to_call = self._safe_int(getattr(view, "to_call", 0))
        min_raise = self._safe_int(getattr(view, "min_raise", 0))
        if to_call == 0 and min_raise > 0:
            return min_raise

        raises = self._count_raises(view)
        limpers = self._count_limpers(view)
        raise_spec = self._spec_for_type(view, "raise")
        raise_bounds = self._bounds(raise_spec)
        if raises == 0 and limpers == 0 and raise_bounds is not None:
            return max(1, raise_bounds[0] // 2)

        bet_spec = self._spec_for_type(view, "bet")
        bet_bounds = self._bounds(bet_spec)
        if getattr(view, "street", None) == "preflop" and bet_bounds is not None:
            return max(1, bet_bounds[0])

        for entry in getattr(view, "history", None) or []:
            if not isinstance(entry, dict) or entry.get("street") != "preflop":
                continue
            to_call_before = self._safe_int(entry.get("to_call_before"))
            if to_call_before > 0:
                return to_call_before
            pot_before = self._safe_int(entry.get("pot_before"))
            if pot_before > 0:
                return max(1, int(round(pot_before / 1.5)))
            break

        pot = self._safe_int(getattr(view, "pot", 0))
        if pot > 0:
            return max(1, int(round(pot / 1.5)))
        return 10

    @classmethod
    def _classify_position(cls, pos: Any, n: int) -> str:
        tag = str(pos or "").strip().upper().replace("UTG1", "UTG+1")
        if tag == "SB":
            return "sb"
        if tag == "BB":
            return "bb"
        if n == 2 and tag == "BTN":
            return "late"
        if tag in {"BTN", "CO", "HJ"}:
            return "late"
        if tag == "MP" and n == 6:
            return "late"
        if tag in {"MP", "LJ"}:
            return "middle"
        if tag in {"UTG", "UTG+1"}:
            return "early"
        return "middle"

    @staticmethod
    def _positions_for_n(n: int) -> list[str]:
        if n == 2:
            return ["BTN", "BB"]
        return ["BTN", "SB", "BB", "UTG", "UTG+1", "MP", "LJ", "HJ", "CO"][:n]

    @classmethod
    def _count_raises(cls, view: PlayerView) -> int:
        return sum(
            1
            for entry in (getattr(view, "history", None) or [])
            if isinstance(entry, dict)
            and entry.get("street") == "preflop"
            and entry.get("type") == "raise"
        )

    @classmethod
    def _last_preflop_raiser(cls, view: PlayerView) -> Any:
        raiser = None
        for entry in getattr(view, "history", None) or []:
            if (
                isinstance(entry, dict)
                and entry.get("street") == "preflop"
                and entry.get("type") == "raise"
            ):
                raiser = entry.get("pid")
        return raiser

    @classmethod
    def _count_limpers(cls, view: PlayerView) -> int:
        count = 0
        for entry in getattr(view, "history", None) or []:
            if not isinstance(entry, dict) or entry.get("street") != "preflop":
                continue
            if entry.get("type") == "raise":
                break
            if entry.get("type") == "call":
                count += 1
        return count

    @classmethod
    def _last_preflop_raise_to(cls, view: PlayerView) -> int:
        amount = 0
        for entry in getattr(view, "history", None) or []:
            if not isinstance(entry, dict):
                continue
            if entry.get("street") == "preflop" and entry.get("type") == "raise":
                amount = cls._safe_int(entry.get("amount"))
        return amount

    def _eff_bb(self, view: PlayerView, context: dict[str, Any]) -> int:
        bb = max(1, self._infer_big_blind(view))
        return max(0, self._safe_int(context.get("effective_stack")) // bb)

    def _hand_key_from_view(self, view: PlayerView) -> str | None:
        cards = getattr(view, "hole_cards", None) or []
        if len(cards) < 2:
            return None
        try:
            r1, s1 = cards[0][0], cards[0][1]
            r2, s2 = cards[1][0], cards[1][1]
            if r1 not in _RANK_INDEX or r2 not in _RANK_INDEX:
                return None
            return _hand_key(r1, r2, s1 == s2)
        except (TypeError, IndexError):
            return None

    def _strategy_category(self, category: str, n_players: int) -> str:
        if n_players == 2:
            return "heads_up"
        return category

    def _shove_threshold(self, category: str, n_players: int) -> int:
        thresholds = self.config.short_stack_bb_thresholds or {}
        key = "heads_up" if n_players == 2 else category
        fallback = self.config.short_stack_bb_threshold
        if fallback is None:
            fallback = 10
        return int(thresholds.get(key, thresholds.get("middle", fallback)))

    def _facing_all_in(self, view: PlayerView, raises: int) -> bool:
        return self._all_in_aggressor(view, raises) is not None

    def _all_in_aggressor(self, view: PlayerView, raises: int) -> Any:
        if raises <= 0 or self._safe_int(getattr(view, "to_call", 0)) <= 0:
            return None
        last_raiser = self._last_preflop_raiser(view)
        all_in_opponents = set(getattr(view, "all_in_opponents", None) or [])
        if last_raiser is not None and last_raiser in all_in_opponents:
            return last_raiser
        if "raise" not in self._legal_types(view):
            return last_raiser
        return None

    def _all_in_aggressor_category(
        self,
        view: PlayerView,
        n_players: int,
        raises: int,
    ) -> str:
        aggressor = self._all_in_aggressor(view, raises)
        pos = self._position_for_pid(view, aggressor, n_players)
        return self._strategy_category(self._classify_position(pos, n_players), n_players)

    def _opener_category(self, view: PlayerView, n_players: int) -> str:
        opener = None
        for entry in getattr(view, "history", None) or []:
            if isinstance(entry, dict) and entry.get("street") == "preflop" and entry.get("type") == "raise":
                opener = entry.get("pid")
                break
        pos = self._position_for_pid(view, opener, n_players)
        return self._strategy_category(self._classify_position(pos, n_players), n_players)

    def _position_for_pid(self, view: PlayerView, pid: Any, n_players: int) -> str | None:
        if pid is None:
            return None
        raw = str(pid).strip().upper()
        if raw in _POSITION_TAGS:
            return raw.replace("UTG1", "UTG+1")

        order = self._seat_order(view)
        if not order:
            order = list((getattr(view, "stacks", None) or {}).keys())
        if pid in order:
            idx = order.index(pid)
            positions = self._positions_for_n(max(n_players, len(order)))
            if idx < len(positions):
                return positions[idx]
        return None

    @staticmethod
    def _seat_order(view: PlayerView) -> list[Any]:
        seat_indices = getattr(view, "seat_indices", None) or {}
        if not isinstance(seat_indices, dict):
            return []
        return [
            pid for pid, _ in sorted(
                seat_indices.items(),
                key=lambda item: item[1],
            )
        ]

    def _can_open_from_here(self, view: PlayerView, category: str, n_players: int, bb: int) -> bool:
        legal = self._legal_types(view)
        if "raise" not in legal and "bet" not in legal:
            return False
        if category == "bb" and n_players > 2 and self._safe_int(getattr(view, "to_call", 0)) == 0:
            return False
        return self._safe_int(getattr(view, "to_call", 0)) <= max(1, bb)

    def _jam_total(self, view: PlayerView) -> int:
        spec = self._first_aggressive_spec(view)
        bounds = self._bounds(spec)
        if bounds is not None:
            return bounds[1]
        stacks = getattr(view, "stacks", None) or {}
        return self._safe_int(stacks.get(getattr(view, "me", None), 0))

    @staticmethod
    def _open_total(bb: int, eff_bb: int, category: str, n_players: int) -> int:
        if category == "sb" or (n_players == 2 and category == "late"):
            mult = 2.7 if eff_bb >= 25 else 2.3
        elif eff_bb >= 40:
            mult = 2.2 if category == "late" else 2.5
        elif eff_bb >= 25:
            mult = 2.2
        else:
            mult = 2.0
        return max(1, int(round(bb * mult)))

    @staticmethod
    def _iso_total(bb: int, eff_bb: int, category: str, limpers: int) -> int:
        oop = category in ("sb", "bb")
        if 20 <= eff_bb <= 30:
            base = 3.0
        else:
            base = 4.5 if oop else 3.5
        return max(1, int(round(bb * (base + max(1, limpers)))))

    def _threebet_total(self, view: PlayerView, bb: int, opener_cat: str, hero_cat: str) -> int:
        open_to = self._last_preflop_raise_to(view) or max(bb * 2, self._safe_int(getattr(view, "to_call", 0)) + bb)
        cold_callers = self._cold_callers_after_first_raise(view)
        oop = hero_cat in ("sb", "bb") or opener_cat in ("late", "sb", "heads_up")
        mult = 3.8 if oop else 3.0
        return max(1, int(round(open_to * (mult + cold_callers))))

    def _fourbet_total(self, view: PlayerView) -> int:
        last = self._last_preflop_raise_to(view)
        if last <= 0:
            spec = self._first_aggressive_spec(view)
            bounds = self._bounds(spec)
            last = bounds[0] if bounds else self._safe_int(getattr(view, "to_call", 0))
        return max(1, int(round(last * 2.3)))

    @classmethod
    def _cold_callers_after_first_raise(cls, view: PlayerView) -> int:
        seen_raise = False
        count = 0
        for entry in getattr(view, "history", None) or []:
            if not isinstance(entry, dict) or entry.get("street") != "preflop":
                continue
            if entry.get("type") == "raise":
                seen_raise = True
                continue
            if seen_raise and entry.get("type") == "call":
                count += 1
        return count

    def _current_contribution(self, view: PlayerView) -> int:
        stacks = getattr(view, "stacks", None) or {}
        hero_stack = self._safe_int(stacks.get(getattr(view, "me", None), 0))
        spec = self._first_aggressive_spec(view)
        bounds = self._bounds(spec)
        if bounds is None:
            return 0
        return max(0, bounds[1] - hero_stack)

    def _commits_fraction(self, view: PlayerView, target_total: int, fraction: float) -> bool:
        stacks = getattr(view, "stacks", None) or {}
        hero_stack = self._safe_int(stacks.get(getattr(view, "me", None), 0))
        contributed = self._current_contribution(view)
        starting = max(1, hero_stack + contributed)
        added = max(0, int(target_total) - contributed)
        return added / starting >= fraction

    def _is_small_open(self, view: PlayerView, bb: int) -> bool:
        open_to = self._last_preflop_raise_to(view)
        if open_to <= 0:
            open_to = self._safe_int(getattr(view, "to_call", 0)) + self._current_contribution(view)
        return open_to <= int(round(2.5 * max(1, bb)))

    @classmethod
    def _fold_or_check(cls, view: PlayerView) -> Action:
        legal = cls._legal_types(view)
        if "fold" in legal:
            return Action("fold")
        if "check" in legal:
            return Action("check")
        return cls._safe_passive(view)

    @classmethod
    def _check_or_fold(cls, view: PlayerView) -> Action:
        legal = cls._legal_types(view)
        if "check" in legal:
            return Action("check")
        if "fold" in legal:
            return Action("fold")
        return cls._safe_passive(view)

    @classmethod
    def _call_or_fold(cls, view: PlayerView) -> Action:
        legal = cls._legal_types(view)
        if "call" in legal:
            return Action("call")
        if "check" in legal:
            return Action("check")
        if "fold" in legal:
            return Action("fold")
        return cls._safe_passive(view)

    @classmethod
    def _call_or_check(cls, view: PlayerView) -> Action:
        legal = cls._legal_types(view)
        if "call" in legal:
            return Action("call")
        if "check" in legal:
            return Action("check")
        if "fold" in legal:
            return Action("fold")
        return cls._safe_passive(view)
