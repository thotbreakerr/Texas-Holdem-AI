"""Deterministic tournament hybrid bot.

Phase 2 adds a real preflop strategy while preserving the Phase 1 safety
contract: all aggressive sizing goes through legal-action specs and every
decision is sanitized before it leaves ``act``.  Postflop remains passive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from core.bot_api import Action, PlayerView


DecisionTrace = dict[str, Any]

_RANKS = "23456789TJQKA"
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

    # Opponent module stubs.  Phase 5 will add memory and exploitation hooks.
    opponent_model_key: Optional[str] = None
    exploit_adjustment: float = 0.0
    memory_hands: Optional[int] = None


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
        self._pending_preflop_trace: DecisionTrace = {}

    def reset_memory(self):
        """Tournament boundary hook; real opponent memory arrives in Phase 5."""
        self.last_decision = None
        self._bb_cache.clear()
        self._pending_preflop_trace = {}
        return None

    def act(self, view: PlayerView) -> Action:
        """Return a safe action and populate ``last_decision``."""
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
        self.last_decision = trace
        return action

    @staticmethod
    def _empty_preflop_trace() -> DecisionTrace:
        return {
            "position_category": None,
            "inferred_bb": None,
            "eff_bb": None,
            "band": None,
            "raises_faced": None,
            "branch": None,
            "range_hit": False,
        }

    def _trace_preflop(self, **updates: Any) -> None:
        self._pending_preflop_trace.update(updates)

    def _preflop_action(self, view: PlayerView, context: dict[str, Any]) -> Action | None:
        key = self._hand_key_from_view(view)
        bb = self._infer_big_blind(view)
        n_players = self._table_size(view, context)
        position = getattr(view, "position", None)
        category = self._classify_position(position, n_players)
        strategy_category = self._strategy_category(category, n_players)
        eff_bb = self._eff_bb(view, context)
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
            band=band,
            raises_faced=raises,
            branch="no_cards",
            range_hit=False,
            reason="missing or invalid hole cards",
        )
        if not key:
            return self._fold_or_check(view)

        if band == "short":
            return self._short_stack_action(
                view, key, strategy_category, n_players, raises, limpers
            )
        if band == "medium":
            return self._medium_stack_action(
                view, key, strategy_category, category, n_players, raises, limpers, bb
            )
        return self._deep_stack_action(
            view, key, strategy_category, category, n_players, raises, limpers, bb, eff_bb
        )

    def _short_stack_action(
        self,
        view: PlayerView,
        key: str,
        strategy_category: str,
        n_players: int,
        raises: int,
        limpers: int,
    ) -> Action:
        if self._facing_all_in(view, raises):
            all_in_cat = self._all_in_aggressor_category(view, n_players, raises)
            call_range = _CALL_SHOVE_RANGES[self.config.name].get(all_in_cat, frozenset())
            hit = key in call_range
            self._trace_preflop(branch=f"short_call_off_vs_{all_in_cat}", range_hit=hit,
                                reason="short stack facing all-in")
            return self._call_or_fold(view) if hit else self._fold_or_check(view)

        shove_range = _SHORT_SHOVE_RANGES[self.config.name].get(strategy_category, frozenset())
        hit = key in shove_range
        branch = "short_shove" if raises or limpers else "short_open_shove"
        self._trace_preflop(
            branch=branch,
            range_hit=hit,
            target_total=self._jam_total(view) if hit else None,
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
    ) -> Action:
        if self._facing_all_in(view, raises):
            all_in_cat = self._all_in_aggressor_category(view, n_players, raises)
            call_range = _CALL_SHOVE_RANGES[self.config.name].get(all_in_cat, frozenset())
            hit = key in call_range
            self._trace_preflop(branch=f"medium_call_off_vs_{all_in_cat}", range_hit=hit,
                                reason="medium stack facing all-in")
            return self._call_or_fold(view) if hit else self._fold_or_check(view)

        if raises >= 2:
            hit = key in _VS_3BET_VALUE
            self._trace_preflop(
                branch="medium_vs_3bet_jam",
                range_hit=hit,
                target_total=self._jam_total(view) if hit else None,
                reason="medium stack never 3bet-then-folds",
            )
            return self._safe_aggressive(view, self._jam_total(view)) if hit else self._fold_or_check(view)

        if raises == 1:
            opener_cat = self._opener_category(view, n_players)
            resteal_range = _RESTEAL_RANGES[self.config.name].get(opener_cat, _RESTEAL_RANGES[self.config.name]["middle"])
            hit = key in resteal_range
            if opener_cat in ("early", "middle"):
                hit = key in _RESTEAL_RANGES[self.config.name][opener_cat]
            self._trace_preflop(
                branch=f"medium_resteal_vs_{opener_cat}",
                range_hit=hit,
                target_total=self._jam_total(view) if hit else None,
                reason="medium stack resteal jam or fold",
            )
            return self._safe_aggressive(view, self._jam_total(view)) if hit else self._fold_or_check(view)

        if limpers:
            iso_range = _ISO_VALUE_SHORT if self._eff_bb(view, self._context(view)) <= 15 else _ISO_VALUE
            hit = key in iso_range
            self._trace_preflop(
                branch="medium_iso_limpers",
                range_hit=hit,
                target_total=self._jam_total(view) if hit and self._eff_bb(view, self._context(view)) <= 15 else None,
                reason="medium stack iso against limpers",
            )
            if not hit:
                return self._check_or_fold(view) if category == "bb" else self._fold_or_check(view)
            if self._eff_bb(view, self._context(view)) <= 15:
                return self._safe_aggressive(view, self._jam_total(view))
            target = self._iso_total(bb, self._eff_bb(view, self._context(view)), category, limpers)
            self._trace_preflop(target_total=target)
            return self._safe_aggressive(view, target)

        open_jam_range = _MEDIUM_OPEN_JAM.get(self.config.name, {}).get(strategy_category, frozenset())
        can_open_jam = strategy_category in ("late", "sb", "heads_up") and key in open_jam_range
        if can_open_jam:
            self._trace_preflop(
                branch="medium_open_jam",
                range_hit=True,
                target_total=self._jam_total(view),
                reason="medium late-position open jam",
            )
            return self._safe_aggressive(view, self._jam_total(view))

        rfi_range = _RFI_RANGES[self.config.name].get(strategy_category, frozenset())
        hit = key in rfi_range
        self._trace_preflop(
            branch="medium_rfi",
            range_hit=hit,
            reason="medium stack open/fold",
        )
        if hit and self._can_open_from_here(view, category, n_players, bb):
            target = self._open_total(bb, self._eff_bb(view, self._context(view)), category, n_players)
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
    ) -> Action:
        if raises >= 3:
            core_hit = key in _VS_4BET_CORE
            conditional_hit = eff_bb <= 30 and key in _VS_4BET_CONDITIONAL
            hit = core_hit or conditional_hit
            target = self._jam_total(view) if hit and eff_bb <= 50 else self._fourbet_total(view)
            self._trace_preflop(
                branch="deep_vs_4bet",
                range_hit=hit,
                target_total=target if hit else None,
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
                    target_total=target,
                    reason="premium 4bet or jam versus 3bet",
                )
                return self._safe_aggressive(view, target)
            flat_hit = key in _VS_3BET_FLAT and eff_bb >= 40 and category != "sb"
            self._trace_preflop(
                branch="deep_vs_3bet_flat" if flat_hit else "deep_vs_3bet_fold",
                range_hit=flat_hit,
                reason="flat narrow range versus 3bet",
            )
            return self._call_or_fold(view) if flat_hit else self._fold_or_check(view)

        if raises == 1:
            opener_cat = self._opener_category(view, n_players)
            value_range = _VALUE_3BET_RANGES[self.config.name].get(opener_cat, _PREMIUM_3BET)
            if key in value_range:
                target = self._threebet_total(view, bb, opener_cat, category)
                if eff_bb <= 30 or self._commits_fraction(view, target, 0.30):
                    target = self._jam_total(view)
                self._trace_preflop(
                    branch=f"deep_value_3bet_vs_{opener_cat}",
                    range_hit=True,
                    target_total=target,
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
                reason="flat call-set versus open",
            )
            return self._call_or_fold(view) if flat_hit else self._fold_or_check(view)

        if limpers:
            return self._limped_pot_action(view, key, category, bb, limpers, eff_bb)

        rfi_range = _RFI_RANGES[self.config.name].get(strategy_category, frozenset())
        hit = key in rfi_range
        self._trace_preflop(
            branch="deep_rfi",
            range_hit=hit,
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
    ) -> Action:
        iso_hit = key in _ISO_VALUE
        if eff_bb <= 15:
            iso_hit = key in _ISO_VALUE_SHORT
        if iso_hit:
            target = self._jam_total(view) if eff_bb <= 15 else self._iso_total(bb, eff_bb, category, limpers)
            self._trace_preflop(
                branch="deep_iso_limpers",
                range_hit=True,
                target_total=target,
                reason="value iso versus limpers",
            )
            return self._safe_aggressive(view, target)

        overlimp_hit = key in _OVERLIMP_SPECULATIVE and category in ("late", "middle", "sb")
        self._trace_preflop(
            branch="deep_overlimp" if overlimp_hit else "deep_limped_fold_or_check",
            range_hit=overlimp_hit,
            reason="speculative overlimp or blind check",
        )
        if overlimp_hit:
            return self._call_or_check(view)
        if category == "bb":
            return self._check_or_fold(view)
        return self._fold_or_check(view)

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

    @classmethod
    def _context(cls, view: PlayerView) -> dict[str, Any]:
        stacks = getattr(view, "stacks", None) or {}
        opponents = list(getattr(view, "opponents", None) or [])
        hero = getattr(view, "me", None)
        hero_stack = cls._safe_int(stacks.get(hero, 0))
        opponent_stacks = [
            cls._safe_int(stacks.get(pid, 0))
            for pid in opponents
        ]
        effective_stack = (
            min(hero_stack, max(opponent_stacks))
            if opponent_stacks
            else hero_stack
        )
        return {
            "players_remaining": 1 + len(opponents),
            "position": getattr(view, "position", None),
            "effective_stack": effective_stack,
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
