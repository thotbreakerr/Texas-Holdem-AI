"""Phase 1 skeleton for the final tournament bot.

The implementation is intentionally conservative: no poker strategy lives here
yet.  This file only provides profile selection, legal-action safety helpers,
and a cheap decision trace for later phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from core.bot_api import Action, PlayerView


DecisionTrace = dict[str, Any]


@dataclass(frozen=True)
class ProfileConfig:
    name: str

    # Preflop module stubs.  Phase 2 will consume range keys and adjustments.
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

    # Tournament module stubs.  Phase 4 will add stack-depth and ICM routing.
    survival_pressure: float = 0.0
    short_stack_bb_threshold: Optional[int] = None
    bubble_factor: Optional[float] = None

    # Opponent module stubs.  Phase 5 will add memory and exploitation hooks.
    opponent_model_key: Optional[str] = None
    exploit_adjustment: float = 0.0
    memory_hands: Optional[int] = None


SURVIVAL_CONFIG = ProfileConfig(
    name="survival",
    max_risk_fraction=0.35,
    survival_pressure=1.0,
)

AGGRO_CONFIG = ProfileConfig(
    name="aggro",
    preflop_open_adjustment=0.10,
    postflop_probe_frequency=0.10,
    max_risk_fraction=0.50,
    survival_pressure=0.5,
    exploit_adjustment=0.10,
)

_PROFILE_CONFIGS = {
    "survival": SURVIVAL_CONFIG,
    "aggro": AGGRO_CONFIG,
}


class TournamentHybridBot:
    """Legal-by-construction tournament bot skeleton.

    Phase 1 deliberately returns a passive action from ``act``.  Aggressive
    helpers exist now so future strategy modules can ask for a bet/raise target
    without touching the engine's sanitizer path.
    """

    def __init__(self, profile: str = "survival"):
        key = str(profile).strip().lower()
        if key not in _PROFILE_CONFIGS:
            names = ", ".join(sorted(_PROFILE_CONFIGS))
            raise ValueError(f"unknown TournamentHybridBot profile {profile!r}; "
                             f"expected one of: {names}")
        self.config = _PROFILE_CONFIGS[key]
        self.profile = self.config.name
        self.last_decision: DecisionTrace | None = None

    def reset_memory(self):
        """Tournament boundary hook; real memory arrives in Phase 5."""
        self.last_decision = None
        return None

    def act(self, view: PlayerView) -> Action:
        """Return a safe passive action and populate ``last_decision``."""
        context = self._context(view)
        legal_types = self._legal_types(view)
        proposed = self._safe_passive(view)
        action = self._sanitize(view, proposed)
        path = "passive"
        reason = "phase 1 passive skeleton"
        if legal_types and action.type not in legal_types:
            path = "fallback"
            reason = "degenerate legal_actions had no passive action"
        elif action.type != proposed.type or action.amount != proposed.amount:
            path = "fallback"
            reason = "final sanitizer adjusted passive action"
        self.last_decision = {
            "profile": self.config.name,
            "legal_types": tuple(sorted(legal_types)),
            "path": path,
            "target_total": None,
            "clamped_amount": action.amount,
            "all_in_detected": self._action_is_all_in(view, action),
            "reason": reason,
            "context": context,
        }
        return action

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
        # Phase 4: infer blind/ante context here only if the public view can
        # support it.  Bet sizing must come from the active legal-action spec.
        return {
            "players_remaining": 1 + len(opponents),
            "position": getattr(view, "position", None),
            "effective_stack": effective_stack,
        }

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
