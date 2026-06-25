"""Phase 7 stress-opponent archetypes.

These bots are deliberate opponent-field probes for the hybrid bot's Phase-5
read machinery.  They are frequency stressors, not realistic player models.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from core.bot_api import Action, PlayerView, acting_opponents_for


_POSTFLOP = ("flop", "turn", "river")
_RANK_ORDER = "23456789TJQKA"
_RANK_INDEX = {rank: idx for idx, rank in enumerate(_RANK_ORDER)}
_HERO_PID = "HERO"


@dataclass(frozen=True)
class ArchetypeConfig:
    name: str
    policy_name: str
    knobs: Mapping[str, Any]


def _cfg(name: str, policy_name: str, **knobs: Any) -> ArchetypeConfig:
    return ArchetypeConfig(name=name, policy_name=policy_name, knobs=MappingProxyType(dict(knobs)))


ARCHETYPE_CONFIGS: dict[str, ArchetypeConfig] = {
    "maniac_trigger": _cfg(
        "maniac_trigger",
        "maniac",
        jam_freq=0.65,
        preflop_raise_freq=0.56,
        preflop_call_freq=0.32,
        postflop_probe_freq=0.42,
        anti_bust_bb=9,
    ),
    "maniac_mixed": _cfg(
        "maniac_mixed",
        "maniac",
        jam_freq=0.43,
        preflop_raise_freq=0.46,
        preflop_call_freq=0.34,
        postflop_probe_freq=0.34,
        anti_bust_bb=9,
    ),
    "maniac": _cfg(
        "maniac_trigger",
        "maniac",
        jam_freq=0.65,
        preflop_raise_freq=0.56,
        preflop_call_freq=0.32,
        postflop_probe_freq=0.42,
        anti_bust_bb=9,
    ),
    "overbet_merchant": _cfg(
        "overbet_merchant",
        "overbet",
        overbet_freq=0.64,
        late_jam_freq=0.14,
        preflop_raise_freq=0.34,
        preflop_call_freq=0.24,
        keepbehind_bb=3,
    ),
    "calling_station": _cfg(
        "calling_station",
        "station",
        preflop_call_freq=0.58,
        preflop_raise_freq=0.03,
        pressure_call_freq=0.84,
        allin_call_freq=0.44,
    ),
    "nit": _cfg(
        "nit",
        "nit",
        pressure_fold_freq=0.88,
        premium_raise_freq=0.72,
        premium_call_freq=0.62,
    ),
    "folder": _cfg(
        "nit",
        "nit",
        pressure_fold_freq=0.88,
        premium_raise_freq=0.72,
        premium_call_freq=0.62,
    ),
    "loose_passive": _cfg(
        "loose_passive",
        "loose_passive",
        preflop_call_freq=0.52,
        preflop_raise_freq=0.05,
        pressure_call_freq=0.56,
        check_freq=0.90,
    ),
    "minraise": _cfg(
        "minraise",
        "minraise",
        preflop_raise_freq=0.30,
        preflop_call_freq=0.24,
        postflop_minraise_freq=0.30,
        stop_below_bb=9,
    ),
    "minraiser": _cfg(
        "minraise",
        "minraise",
        preflop_raise_freq=0.30,
        preflop_call_freq=0.24,
        postflop_minraise_freq=0.30,
        stop_below_bb=9,
    ),
    "baseline_sane": _cfg(
        "baseline_sane",
        "baseline_sane",
        preflop_raise_freq=0.18,
        preflop_call_freq=0.22,
        postflop_bet_freq=0.20,
    ),
    "pressure_filler": _cfg(
        "pressure_filler",
        "pressure_filler",
        postflop_bet_freq=0.86,
        preflop_raise_freq=0.16,
        preflop_call_freq=0.26,
    ),
}


class ArchetypeBot:
    """Single wrapper around separate Phase 7 stress policies."""

    def __init__(self, preset: str = "baseline_sane"):
        key = str(preset or "").strip().lower()
        if key not in ARCHETYPE_CONFIGS:
            names = ", ".join(sorted(ARCHETYPE_CONFIGS))
            raise ValueError(f"unknown stress archetype {preset!r}; expected one of: {names}")
        self.config = ARCHETYPE_CONFIGS[key]
        self.policy = _POLICIES[self.config.policy_name]()
        self.last_decision: dict[str, Any] | None = None
        self._seen_action_keys: set[tuple[Any, int, Any, str, str]] = set()
        self._self_large_bet_count = 0
        self._telemetry = self._blank_telemetry()

    def reset_memory(self) -> None:
        self.last_decision = None
        self._seen_action_keys.clear()
        self._self_large_bet_count = 0
        self._telemetry = self._blank_telemetry()

    def act(self, view: PlayerView) -> Action:
        self._ingest_visible_history(view)
        proposed = self.policy.act(self, view)
        action = self.sanitize(view, proposed)
        self._record_action_telemetry(view, action)
        self.last_decision = {
            "archetype": self.config.name,
            "policy": self.config.policy_name,
            "street": getattr(view, "street", None),
            "action": action.type,
            "amount": action.amount,
            "decision_count_this_hand": self.decision_count_this_hand(view),
        }
        return action

    def stress_telemetry_summary(self) -> dict[str, int]:
        return dict(self._telemetry)

    @staticmethod
    def _blank_telemetry() -> dict[str, int]:
        keys = [
            "true_jam_like_count",
            "true_short_jam_like_count",
            "true_large_bet_count",
            "true_station_count",
            "true_fold_to_pressure_count",
            "true_vpip_count",
            "true_pfr_count",
            "true_minraise_count",
            "true_pressure_bet_count",
            "true_missed_jam_opportunity_count",
            "jam_opportunity_n",
            "jam_opportunity_taken_count",
            "missed_due_to_hero_already_folded_count",
            "missed_due_to_hero_already_folded_preflop_count",
            "missed_due_to_hero_already_folded_postflop_count",
            "missed_due_to_hero_already_folded_unknown_count",
        ]
        return {key: 0 for key in keys}

    def knob(self, name: str, default: Any = None) -> Any:
        return self.config.knobs.get(name, default)

    def legal_specs(self, view: PlayerView) -> list[dict[str, Any]]:
        return [spec for spec in (getattr(view, "legal_actions", None) or []) if isinstance(spec, dict)]

    def legal_types(self, view: PlayerView) -> tuple[str, ...]:
        types: list[str] = []
        for spec in self.legal_specs(view):
            action_type = spec.get("type")
            if isinstance(action_type, str) and action_type not in types:
                types.append(action_type)
        return tuple(types)

    def spec_for(self, view: PlayerView, action_type: str) -> dict[str, Any] | None:
        for spec in self.legal_specs(view):
            if spec.get("type") == action_type:
                return spec
        return None

    def bounds(self, spec: dict[str, Any] | None) -> tuple[int, int] | None:
        if not isinstance(spec, dict):
            return None
        try:
            lo = int(spec["min"])
            hi = int(spec["max"])
        except (KeyError, TypeError, ValueError):
            return None
        if lo > hi:
            return None
        return lo, hi

    def aggressive_spec(self, view: PlayerView) -> dict[str, Any] | None:
        if self.safe_int(getattr(view, "to_call", 0)) <= 0:
            return self.spec_for(view, "bet") or self.spec_for(view, "raise")
        return self.spec_for(view, "raise")

    def safe_passive(self, view: PlayerView) -> Action:
        legal = self.legal_types(view)
        for action_type in ("check", "call", "fold"):
            if action_type in legal:
                return Action(action_type)
        return Action("fold")

    def fold_or_check(self, view: PlayerView) -> Action:
        legal = self.legal_types(view)
        if self.safe_int(getattr(view, "to_call", 0)) <= 0 and "check" in legal:
            return Action("check")
        if "fold" in legal:
            return Action("fold")
        return self.safe_passive(view)

    def call_or_check(self, view: PlayerView) -> Action:
        legal = self.legal_types(view)
        if self.safe_int(getattr(view, "to_call", 0)) > 0 and "call" in legal:
            return Action("call")
        if "check" in legal:
            return Action("check")
        return self.safe_passive(view)

    def aggressive_to(self, view: PlayerView, target_total: Any) -> Action:
        spec = self.aggressive_spec(view)
        limits = self.bounds(spec)
        if spec is None or limits is None:
            return self.safe_passive(view)
        lo, hi = limits
        target = self.safe_int(target_total)
        if target <= 0:
            target = lo
        return Action(str(spec["type"]), max(lo, min(hi, target)))

    def min_aggressive(self, view: PlayerView) -> Action:
        spec = self.aggressive_spec(view)
        limits = self.bounds(spec)
        if spec is None or limits is None:
            return self.safe_passive(view)
        return Action(str(spec["type"]), limits[0])

    def shove(self, view: PlayerView) -> Action:
        spec = self.aggressive_spec(view)
        limits = self.bounds(spec)
        if spec is None or limits is None:
            return self.safe_passive(view)
        return Action(str(spec["type"]), limits[1])

    def sanitize(self, view: PlayerView, action: Action | None) -> Action:
        if action is None:
            return self.safe_passive(view)
        legal = self.legal_types(view)
        action_type = getattr(action, "type", None)
        if action_type in ("bet", "raise") and action_type in legal:
            spec = self.spec_for(view, action_type)
            limits = self.bounds(spec)
            if limits is None:
                return self.safe_passive(view)
            lo, hi = limits
            amount = self.safe_int(getattr(action, "amount", None))
            if amount <= 0:
                amount = lo
            return Action(action_type, max(lo, min(hi, amount)))
        if action_type in legal:
            if action_type == "fold" and self.safe_int(getattr(view, "to_call", 0)) <= 0 and "check" in legal:
                return Action("check")
            return Action(str(action_type))
        return self.safe_passive(view)

    def infer_bb(self, view: PlayerView) -> int:
        min_raise = self.safe_int(getattr(view, "min_raise", 0))
        if min_raise > 0:
            return max(1, min_raise)
        for entry in getattr(view, "history", None) or []:
            if not isinstance(entry, dict):
                continue
            to_call = self.safe_int(entry.get("to_call_before"))
            if to_call > 0:
                return max(1, to_call)
        pot = self.safe_int(getattr(view, "pot", 0))
        if pot > 0:
            return max(1, int(round(pot / 1.5)))
        return 10

    def stack(self, view: PlayerView, pid: Any | None = None) -> int:
        stacks = getattr(view, "stacks", None) or {}
        target = getattr(view, "me", None) if pid is None else pid
        return self.safe_int(stacks.get(target, 0))

    def effective_stack_bb(self, view: PlayerView) -> float:
        bb = max(1, self.infer_bb(view))
        hero_stack = self.stack(view)
        opp_stacks = [
            self.stack(view, pid)
            for pid in (getattr(view, "opponents", None) or [])
            if self.stack(view, pid) > 0
        ]
        if not opp_stacks:
            return hero_stack / bb
        return min(hero_stack, max(opp_stacks)) / bb

    def live_player_count(self, view: PlayerView) -> int:
        return 1 + len(acting_opponents_for(view))

    def current_contrib(self, view: PlayerView, pid: Any | None = None) -> int:
        target = getattr(view, "me", None) if pid is None else pid
        street = getattr(view, "street", None)
        contrib = 0
        for entry in getattr(view, "history", None) or []:
            if not isinstance(entry, dict) or entry.get("pid") != target or entry.get("street") != street:
                continue
            action_type = entry.get("type")
            amount = self.safe_int(entry.get("amount"))
            if action_type == "call":
                contrib += max(0, amount)
            elif action_type in ("bet", "raise"):
                contrib = max(contrib, amount)
        return contrib

    def decision_count_this_hand(self, view: PlayerView) -> int:
        me = getattr(view, "me", None)
        count = 0
        for entry in getattr(view, "history", None) or []:
            if isinstance(entry, dict) and entry.get("pid") == me:
                count += 1
        return count

    def stable_uniform(self, view: PlayerView, decision_kind: str) -> float:
        key = (
            getattr(view, "hand_id", None),
            self._stable_cards(getattr(view, "hole_cards", None) or []),
            self._stable_cards(getattr(view, "board", None) or []),
            getattr(view, "street", None),
            self.safe_int(getattr(view, "pot", 0)),
            self.safe_int(getattr(view, "to_call", 0)),
            getattr(view, "me", None),
            self.decision_count_this_hand(view),
            decision_kind,
        )
        h = 2166136261
        for ch in self._stable_value(key):
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        return h / 4294967296.0

    def bernoulli(self, view: PlayerView, decision_kind: str, freq: Any) -> bool:
        try:
            f = float(freq)
        except (TypeError, ValueError):
            return False
        if f <= 0.0:
            return False
        if f >= 1.0:
            return True
        return self.stable_uniform(view, decision_kind) < f

    def hand_class(self, view: PlayerView) -> str:
        cards = getattr(view, "hole_cards", None) or []
        if len(cards) < 2:
            return "trash"
        try:
            r1, s1 = str(cards[0][0]), str(cards[0][1])
            r2, s2 = str(cards[1][0]), str(cards[1][1])
        except (TypeError, IndexError):
            return "trash"
        if r1 not in _RANK_INDEX or r2 not in _RANK_INDEX:
            return "trash"
        hi = max(_RANK_INDEX[r1], _RANK_INDEX[r2])
        lo = min(_RANK_INDEX[r1], _RANK_INDEX[r2])
        pair = r1 == r2
        suited = s1 == s2
        gap = hi - lo
        if pair and hi >= _RANK_INDEX["T"]:
            return "premium"
        if {r1, r2} in ({"A", "K"}, {"A", "Q"}):
            return "premium"
        if pair or hi >= _RANK_INDEX["A"] or (hi >= _RANK_INDEX["K"] and lo >= _RANK_INDEX["T"]):
            return "playable"
        if suited and hi >= _RANK_INDEX["T"] and gap <= 4:
            return "playable"
        if suited and gap <= 2 and lo >= _RANK_INDEX["5"]:
            return "speculative"
        return "trash"

    def note_jam_opportunity(self, view: PlayerView, taken: bool) -> None:
        self._telemetry["jam_opportunity_n"] += 1
        if taken:
            self._telemetry["jam_opportunity_taken_count"] += 1
        else:
            self._record_read("missed_jam_opportunity", view)

    def _record_read(self, read_name: str, view: PlayerView) -> None:
        key = f"true_{read_name}_count"
        if key in self._telemetry:
            self._telemetry[key] += 1
        if getattr(view, "me", None) != _HERO_PID:
            opponents = getattr(view, "opponents", None) or []
            if _HERO_PID not in opponents:
                fold_bucket = self._hero_fold_bucket(view)
                self._telemetry["missed_due_to_hero_already_folded_count"] += 1
                folded_bucket = f"missed_due_to_hero_already_folded_{fold_bucket}_count"
                self._telemetry[folded_bucket] = self._telemetry.get(folded_bucket, 0) + 1
                specific = f"missed_due_to_hero_already_folded_{read_name}_count"
                self._telemetry[specific] = self._telemetry.get(specific, 0) + 1
                specific_bucket = f"missed_due_to_hero_already_folded_{fold_bucket}_{read_name}_count"
                self._telemetry[specific_bucket] = self._telemetry.get(specific_bucket, 0) + 1

    def _hero_fold_bucket(self, view: PlayerView) -> str:
        for entry in getattr(view, "history", None) or []:
            if (
                isinstance(entry, dict)
                and entry.get("pid") == _HERO_PID
                and entry.get("type") == "fold"
            ):
                street = entry.get("street")
                if street == "preflop":
                    return "preflop"
                if street in _POSTFLOP:
                    return "postflop"
                return "unknown"
        return "unknown"

    def _record_action_telemetry(self, view: PlayerView, action: Action) -> None:
        street = getattr(view, "street", None)
        action_type = getattr(action, "type", None)
        to_call = self.safe_int(getattr(view, "to_call", 0))
        self._seen_action_keys.add((
            getattr(view, "hand_id", None),
            len(getattr(view, "history", None) or []),
            getattr(view, "me", None),
            str(street or ""),
            str(action_type or ""),
        ))
        if street == "preflop":
            if action_type == "raise":
                self._record_read("vpip", view)
                self._record_read("pfr", view)
                spec = self.spec_for(view, "raise")
                limits = self.bounds(spec)
                if limits is not None and action.amount == limits[0]:
                    self._record_read("minraise", view)
            elif action_type == "call" and to_call > 0:
                self._record_read("vpip", view)

        if to_call > 0 and action_type == "fold":
            self._record_read("fold_to_pressure", view)
        if street in _POSTFLOP and to_call > 0 and action_type == "call":
            self._record_read("station", view)
        if street in _POSTFLOP and action_type in ("bet", "raise"):
            self._record_read("pressure_bet", view)
            read = self._classify_own_postflop_bet(view, action)
            if read is not None:
                self._record_read(read, view)
                if read == "large_bet":
                    self._self_large_bet_count += 1

    def _classify_own_postflop_bet(self, view: PlayerView, action: Action) -> str | None:
        spec = self.spec_for(view, str(action.type))
        limits = self.bounds(spec)
        if limits is None:
            return None
        bb = max(1, self.infer_bb(view))
        prior = self.current_contrib(view)
        amount = self.safe_int(getattr(action, "amount", 0))
        incremental = max(0, amount - prior)
        all_in = amount >= limits[1]
        if all_in:
            if incremental / bb < 12.0:
                return "short_jam_like"
            return "jam_like"
        pot_before = self.safe_int(getattr(view, "pot", 0))
        if incremental >= max(8 * bb, int(round(0.75 * max(1, pot_before)))):
            return "large_bet"
        return None

    def _ingest_visible_history(self, view: PlayerView) -> None:
        hand_id = getattr(view, "hand_id", None)
        bb = max(1, self.infer_bb(view))
        state: dict[str, dict[Any, int]] = {"contrib": {}}
        for seq, entry in enumerate(getattr(view, "history", None) or []):
            if not isinstance(entry, dict):
                continue
            street = str(entry.get("street") or "")
            pid = entry.get("pid")
            action_type = str(entry.get("type") or "")
            key = (hand_id, seq, pid, street, action_type)
            prior = state["contrib"].get((street, pid), 0)
            amount = self.safe_int(entry.get("amount"))
            if key not in self._seen_action_keys and pid == getattr(view, "me", None):
                incremental = max(0, amount - prior) if action_type in ("bet", "raise") else 0
                pot_before = self.safe_int(entry.get("pot_before"))
                all_in = pid in (getattr(view, "all_in_opponents", None) or [])
                large = (
                    street in _POSTFLOP
                    and action_type in ("bet", "raise")
                    and not all_in
                    and incremental >= max(8 * bb, int(round(0.75 * max(1, pot_before))))
                )
                if large:
                    self._self_large_bet_count += 1
                self._seen_action_keys.add(key)
            if action_type == "call":
                state["contrib"][(street, pid)] = prior + max(0, amount)
            elif action_type in ("bet", "raise"):
                state["contrib"][(street, pid)] = max(prior, amount)

    @staticmethod
    def _stable_cards(cards: list[Any] | tuple[Any, ...]) -> tuple[str, ...]:
        out: list[str] = []
        for card in cards:
            try:
                out.append(f"{card[0]}{card[1]}")
            except (TypeError, IndexError):
                out.append(str(card))
        return tuple(out)

    @classmethod
    def _stable_value(cls, value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return "(" + ",".join(cls._stable_value(v) for v in value) + ")"
        if value is None:
            return "None"
        return str(value)

    @staticmethod
    def safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0


class BasePolicy:
    def act(self, bot: ArchetypeBot, view: PlayerView) -> Action:
        return bot.safe_passive(view)

    def loose_preflop(self, bot: ArchetypeBot, view: PlayerView, *, raise_freq: float, call_freq: float) -> Action:
        hand = bot.hand_class(view)
        if "raise" in bot.legal_types(view) and (
            hand in ("premium", "playable")
            or bot.bernoulli(view, f"{bot.config.name}:preflop_raise", raise_freq)
        ):
            return bot.aggressive_to(view, self.open_target(bot, view))
        if bot.safe_int(getattr(view, "to_call", 0)) > 0 and "call" in bot.legal_types(view):
            if hand != "trash" or bot.bernoulli(view, f"{bot.config.name}:preflop_call", call_freq):
                return Action("call")
        return bot.fold_or_check(view)

    def open_target(self, bot: ArchetypeBot, view: PlayerView, mult: float = 3.0) -> int:
        spec = bot.aggressive_spec(view)
        limits = bot.bounds(spec)
        if limits is None:
            return 0
        lo, hi = limits
        bb = bot.infer_bb(view)
        target = max(lo, int(round(mult * bb)))
        return min(hi, target)

    def capped_bet(self, bot: ArchetypeBot, view: PlayerView, *, pot_frac: float, keepbehind_bb: int = 4) -> Action:
        spec = bot.aggressive_spec(view)
        limits = bot.bounds(spec)
        if spec is None or limits is None:
            return bot.safe_passive(view)
        lo, hi = limits
        bb = bot.infer_bb(view)
        prior = bot.current_contrib(view)
        cap = max(lo, hi - keepbehind_bb * bb)
        if cap < lo:
            return bot.safe_passive(view)
        incremental = max(bb, int(round(max(1, bot.safe_int(getattr(view, "pot", 0))) * pot_frac)))
        return Action(str(spec["type"]), max(lo, min(cap, prior + incremental)))


class ManiacPolicy(BasePolicy):
    def act(self, bot: ArchetypeBot, view: PlayerView) -> Action:
        if getattr(view, "street", None) == "preflop":
            return self.loose_preflop(
                bot,
                view,
                raise_freq=float(bot.knob("preflop_raise_freq", 0.50)),
                call_freq=float(bot.knob("preflop_call_freq", 0.30)),
            )

        if getattr(view, "street", None) in _POSTFLOP:
            eligible = self._eligible_jam(bot, view)
            if eligible:
                take = bot.bernoulli(view, f"{bot.config.name}:postflop_jam", bot.knob("jam_freq", 0.45))
                bot.note_jam_opportunity(view, taken=take)
                if take:
                    return bot.shove(view)
            if "raise" in bot.legal_types(view) and bot.bernoulli(view, f"{bot.config.name}:postflop_raise", 0.20):
                return self.capped_bet(bot, view, pot_frac=0.55, keepbehind_bb=4)
            if "bet" in bot.legal_types(view) and bot.bernoulli(
                view,
                f"{bot.config.name}:postflop_probe",
                bot.knob("postflop_probe_freq", 0.35),
            ):
                return self.capped_bet(bot, view, pot_frac=0.55, keepbehind_bb=4)
            if bot.safe_int(getattr(view, "to_call", 0)) > 0 and "call" in bot.legal_types(view):
                if bot.bernoulli(view, f"{bot.config.name}:postflop_call", 0.46):
                    return Action("call")
                return bot.fold_or_check(view)
        return bot.safe_passive(view)

    def _eligible_jam(self, bot: ArchetypeBot, view: PlayerView) -> bool:
        if getattr(view, "street", None) not in _POSTFLOP:
            return False
        spec = bot.aggressive_spec(view)
        limits = bot.bounds(spec)
        if limits is None:
            return False
        bb = bot.infer_bb(view)
        eff_bb = bot.effective_stack_bb(view)
        if eff_bb < 12.0 or eff_bb > 35.0:
            return False
        if bot.stack(view) / bb < float(bot.knob("anti_bust_bb", 9)):
            return False
        incremental = limits[1] - bot.current_contrib(view)
        if incremental < 12 * bb:
            return False
        return True


class OverbetPolicy(BasePolicy):
    def act(self, bot: ArchetypeBot, view: PlayerView) -> Action:
        if getattr(view, "street", None) == "preflop":
            return self.loose_preflop(
                bot,
                view,
                raise_freq=float(bot.knob("preflop_raise_freq", 0.34)),
                call_freq=float(bot.knob("preflop_call_freq", 0.24)),
            )
        if getattr(view, "street", None) in _POSTFLOP:
            if bot._self_large_bet_count >= 3 and self._eligible_late_jam(bot, view):
                if bot.bernoulli(view, "overbet:late_jam", bot.knob("late_jam_freq", 0.14)):
                    return bot.shove(view)
            if self._eligible_overbet(bot, view) and bot.bernoulli(
                view,
                "overbet:overbet",
                bot.knob("overbet_freq", 0.64),
            ):
                return self._overbet(bot, view)
            if bot.safe_int(getattr(view, "to_call", 0)) > 0:
                return bot.call_or_check(view)
        return bot.safe_passive(view)

    def _eligible_overbet(self, bot: ArchetypeBot, view: PlayerView) -> bool:
        spec = bot.aggressive_spec(view)
        limits = bot.bounds(spec)
        if limits is None:
            return False
        eff_bb = bot.effective_stack_bb(view)
        if eff_bb < 18.0 or eff_bb > 60.0:
            return False
        bb = bot.infer_bb(view)
        keep = int(bot.knob("keepbehind_bb", 3)) * bb
        prior = bot.current_contrib(view)
        max_non_allin = limits[1] - keep
        threshold = max(8 * bb, int(round(0.75 * max(1, bot.safe_int(getattr(view, "pot", 0))))))
        return max_non_allin >= limits[0] and max_non_allin - prior >= threshold

    def _overbet(self, bot: ArchetypeBot, view: PlayerView) -> Action:
        spec = bot.aggressive_spec(view)
        limits = bot.bounds(spec)
        if spec is None or limits is None:
            return bot.safe_passive(view)
        lo, hi = limits
        bb = bot.infer_bb(view)
        keep = int(bot.knob("keepbehind_bb", 3)) * bb
        prior = bot.current_contrib(view)
        pot = max(1, bot.safe_int(getattr(view, "pot", 0)))
        incremental = max(8 * bb, int(round(0.95 * pot)))
        target = max(lo, prior + incremental)
        target = min(target, hi - keep)
        return Action(str(spec["type"]), target)

    def _eligible_late_jam(self, bot: ArchetypeBot, view: PlayerView) -> bool:
        spec = bot.aggressive_spec(view)
        limits = bot.bounds(spec)
        if limits is None:
            return False
        bb = bot.infer_bb(view)
        return limits[1] - bot.current_contrib(view) >= 12 * bb


class StationPolicy(BasePolicy):
    def act(self, bot: ArchetypeBot, view: PlayerView) -> Action:
        if getattr(view, "street", None) == "preflop":
            if "raise" in bot.legal_types(view) and bot.bernoulli(
                view,
                "station:preflop_raise",
                bot.knob("preflop_raise_freq", 0.03),
            ):
                return bot.min_aggressive(view)
            if bot.safe_int(getattr(view, "to_call", 0)) > 0 and "call" in bot.legal_types(view):
                if bot.hand_class(view) != "trash" or bot.bernoulli(
                    view,
                    "station:preflop_call",
                    bot.knob("preflop_call_freq", 0.58),
                ):
                    return Action("call")
            return bot.fold_or_check(view)
        if getattr(view, "street", None) in _POSTFLOP:
            if bot.safe_int(getattr(view, "to_call", 0)) > 0:
                call_freq = bot.knob("pressure_call_freq", 0.84)
                if bot.safe_int(getattr(view, "to_call", 0)) >= bot.stack(view):
                    call_freq = bot.knob("allin_call_freq", 0.44)
                if "call" in bot.legal_types(view) and bot.bernoulli(view, "station:call", call_freq):
                    return Action("call")
                return bot.fold_or_check(view)
            return bot.safe_passive(view)
        return bot.safe_passive(view)


class NitPolicy(BasePolicy):
    def act(self, bot: ArchetypeBot, view: PlayerView) -> Action:
        hand = bot.hand_class(view)
        if bot.safe_int(getattr(view, "to_call", 0)) > 0:
            if hand == "premium" and "raise" in bot.legal_types(view) and bot.bernoulli(
                view,
                "nit:premium_raise",
                bot.knob("premium_raise_freq", 0.72),
            ):
                return bot.min_aggressive(view)
            if hand in ("premium", "playable") and "call" in bot.legal_types(view) and bot.bernoulli(
                view,
                "nit:premium_call",
                bot.knob("premium_call_freq", 0.62),
            ):
                return Action("call")
            if bot.bernoulli(view, "nit:fold", bot.knob("pressure_fold_freq", 0.88)):
                return bot.fold_or_check(view)
            return bot.call_or_check(view)
        if getattr(view, "street", None) == "preflop" and hand == "premium" and "raise" in bot.legal_types(view):
            return bot.min_aggressive(view)
        return bot.safe_passive(view)


class LoosePassivePolicy(BasePolicy):
    def act(self, bot: ArchetypeBot, view: PlayerView) -> Action:
        if getattr(view, "street", None) == "preflop":
            if "raise" in bot.legal_types(view) and bot.bernoulli(
                view,
                "loose_passive:rare_raise",
                bot.knob("preflop_raise_freq", 0.05),
            ):
                return bot.min_aggressive(view)
            if bot.safe_int(getattr(view, "to_call", 0)) > 0 and "call" in bot.legal_types(view):
                if bot.hand_class(view) != "trash" or bot.bernoulli(
                    view,
                    "loose_passive:call",
                    bot.knob("preflop_call_freq", 0.52),
                ):
                    return Action("call")
            return bot.fold_or_check(view)
        if getattr(view, "street", None) in _POSTFLOP:
            if bot.safe_int(getattr(view, "to_call", 0)) > 0:
                if "call" in bot.legal_types(view) and bot.bernoulli(
                    view,
                    "loose_passive:pressure_call",
                    bot.knob("pressure_call_freq", 0.56),
                ):
                    return Action("call")
                return bot.fold_or_check(view)
            if bot.bernoulli(view, "loose_passive:check", bot.knob("check_freq", 0.90)):
                return bot.safe_passive(view)
        return bot.safe_passive(view)


class MinRaisePolicy(BasePolicy):
    def act(self, bot: ArchetypeBot, view: PlayerView) -> Action:
        if bot.effective_stack_bb(view) < float(bot.knob("stop_below_bb", 9)):
            return bot.call_or_check(view)
        if getattr(view, "street", None) == "preflop":
            if "raise" in bot.legal_types(view) and bot.bernoulli(
                view,
                "minraise:preflop_raise",
                bot.knob("preflop_raise_freq", 0.30),
            ):
                return self._safe_minraise(bot, view)
            if bot.safe_int(getattr(view, "to_call", 0)) > 0 and "call" in bot.legal_types(view):
                if bot.bernoulli(view, "minraise:preflop_call", bot.knob("preflop_call_freq", 0.24)):
                    return Action("call")
            return bot.fold_or_check(view)
        if getattr(view, "street", None) in _POSTFLOP:
            if "raise" in bot.legal_types(view) and bot.bernoulli(
                view,
                "minraise:postflop_raise",
                bot.knob("postflop_minraise_freq", 0.30),
            ):
                return self._safe_minraise(bot, view)
            if "bet" in bot.legal_types(view) and bot.bernoulli(
                view,
                "minraise:postflop_bet",
                bot.knob("postflop_minraise_freq", 0.30),
            ):
                return self._safe_minraise(bot, view)
            return bot.call_or_check(view)
        return bot.safe_passive(view)

    def _safe_minraise(self, bot: ArchetypeBot, view: PlayerView) -> Action:
        spec = bot.aggressive_spec(view)
        limits = bot.bounds(spec)
        if spec is None or limits is None:
            return bot.safe_passive(view)
        lo, hi = limits
        if bool(spec.get("all_in")) or lo >= hi:
            return bot.call_or_check(view)
        return Action(str(spec["type"]), lo)


class BaselineSanePolicy(BasePolicy):
    def act(self, bot: ArchetypeBot, view: PlayerView) -> Action:
        hand = bot.hand_class(view)
        if getattr(view, "street", None) == "preflop":
            if hand == "premium" and "raise" in bot.legal_types(view):
                return bot.aggressive_to(view, self.open_target(bot, view, mult=3.0))
            if hand in ("playable", "speculative"):
                if "raise" in bot.legal_types(view) and bot.bernoulli(
                    view,
                    "baseline:raise",
                    bot.knob("preflop_raise_freq", 0.18),
                ):
                    return bot.aggressive_to(view, self.open_target(bot, view, mult=2.5))
                if bot.safe_int(getattr(view, "to_call", 0)) > 0 and "call" in bot.legal_types(view):
                    if bot.bernoulli(view, "baseline:call", bot.knob("preflop_call_freq", 0.22)):
                        return Action("call")
            return bot.fold_or_check(view)
        if getattr(view, "street", None) in _POSTFLOP:
            if bot.safe_int(getattr(view, "to_call", 0)) > 0:
                pot = max(1, bot.safe_int(getattr(view, "pot", 0)))
                if bot.safe_int(getattr(view, "to_call", 0)) <= int(round(0.35 * pot)) and "call" in bot.legal_types(view):
                    return Action("call")
                return bot.fold_or_check(view)
            if hand in ("premium", "playable") and "bet" in bot.legal_types(view) and bot.bernoulli(
                view,
                "baseline:postflop_bet",
                bot.knob("postflop_bet_freq", 0.20),
            ):
                return self.capped_bet(bot, view, pot_frac=0.35, keepbehind_bb=6)
        return bot.safe_passive(view)


class PressureFillerPolicy(BasePolicy):
    def act(self, bot: ArchetypeBot, view: PlayerView) -> Action:
        if getattr(view, "street", None) == "preflop":
            return self.loose_preflop(
                bot,
                view,
                raise_freq=float(bot.knob("preflop_raise_freq", 0.16)),
                call_freq=float(bot.knob("preflop_call_freq", 0.26)),
            )
        if getattr(view, "street", None) in _POSTFLOP:
            if bot.safe_int(getattr(view, "to_call", 0)) <= 0 and "bet" in bot.legal_types(view):
                if bot.bernoulli(view, "pressure_filler:bet", bot.knob("postflop_bet_freq", 0.86)):
                    return self.capped_bet(bot, view, pot_frac=0.55, keepbehind_bb=5)
            if bot.safe_int(getattr(view, "to_call", 0)) > 0:
                return bot.call_or_check(view)
        return bot.safe_passive(view)


_POLICIES = {
    "maniac": ManiacPolicy,
    "overbet": OverbetPolicy,
    "station": StationPolicy,
    "nit": NitPolicy,
    "loose_passive": LoosePassivePolicy,
    "minraise": MinRaisePolicy,
    "baseline_sane": BaselineSanePolicy,
    "pressure_filler": PressureFillerPolicy,
}
