#!/usr/bin/env python3
"""Smoke tests for the Phase 1 TournamentHybridBot skeleton."""

from __future__ import annotations

import io
import os
import random
import sys
from contextlib import redirect_stdout

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from bots import create_bot
from bots.tournament_hybrid_bot import (
    AGGRO_CONFIG,
    SURVIVAL_CONFIG,
    TournamentHybridBot,
)
from core.bot_api import Action, PlayerView
from core.engine import Seat, Table
from core.tournament import run_tournament


def _view(legal_actions, **overrides):
    data = {
        "me": "Hero",
        "street": "preflop",
        "position": "BTN",
        "hole_cards": [("A", "s"), ("K", "s")],
        "board": [],
        "pot": 30,
        "to_call": 0,
        "min_raise": 10,
        "max_raise": 100,
        "legal_actions": legal_actions,
        "stacks": {"Hero": 100, "Villain1": 120, "Villain2": 80},
        "opponents": ["Villain1", "Villain2"],
        "history": [],
        "hand_id": 1,
        "seat_indices": {"Hero": 0, "Villain1": 1, "Villain2": 2},
        "acting_opponents": ["Villain1", "Villain2"],
        "all_in_opponents": [],
    }
    data.update(overrides)
    return PlayerView(**data)


def _legal_types(legal_actions):
    return {
        spec.get("type")
        for spec in legal_actions
        if isinstance(spec, dict) and isinstance(spec.get("type"), str)
    }


def _spec_for(legal_actions, action_type):
    for spec in legal_actions:
        if isinstance(spec, dict) and spec.get("type") == action_type:
            return spec
    return None


def _has_valid_aggressive_spec(legal_actions, action_type):
    spec = _spec_for(legal_actions, action_type)
    if not isinstance(spec, dict):
        return False
    try:
        lo = int(spec["min"])
        hi = int(spec["max"])
    except (KeyError, TypeError, ValueError):
        return False
    return lo <= hi


def _valid_action(action, legal_actions):
    legal = _legal_types(legal_actions)
    if not legal:
        return action.type == "fold"
    if action.type not in legal:
        return False
    if action.type in ("bet", "raise"):
        if not _has_valid_aggressive_spec(legal_actions, action.type):
            return False
        spec = _spec_for(legal_actions, action.type)
        return int(spec["min"]) <= int(action.amount) <= int(spec["max"])
    return action.amount is None


def _expected_trace_all_in(action, view):
    if action.type == "call":
        hero_stack = int(view.stacks.get(view.me, 0))
        return hero_stack > 0 and int(view.to_call or 0) >= hero_stack
    if action.type in ("bet", "raise"):
        spec = _spec_for(view.legal_actions, action.type)
        if not isinstance(spec, dict):
            return False
        if bool(spec.get("all_in")):
            return True
        return action.amount is not None and int(action.amount) >= int(spec["max"])
    return False


def _passive_action(view):
    legal = _legal_types(view.legal_actions)
    for action_type in ("check", "call", "fold"):
        if action_type in legal:
            return Action(action_type)
    return Action("fold")


def _check_registry():
    adapters = {
        "final": create_bot("final"),
        "final_survival": create_bot("final_survival"),
        "final_aggro": create_bot("final_aggro"),
    }
    final_bot = adapters["final"].bot
    survival_bot = adapters["final_survival"].bot
    aggro_bot = adapters["final_aggro"].bot

    ok = (
        final_bot.config is SURVIVAL_CONFIG
        and survival_bot.config is SURVIVAL_CONFIG
        and aggro_bot.config is AGGRO_CONFIG
        and SURVIVAL_CONFIG is not AGGRO_CONFIG
        and SURVIVAL_CONFIG.name == "survival"
        and AGGRO_CONFIG.name == "aggro"
    )
    try:
        create_bot("__definitely_unknown__")
    except ValueError as exc:
        unknown_ok = "final_aggro" in str(exc)
    else:
        unknown_ok = False

    ok = ok and unknown_ok
    print(f"[CHECK 1] {'PASS' if ok else 'FAIL'} - registry aliases and "
          "distinct profile configs")
    return ok


def _check_fuzz():
    random.seed(20260619)
    bot = TournamentHybridBot("survival")
    normal_cases = [
        ("check-only", _view([{"type": "check"}])),
        ("check/bet", _view([{"type": "check"}, {"type": "bet", "min": 10, "max": 95}])),
        ("fold/call", _view([{"type": "fold"}, {"type": "call"}], to_call=15)),
        (
            "fold/call/raise",
            _view([
                {"type": "fold"},
                {"type": "call"},
                {"type": "raise", "min": 40, "max": 120},
            ], to_call=20),
        ),
        (
            "all-in raise",
            _view([
                {"type": "fold"},
                {"type": "call"},
                {
                    "type": "raise",
                    "min": 42,
                    "max": 42,
                    "all_in": True,
                    "reopens": False,
                },
            ], to_call=35),
        ),
        ("call-for-less", _view([{"type": "fold"}, {"type": "call"}], to_call=200)),
    ]
    degenerate_cases = [
        ("empty", _view([])),
        ("unknown action", _view([{"type": "dance"}])),
        ("malformed bet", _view([{"type": "bet"}])),
        ("bad raise bounds", _view([{"type": "raise", "min": "x", "max": 5}])),
        ("mixed odd", _view([{"foo": "bar"}, {"type": "call"}], to_call=10)),
    ]

    ok = True

    def check_normal_case(name, view, target):
        nonlocal ok
        for profile in ("survival", "aggro"):
            active_bot = TournamentHybridBot(profile)
            try:
                action = active_bot.act(view)
                aggressive = active_bot._safe_aggressive(view, target)
                aggressive = active_bot._sanitize(view, aggressive)
            except Exception:
                ok = False
                continue
            ok &= _valid_action(action, view.legal_actions)
            ok &= _valid_action(aggressive, view.legal_actions)
            ok &= active_bot.last_decision is not None
            ok &= active_bot.last_decision.get("profile") == profile
            ok &= active_bot.last_decision.get("path") == "passive"
            ok &= (
                active_bot.last_decision.get("all_in_detected")
                == _expected_trace_all_in(action, view)
            )
            if name == "call-for-less":
                ok &= active_bot.last_decision.get("all_in_detected") is True

    for name, view in normal_cases:
        check_normal_case(name, view, random.randint(-50, 180))
    for _ in range(40):
        name, view = random.choice(normal_cases)
        check_normal_case(name, view, random.randint(-50, 180))

    all_in_raise_view = next(
        view for name, view in normal_cases if name == "all-in raise"
    )
    spec = _spec_for(all_in_raise_view.legal_actions, "raise")
    raised = bot._safe_aggressive(all_in_raise_view, target_total=999)
    ok &= raised.amount == 42
    ok &= bot._is_all_in(spec, raised.amount)

    for _, view in degenerate_cases:
        try:
            action = bot.act(view)
            aggressive = bot._safe_aggressive(view, random.randint(-10, 50))
            bot._sanitize(view, aggressive)
        except Exception:
            ok = False
            continue
        enforceable = bool(
            _legal_types(view.legal_actions) & {"check", "call", "fold"}
        )
        enforceable |= any(
            _has_valid_aggressive_spec(view.legal_actions, typ)
            for typ in ("bet", "raise")
        )
        if enforceable:
            ok &= _valid_action(action, view.legal_actions)
        else:
            ok &= action.type == "fold"
        ok &= bot.last_decision is not None

    reset_bot = TournamentHybridBot("survival")
    reset_bot.act(normal_cases[0][1])
    reset_bot.reset_memory()
    ok &= reset_bot.last_decision is None

    class _ExplodingContextBot(TournamentHybridBot):
        @classmethod
        def _context(cls, view):
            raise RuntimeError("intentional internal failure")

    try:
        _ExplodingContextBot("survival").act(normal_cases[0][1])
    except RuntimeError:
        internal_exception_visible = True
    else:
        internal_exception_visible = False
    ok &= internal_exception_visible

    print(f"[CHECK 2] {'PASS' if ok else 'FAIL'} - seeded legal-action fuzz "
          "plus trace/reset invariants")
    return ok


class _FoldBot:
    def act(self, view):
        legal = _legal_types(view.legal_actions)
        if "fold" in legal:
            return Action("fold")
        if "check" in legal:
            return Action("check")
        if "call" in legal:
            return Action("call")
        return Action("fold")


class _OpeningRaiseBot:
    def __init__(self, target_total):
        self.target_total = target_total
        self.opened = False

    def act(self, view):
        legal = _legal_types(view.legal_actions)
        if not self.opened and "raise" in legal:
            spec = _spec_for(view.legal_actions, "raise")
            amount = max(int(spec["min"]), min(int(spec["max"]), self.target_total))
            self.opened = True
            return Action("raise", amount)
        return _passive_action(view)


class _SmallBlindAggressiveProbe:
    def __init__(self, target_total):
        self.core = TournamentHybridBot("survival")
        self.target_total = target_total
        self.before_stack = None
        self.to_call = None
        self.action = None

    def act(self, view):
        if self.action is None and "raise" in _legal_types(view.legal_actions):
            self.before_stack = view.stacks[view.me]
            self.to_call = view.to_call
            self.action = self.core._safe_aggressive(view, self.target_total)
            return self.action
        return self.core._safe_passive(view)


class _BigBlindObserver:
    def __init__(self, probe, observed_pid):
        self.probe = probe
        self.observed_pid = observed_pid
        self.after_stack = None

    def act(self, view):
        if self.probe.action is not None and self.after_stack is None:
            self.after_stack = view.stacks.get(self.observed_pid)
        return _passive_action(view)


def _check_contribution_semantics():
    seats = [
        Seat("BTN", 100),
        Seat("SB", 100),
        Seat("BB", 100),
        Seat("UTG", 100),
        Seat("UTG1", 100),
        Seat("MP", 100),
    ]
    probe = _SmallBlindAggressiveProbe(target_total=57)
    observer = _BigBlindObserver(probe, "SB")
    bots = {
        "BTN": _FoldBot(),
        "SB": probe,
        "BB": observer,
        "UTG": _OpeningRaiseBot(target_total=30),
        "UTG1": _FoldBot(),
        "MP": _FoldBot(),
    }

    buf = io.StringIO()
    with redirect_stdout(buf):
        Table(rng=random.Random(11)).play_hand(
            seats=seats,
            small_blind=5,
            big_blind=10,
            dealer_index=0,
            bot_for=bots,
            log_decisions=False,
        )
    output = buf.getvalue()

    action = probe.action or Action("fold")
    already_contributed = 100 - int(probe.before_stack or 0)
    expected_after = int(probe.before_stack or 0) - (
        int(action.amount or 0) - already_contributed
    )
    ok = (
        "[WARN] Illegal action" not in output
        and probe.before_stack == 95
        and probe.to_call == 25
        and action.type == "raise"
        and action.amount == 57
        and already_contributed == 5
        and observer.after_stack == expected_after
        and observer.after_stack == 43
    )
    print(f"[CHECK 3] {'PASS' if ok else 'FAIL'} - raise amount is raise-to "
          "total and engine pays only the remaining contribution")
    return ok


def _check_integration():
    ok = True
    warnings = []
    for seed in (101, 202, 303):
        random.seed(seed)
        player_specs = [
            ("S", "final_survival"),
            ("A", "final_aggro"),
            ("R1", "random"),
            ("R2", "random"),
            ("R3", "random"),
            ("R4", "random"),
        ]
        bots = {pid: create_bot(btype) for pid, btype in player_specs}
        seats = [Seat(player_id=pid, chips=300) for pid, _ in player_specs]
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = run_tournament(
                seats,
                bots,
                small_blind=5,
                big_blind=10,
                blind_increase_every=20,
                max_hands=30,
                dealer_index=0,
                dealer_rotation="full_table",
                winner_resolution="chip_count_on_max_hands",
                rng=random.Random(seed),
                suppress_output=False,
                log_decisions=False,
            )
        output = buf.getvalue()
        if "[WARN] Illegal action" in output:
            warnings.append(seed)
        ok &= result.get("winner") is not None
        ok &= result.get("hand_count", 0) > 0
    ok &= not warnings
    print(f"[CHECK 4] {'PASS' if ok else 'FAIL'} - 3 live six-player "
          "tournaments completed with zero illegal-action warnings")
    return ok


def run():
    PASS = True
    PASS &= _check_registry()
    PASS &= _check_fuzz()
    PASS &= _check_contribution_semantics()
    PASS &= _check_integration()
    print("=" * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED [PASS]' if PASS else 'SOME CHECKS FAILED [FAIL]'}")
    return PASS


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
