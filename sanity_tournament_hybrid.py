#!/usr/bin/env python3
"""Smoke tests for the Phase 2 TournamentHybridBot preflop strategy."""

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
            ok &= active_bot.last_decision.get("path") in {"passive", "preflop", "fallback"}
            ok &= active_bot.last_decision.get("branch") not in {
                "context_error",
                "preflop_exception",
            }
            ok &= "context_error" not in active_bot.last_decision.get("context", {})
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

    postflop_view = _view(
        [{"type": "fold"}, {"type": "call"}, {"type": "raise", "min": 40, "max": 120}],
        street="flop",
        board=[("A", "c"), ("7", "d"), ("2", "h")],
        to_call=20,
    )
    for profile in ("survival", "aggro"):
        postflop_bot = TournamentHybridBot(profile)
        postflop_action = postflop_bot.act(postflop_view)
        ok &= postflop_action == _passive_action(postflop_view)
        ok &= postflop_bot.last_decision.get("path") == "passive"

    reset_bot = TournamentHybridBot("survival")
    reset_bot.act(normal_cases[0][1])
    reset_bot.reset_memory()
    ok &= reset_bot.last_decision is None

    class _ExplodingContextBot(TournamentHybridBot):
        @classmethod
        def _context(cls, view):
            raise RuntimeError("intentional internal failure")

    class _ExplodingPreflopBot(TournamentHybridBot):
        def _preflop_action(self, view, context):
            raise RuntimeError("intentional preflop failure")

    exploding_bot = _ExplodingContextBot("survival")
    try:
        exploding_action = exploding_bot.act(normal_cases[0][1])
    except Exception:
        ok = False
    else:
        ok &= exploding_action.type == "check"
        ok &= exploding_bot.last_decision is not None
        ok &= exploding_bot.last_decision.get("branch") == "context_error"

    exploding_preflop_bot = _ExplodingPreflopBot("survival")
    try:
        exploding_preflop_action = exploding_preflop_bot.act(normal_cases[2][1])
    except Exception:
        ok = False
    else:
        ok &= exploding_preflop_action.type == "fold"
        ok &= exploding_preflop_bot.last_decision is not None
        ok &= exploding_preflop_bot.last_decision.get("branch") == "preflop_exception"

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


def _positions(n):
    return TournamentHybridBot._positions_for_n(n)


def _table_overrides(hero_pos="BTN", n=6, chips=1000, hero_stack=None):
    positions = _positions(n)
    if hero_pos not in positions:
        hero_pos = positions[0]
    pids = ["Hero" if pos == hero_pos else pos for pos in positions]
    stacks = {pid: chips for pid in pids}
    if hero_stack is not None:
        stacks["Hero"] = hero_stack
    return {
        "position": hero_pos,
        "stacks": stacks,
        "opponents": [pid for pid in pids if pid != "Hero"],
        "seat_indices": {pid: idx for idx, pid in enumerate(pids)},
        "acting_opponents": [pid for pid in pids if pid != "Hero"],
        "all_in_opponents": [],
    }


def _raise_spec(min_total, max_total, *, all_in=False):
    min_total = int(min_total)
    max_total = int(max_total)
    if max_total >= min_total:
        spec = {"type": "raise", "min": min_total, "max": max_total}
        if all_in:
            spec.update({"all_in": True, "reopens": False})
        return spec
    if max_total > 0:
        return {
            "type": "raise",
            "min": max_total,
            "max": max_total,
            "all_in": True,
            "reopens": False,
        }
    return None


def _rfi_view(position, hole, *, chips=1000, n=6, bb=10, hand_id=100):
    to_call = 0 if position == "BB" and n > 2 else bb
    legal = (
        [{"type": "check"}]
        if to_call == 0
        else [{"type": "fold"}, {"type": "call"}]
    )
    spec = _raise_spec(2 * bb, chips)
    if spec:
        legal.append(spec)
    return _view(
        legal,
        hole_cards=hole,
        pot=round(1.5 * bb),
        to_call=to_call,
        min_raise=bb if to_call == 0 else 2 * bb,
        max_raise=chips,
        hand_id=hand_id,
        **_table_overrides(position, n=n, chips=chips),
    )


def _open_history(pid="UTG", amount=30, bb=10):
    return [{
        "street": "preflop",
        "pid": pid,
        "type": "raise",
        "amount": amount,
        "to_call_before": bb,
        "pot_before": round(1.5 * bb),
    }]


def _facing_open_view(
    hole,
    *,
    opener="UTG",
    hero_pos="BTN",
    chips=1000,
    n=6,
    bb=10,
    open_to=30,
    hand_id=200,
):
    hero_contrib = bb if hero_pos == "BB" else (bb // 2 if hero_pos == "SB" else 0)
    hero_stack = max(1, chips - hero_contrib)
    to_call = max(0, open_to - hero_contrib)
    max_total = hero_stack + hero_contrib
    spec = _raise_spec(open_to + max(bb, open_to - bb), max_total)
    legal = [{"type": "fold"}, {"type": "call"}]
    if spec:
        legal.append(spec)
    return _view(
        legal,
        hole_cards=hole,
        pot=round(1.5 * bb) + open_to,
        to_call=to_call,
        min_raise=open_to + max(bb, open_to - bb),
        max_raise=hero_stack,
        history=_open_history(opener, open_to, bb),
        hand_id=hand_id,
        **_table_overrides(hero_pos, n=n, chips=chips, hero_stack=hero_stack),
    )


def _bb_vs_mp_open_after_folds_view():
    bb = 10
    open_to = 30
    hero_stack = 200
    pids = ["BTN_ID", "SB_ID", "Hero", "UTG_ID", "UTG1_ID", "MP_ID"]
    stacks = {pid: 1000 for pid in pids}
    stacks["Hero"] = hero_stack
    history = [
        {
            "street": "preflop",
            "pid": "UTG_ID",
            "type": "fold",
            "amount": None,
            "to_call_before": bb,
            "pot_before": 15,
        },
        {
            "street": "preflop",
            "pid": "UTG1_ID",
            "type": "fold",
            "amount": None,
            "to_call_before": bb,
            "pot_before": 15,
        },
        {
            "street": "preflop",
            "pid": "MP_ID",
            "type": "raise",
            "amount": open_to,
            "to_call_before": bb,
            "pot_before": 15,
        },
    ]
    return _view(
        [{"type": "fold"}, {"type": "call"}, {"type": "raise", "min": 50, "max": 210}],
        position="BB",
        hole_cards=[("A", "s"), ("T", "h")],
        pot=45,
        to_call=20,
        min_raise=40,
        max_raise=hero_stack,
        stacks=stacks,
        opponents=["BTN_ID", "SB_ID", "MP_ID"],
        acting_opponents=["BTN_ID", "SB_ID", "MP_ID"],
        all_in_opponents=[],
        seat_indices={pid: idx for idx, pid in enumerate(pids)},
        history=history,
        hand_id=1203,
    )


def _facing_multi_raise_view(
    hole,
    *,
    raises=2,
    hero_pos="BTN",
    chips=1000,
    bb=10,
    hand_id=300,
    forced_all_in=False,
):
    history = [
        {
            "street": "preflop",
            "pid": "Hero" if raises >= 2 else "UTG",
            "type": "raise",
            "amount": 30,
            "to_call_before": bb,
            "pot_before": round(1.5 * bb),
        },
        {
            "street": "preflop",
            "pid": "BTN",
            "type": "raise",
            "amount": 90,
            "to_call_before": 30,
            "pot_before": round(1.5 * bb) + 30,
        },
    ]
    to_call = 60
    min_total = 150
    pot = round(1.5 * bb) + 120
    if raises >= 3:
        history.append({
            "street": "preflop",
            "pid": "Hero",
            "type": "raise",
            "amount": 210,
            "to_call_before": 60,
            "pot_before": pot,
        })
        to_call = 120
        min_total = 330
        pot += 210
    max_total = chips
    spec = (
        {"type": "raise", "min": chips, "max": chips, "all_in": True, "reopens": False}
        if forced_all_in
        else _raise_spec(min_total, max_total)
    )
    legal = [{"type": "fold"}, {"type": "call"}]
    if spec:
        legal.append(spec)
    return _view(
        legal,
        hole_cards=hole,
        pot=pot,
        to_call=to_call,
        min_raise=min_total,
        max_raise=chips,
        history=history,
        hand_id=hand_id,
        **_table_overrides(hero_pos, n=6, chips=chips),
    )


def _limped_view(hole, *, hero_pos="BTN", limpers=2, chips=1000, bb=10, hand_id=400):
    history = [
        {
            "street": "preflop",
            "pid": f"L{i}",
            "type": "call",
            "amount": bb,
            "to_call_before": bb,
            "pot_before": round(1.5 * bb) + i * bb,
        }
        for i in range(limpers)
    ]
    to_call = 0 if hero_pos == "BB" else bb
    legal = [{"type": "check"}] if to_call == 0 else [{"type": "fold"}, {"type": "call"}]
    spec = _raise_spec(2 * bb, chips)
    if spec:
        legal.append(spec)
    return _view(
        legal,
        hole_cards=hole,
        pot=round(1.5 * bb) + limpers * bb,
        to_call=to_call,
        min_raise=bb if to_call == 0 else 2 * bb,
        max_raise=chips,
        history=history,
        hand_id=hand_id,
        **_table_overrides(hero_pos, n=6, chips=chips),
    )


def _check_blind_inference():
    bot = TournamentHybridBot("survival")
    unopened = _view(
        [{"type": "fold"}, {"type": "call"}, {"type": "raise", "min": 20, "max": 1000}],
        pot=15,
        to_call=10,
        min_raise=20,
        hand_id=501,
    )
    bb_option = _view(
        [{"type": "check"}, {"type": "raise", "min": 20, "max": 1000}],
        position="BB",
        pot=45,
        to_call=0,
        min_raise=10,
        hand_id=502,
    )
    facing_raise = _facing_open_view([("A", "s"), ("K", "s")], hand_id=503)
    ante_limped = _limped_view(
        [("A", "s"), ("K", "s")],
        hero_pos="BTN",
        limpers=1,
        bb=10,
        hand_id=506,
    )
    ante_limped.history[0]["pot_before"] = 33
    ante_limped.pot = 43
    cached = _view(
        [{"type": "check"}, {"type": "raise", "min": 20, "max": 1000}],
        pot=15,
        to_call=0,
        min_raise=10,
        hand_id=504,
    )
    cached_repeat = _view(
        [{"type": "check"}, {"type": "raise", "min": 198, "max": 1000}],
        pot=150,
        to_call=0,
        min_raise=99,
        hand_id=504,
    )
    ok = (
        bot._infer_big_blind(unopened) == 10
        and bot._infer_big_blind(bb_option) == 10
        and bot._infer_big_blind(facing_raise) == 10
        and bot._infer_big_blind(ante_limped) == 10
        and bot._infer_big_blind(cached) == 10
        and bot._infer_big_blind(cached_repeat) == 10
    )
    bot.reset_memory()
    fresh = _view(
        [{"type": "check"}, {"type": "raise", "min": 40, "max": 1000}],
        pot=30,
        to_call=0,
        min_raise=20,
        hand_id=505,
    )
    ok &= bot._infer_big_blind(fresh) == 20
    print(f"[CHECK 5] {'PASS' if ok else 'FAIL'} - big-blind inference and per-hand cache")
    return ok


def _check_position_classification():
    ok = True
    for n in range(2, 7):
        ok &= TournamentHybridBot._classify_position("SB", n) == "sb"
        ok &= TournamentHybridBot._classify_position("BB", n) == "bb"
    ok &= TournamentHybridBot._classify_position("BTN", 2) == "late"
    ok &= TournamentHybridBot._classify_position("MP", 6) == "late"
    ok &= TournamentHybridBot._classify_position("MP", 5) == "middle"
    ok &= TournamentHybridBot._classify_position("UTG", 6) == "early"
    ok &= TournamentHybridBot._classify_position("UTG+1", 6) == "early"
    ok &= TournamentHybridBot._classify_position("CO", 6) == "late"
    ok &= TournamentHybridBot._classify_position("mystery", 6) == "middle"
    print(f"[CHECK 6] {'PASS' if ok else 'FAIL'} - position classification n=2..6")
    return ok


def _check_utg_rfi():
    ok = True
    junk = _rfi_view("UTG", [("7", "s"), ("2", "h")], hand_id=701)
    aa = _rfi_view("UTG", [("A", "s"), ("A", "h")], hand_id=702)
    for profile in ("survival", "aggro"):
        bot = TournamentHybridBot(profile)
        action = bot.act(junk)
        ok &= action.type == "fold"
        ok &= _valid_action(action, junk.legal_actions)
    bot = TournamentHybridBot("survival")
    action = bot.act(aa)
    ok &= action.type == "raise"
    ok &= 20 <= int(action.amount or 0) <= 30
    ok &= _valid_action(action, aa.legal_actions)
    print(f"[CHECK 7] {'PASS' if ok else 'FAIL'} - UTG junk folds and AA opens")
    return ok


def _check_profile_divergence():
    ok = True
    suited_j8 = _rfi_view("BTN", [("J", "s"), ("8", "s")], hand_id=801)
    aa = _rfi_view("BTN", [("A", "s"), ("A", "h")], hand_id=802)
    survival_j8 = TournamentHybridBot("survival").act(suited_j8)
    aggro_j8 = TournamentHybridBot("aggro").act(suited_j8)
    survival_aa = TournamentHybridBot("survival").act(aa)
    aggro_aa = TournamentHybridBot("aggro").act(aa)
    ok &= survival_j8.type == "fold"
    ok &= aggro_j8.type == "raise"
    ok &= survival_aa.type == "raise"
    ok &= aggro_aa.type == "raise"
    ok &= all(_valid_action(a, v.legal_actions) for a, v in [
        (survival_j8, suited_j8),
        (aggro_j8, suited_j8),
        (survival_aa, aa),
        (aggro_aa, aa),
    ])
    print(f"[CHECK 8] {'PASS' if ok else 'FAIL'} - survival/aggro BTN range divergence")
    return ok


def _check_raise_response_tree():
    ok = True
    kk = _facing_open_view([("K", "s"), ("K", "h")], opener="UTG", hand_id=901)
    tt = _facing_open_view([("T", "s"), ("T", "h")], opener="UTG", hand_id=902)
    trash = _facing_open_view([("7", "s"), ("2", "h")], opener="UTG", hand_id=903)
    aq_early = _facing_open_view([("A", "s"), ("Q", "s")], opener="UTG", hand_id=904)
    aq_late = _facing_open_view([("A", "s"), ("Q", "s")], opener="BTN", hero_pos="BB", hand_id=905)
    actions = [
        (TournamentHybridBot("survival").act(kk), kk),
        (TournamentHybridBot("survival").act(tt), tt),
        (TournamentHybridBot("survival").act(trash), trash),
        (TournamentHybridBot("survival").act(aq_early), aq_early),
        (TournamentHybridBot("survival").act(aq_late), aq_late),
    ]
    ok &= actions[0][0].type == "raise"
    ok &= actions[1][0].type == "call"
    ok &= actions[2][0].type == "fold"
    ok &= actions[3][0].type == "call"
    ok &= actions[4][0].type == "raise"

    aa_3bet = _facing_multi_raise_view([("A", "s"), ("A", "h")], raises=2, hand_id=906)
    kq_3bet = _facing_multi_raise_view([("K", "s"), ("Q", "s")], raises=2, hand_id=907)
    aks_4bet = _facing_multi_raise_view([("A", "s"), ("K", "s")], raises=3, hand_id=908)
    qq_4bet = _facing_multi_raise_view([("Q", "s"), ("Q", "h")], raises=3, hand_id=909)
    qq_4bet_30bb = _facing_multi_raise_view(
        [("Q", "s"), ("Q", "h")],
        raises=3,
        chips=300,
        forced_all_in=True,
        hand_id=910,
    )
    a_aa = TournamentHybridBot("survival").act(aa_3bet)
    a_kq = TournamentHybridBot("survival").act(kq_3bet)
    a_aks = TournamentHybridBot("survival").act(aks_4bet)
    a_qq = TournamentHybridBot("survival").act(qq_4bet)
    a_qq_30 = TournamentHybridBot("survival").act(qq_4bet_30bb)
    ok &= a_aa.type == "raise"
    ok &= a_kq.type == "fold"
    ok &= a_aks.type == "raise"
    ok &= a_qq.type == "fold"
    ok &= a_qq_30.type == "raise" and a_qq_30.amount == 300
    actions.extend([
        (a_aa, aa_3bet),
        (a_kq, kq_3bet),
        (a_aks, aks_4bet),
        (a_qq, qq_4bet),
        (a_qq_30, qq_4bet_30bb),
    ])
    ok &= all(_valid_action(action, view.legal_actions) for action, view in actions)
    print(f"[CHECK 9] {'PASS' if ok else 'FAIL'} - raise/3bet/4bet response tree")
    return ok


def _check_limper_tree():
    ok = True
    aks = _limped_view([("A", "s"), ("K", "s")], hero_pos="BTN", limpers=2, hand_id=1001)
    suited_65 = _limped_view([("6", "s"), ("5", "s")], hero_pos="BTN", limpers=2, hand_id=1002)
    bb_trash = _limped_view([("Q", "s"), ("7", "h")], hero_pos="BB", limpers=2, hand_id=1003)
    a_aks = TournamentHybridBot("survival").act(aks)
    a_65 = TournamentHybridBot("survival").act(suited_65)
    a_bb = TournamentHybridBot("survival").act(bb_trash)
    ok &= a_aks.type == "raise" and 45 <= int(a_aks.amount or 0) <= 65
    ok &= a_65.type == "call"
    ok &= a_bb.type == "check"
    ok &= all(_valid_action(action, view.legal_actions) for action, view in [
        (a_aks, aks),
        (a_65, suited_65),
        (a_bb, bb_trash),
    ])
    print(f"[CHECK 10] {'PASS' if ok else 'FAIL'} - limper iso/overlimp/BB check tree")
    return ok


def _check_short_stack():
    ok = True
    aa_short = _rfi_view("UTG", [("A", "s"), ("A", "h")], chips=100, hand_id=1101)
    trash_short = _rfi_view("UTG", [("7", "s"), ("2", "h")], chips=100, hand_id=1102)
    forced = _rfi_view("UTG", [("A", "s"), ("A", "h")], chips=83, hand_id=1103)
    forced.legal_actions = [
        {"type": "fold"},
        {"type": "call"},
        {"type": "raise", "min": 83, "max": 83, "all_in": True, "reopens": False},
    ]
    a2s_vs_utg_all_in = _facing_open_view(
        [("A", "s"), ("2", "s")],
        opener="UTG",
        hero_pos="BB",
        chips=100,
        open_to=100,
        hand_id=1104,
    )
    a2s_vs_utg_all_in.legal_actions = [{"type": "fold"}, {"type": "call"}]
    a2s_vs_utg_all_in.stacks["UTG"] = 0
    a2s_vs_utg_all_in.all_in_opponents = ["UTG"]
    a_aa_bot = TournamentHybridBot("survival")
    a_aa = a_aa_bot.act(aa_short)
    a_trash = TournamentHybridBot("survival").act(trash_short)
    a_forced = TournamentHybridBot("survival").act(forced)
    a_a2s = TournamentHybridBot("survival").act(a2s_vs_utg_all_in)
    ok &= a_aa.type == "raise" and a_aa_bot._action_is_all_in(aa_short, a_aa)
    ok &= a_trash.type == "fold"
    ok &= a_forced.type == "raise" and a_forced.amount == 83
    ok &= a_a2s.type == "fold"
    ok &= all(_valid_action(action, view.legal_actions) for action, view in [
        (a_aa, aa_short),
        (a_trash, trash_short),
        (a_forced, forced),
        (a_a2s, a2s_vs_utg_all_in),
    ])
    print(f"[CHECK 11] {'PASS' if ok else 'FAIL'} - short-stack shove/fold")
    return ok


def _check_medium_stack():
    ok = True
    resteal = _facing_open_view(
        [("A", "s"), ("T", "h")],
        opener="BTN",
        hero_pos="BB",
        chips=210,
        hand_id=1201,
    )
    commit = _facing_open_view(
        [("K", "s"), ("K", "h")],
        opener="UTG",
        hero_pos="BTN",
        chips=240,
        hand_id=1202,
    )
    side_pot_live_open = _facing_open_view(
        [("A", "s"), ("A", "h")],
        opener="UTG",
        hero_pos="BTN",
        chips=210,
        hand_id=1204,
    )
    side_pot_live_open.stacks["SB"] = 0
    side_pot_live_open.all_in_opponents = ["SB"]
    folded_mp_open = _bb_vs_mp_open_after_folds_view()
    b1 = TournamentHybridBot("survival")
    a_resteal = b1.act(resteal)
    b2 = TournamentHybridBot("survival")
    a_commit = b2.act(commit)
    b3 = TournamentHybridBot("survival")
    a_side_pot = b3.act(side_pot_live_open)
    b4 = TournamentHybridBot("survival")
    a_folded_mp = b4.act(folded_mp_open)
    ok &= a_resteal.type == "raise" and b1._action_is_all_in(resteal, a_resteal)
    ok &= a_commit.type == "raise" and b2._action_is_all_in(commit, a_commit)
    ok &= a_side_pot.type == "raise" and b3._action_is_all_in(side_pot_live_open, a_side_pot)
    ok &= a_folded_mp.type == "raise"
    ok &= b4.last_decision.get("branch") == "medium_resteal_vs_late"
    ok &= all(_valid_action(action, view.legal_actions) for action, view in [
        (a_resteal, resteal),
        (a_commit, commit),
        (a_side_pot, side_pot_live_open),
        (a_folded_mp, folded_mp_open),
    ])
    print(f"[CHECK 12] {'PASS' if ok else 'FAIL'} - medium-stack resteal and commit jams")
    return ok


def _random_hole(rng):
    deck = [(r, s) for r in "23456789TJQKA" for s in "cdhs"]
    return rng.sample(deck, 2)


def _random_fuzz_view(rng, idx):
    n = rng.randint(2, 6)
    positions = _positions(n)
    hero_pos = rng.choice(positions)
    bb = rng.choice([5, 10, 15, 20])
    chips = rng.randint(15, 1500)
    hole = _random_hole(rng)
    scenario = rng.choice(["empty", "unopened", "bb_option", "limped", "single", "threebet", "fourbet", "forced"])
    overrides = _table_overrides(hero_pos, n=n, chips=chips)
    if scenario == "empty":
        return _view([], hole_cards=hole, pot=round(1.5 * bb), to_call=0,
                     min_raise=bb, hand_id=13000 + idx, **overrides)
    if scenario == "bb_option":
        spec = _raise_spec(2 * bb, chips)
        legal = [{"type": "check"}] + ([spec] if spec else [])
        return _view(legal, hole_cards=hole, pot=round(1.5 * bb), to_call=0,
                     min_raise=bb, hand_id=13000 + idx, **overrides)
    if scenario == "limped":
        return _limped_view(hole, hero_pos=hero_pos, limpers=rng.randint(1, 3),
                            chips=chips, bb=bb, hand_id=13000 + idx)
    if scenario == "single":
        opener = rng.choice([p for p in positions if p != hero_pos] or ["UTG"])
        return _facing_open_view(hole, opener=opener, hero_pos=hero_pos,
                                 chips=chips, n=n, bb=bb, hand_id=13000 + idx)
    if scenario == "threebet":
        return _facing_multi_raise_view(hole, raises=2, hero_pos=hero_pos,
                                        chips=max(chips, 80), bb=bb, hand_id=13000 + idx)
    if scenario == "fourbet":
        return _facing_multi_raise_view(hole, raises=3, hero_pos=hero_pos,
                                        chips=max(chips, 120), bb=bb, hand_id=13000 + idx)
    if scenario == "forced":
        view = _facing_open_view(hole, opener="BTN", hero_pos=hero_pos,
                                 chips=max(chips, 20), n=n, bb=bb, hand_id=13000 + idx)
        max_total = max(1, int(view.legal_actions[-1].get("max", chips)))
        view.legal_actions[-1] = {
            "type": "raise",
            "min": max_total,
            "max": max_total,
            "all_in": True,
            "reopens": False,
        }
        return view
    return _rfi_view(hero_pos, hole, chips=chips, n=n, bb=bb, hand_id=13000 + idx)


def _check_determinism_and_preflop_fuzz():
    ok = True
    deterministic_view = _rfi_view("BTN", [("J", "s"), ("8", "s")], hand_id=1301)
    bot = TournamentHybridBot("aggro")
    first = bot.act(deterministic_view)
    second = bot.act(deterministic_view)
    ok &= first == second

    rng = random.Random(20260619)
    required_trace = {
        "position_category",
        "inferred_bb",
        "eff_bb",
        "band",
        "raises_faced",
        "branch",
        "range_hit",
    }
    for i in range(200):
        view = _random_fuzz_view(rng, i)
        for profile in ("survival", "aggro"):
            active = TournamentHybridBot(profile)
            try:
                action = active.act(view)
            except Exception:
                ok = False
                continue
            ok &= _valid_action(action, view.legal_actions)
            ok &= active.last_decision is not None
            ok &= required_trace.issubset(active.last_decision.keys())
            ok &= active.last_decision.get("profile") == profile
            ok &= active.last_decision.get("branch") not in {
                "context_error",
                "preflop_exception",
            }
            ok &= "context_error" not in active.last_decision.get("context", {})
    print(f"[CHECK 13] {'PASS' if ok else 'FAIL'} - deterministic actions and 200-case preflop fuzz")
    return ok


def run():
    PASS = True
    PASS &= _check_registry()
    PASS &= _check_fuzz()
    PASS &= _check_contribution_semantics()
    PASS &= _check_integration()
    PASS &= _check_blind_inference()
    PASS &= _check_position_classification()
    PASS &= _check_utg_rfi()
    PASS &= _check_profile_divergence()
    PASS &= _check_raise_response_tree()
    PASS &= _check_limper_tree()
    PASS &= _check_short_stack()
    PASS &= _check_medium_stack()
    PASS &= _check_determinism_and_preflop_fuzz()
    print("=" * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED [PASS]' if PASS else 'SOME CHECKS FAILED [FAIL]'}")
    return PASS


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
