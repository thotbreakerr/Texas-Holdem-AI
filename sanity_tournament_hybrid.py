#!/usr/bin/env python3
"""Smoke tests for the Phase 2 TournamentHybridBot preflop strategy."""

from __future__ import annotations

import io
import os
import random
import sys
import copy
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

    # Phase 3: top pair top kicker facing a modest bet must continue, not fold.
    postflop_view = _view(
        [{"type": "fold"}, {"type": "call"}, {"type": "raise", "min": 40, "max": 120}],
        street="flop",
        board=[("A", "c"), ("7", "d"), ("2", "h")],
        to_call=20,
    )
    for profile in ("survival", "aggro"):
        postflop_bot = TournamentHybridBot(profile)
        postflop_action = postflop_bot.act(postflop_view)
        ok &= _valid_action(postflop_action, postflop_view.legal_actions)
        ok &= postflop_action.type in {"call", "raise"}
        ok &= postflop_bot.last_decision.get("path") == "postflop"
        ok &= "equity" in postflop_bot.last_decision

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


def _postflop_view(
    hole,
    board,
    *,
    street="flop",
    to_call=0,
    pot=60,
    opponents=("Villain1", "Villain2"),
    hero_stack=400,
    legal=None,
    hand_id=4000,
):
    opp_list = list(opponents)
    stacks = {"Hero": hero_stack}
    for opp in opp_list:
        stacks[opp] = 400
    if legal is None:
        if to_call > 0:
            min_total = max(2 * to_call, to_call + max(1, pot // 2))
            legal = [
                {"type": "fold"},
                {"type": "call"},
                {"type": "raise", "min": min(min_total, hero_stack), "max": hero_stack},
            ]
        else:
            legal = [
                {"type": "check"},
                {"type": "bet", "min": max(1, pot // 4), "max": hero_stack},
            ]
    return _view(
        legal,
        street=street,
        board=list(board),
        hole_cards=list(hole),
        pot=pot,
        to_call=to_call,
        max_raise=hero_stack,
        stacks=stacks,
        opponents=opp_list,
        seat_indices={pid: idx for idx, pid in enumerate(["Hero"] + opp_list)},
        acting_opponents=opp_list,
        all_in_opponents=[],
        hand_id=hand_id,
    )


def _check_postflop():
    ok = True

    # Draw detector unit checks.
    ok &= TournamentHybridBot._draw_category(
        [("K", "s"), ("Q", "s")], [("J", "s"), ("7", "s"), ("2", "d")]
    ) == "flush"
    ok &= TournamentHybridBot._draw_category(
        [("9", "h"), ("8", "d")], [("7", "c"), ("6", "s"), ("2", "h")]
    ) == "oeso"
    ok &= TournamentHybridBot._draw_category(
        [("9", "h"), ("5", "d")], [("7", "c"), ("6", "s"), ("2", "h")]
    ) == "gutshot"
    ok &= TournamentHybridBot._draw_category(
        [("K", "h"), ("3", "d")], [("8", "c"), ("6", "s"), ("2", "h")]
    ) == "none"

    # Monster with no bet to call value-bets.
    monster = _postflop_view(
        [("A", "s"), ("A", "d")],
        [("A", "c"), ("7", "d"), ("2", "h")],
        to_call=0,
        pot=60,
        hand_id=4001,
    )
    for profile in ("survival", "aggro"):
        bot = TournamentHybridBot(profile)
        action = bot.act(monster)
        ok &= _valid_action(action, monster.legal_actions)
        ok &= action.type in {"bet", "raise"}
        ok &= bot.last_decision.get("branch") == "postflop_value_bet"
        ok &= bot.last_decision.get("path") == "postflop"

    # Air facing a large bet folds.
    trash = _postflop_view(
        [("7", "s"), ("2", "d")],
        [("A", "c"), ("K", "d"), ("9", "h")],
        to_call=80,
        pot=100,
        hand_id=4002,
    )
    for profile in ("survival", "aggro"):
        bot = TournamentHybridBot(profile)
        action = bot.act(trash)
        ok &= _valid_action(action, trash.legal_actions)
        ok &= action.type == "fold"
        ok &= bot.last_decision.get("branch") == "postflop_fold"

    missing_hole = _postflop_view(
        [],
        [("A", "c"), ("K", "d"), ("9", "h")],
        to_call=20,
        legal=[{"type": "fold"}, {"type": "call"}],
        hand_id=4004,
    )
    malformed_hole = _postflop_view(
        [("Z", "z"), ("A", "s")],
        [("A", "c"), ("K", "d"), ("9", "h")],
        to_call=20,
        legal=[{"type": "fold"}, {"type": "call"}],
        hand_id=4005,
    )

    class _NoEquityBot(TournamentHybridBot):
        def _postflop_equity(self, view, hole, board, n_opp, sims):
            return None

    class _ExplodingPostflopBot(TournamentHybridBot):
        def _postflop_action(self, view, context):
            raise RuntimeError("intentional postflop failure")

    no_equity = _postflop_view(
        [("A", "s"), ("K", "s")],
        [("A", "c"), ("K", "d"), ("9", "h")],
        to_call=20,
        legal=[{"type": "fold"}, {"type": "call"}],
        hand_id=4006,
    )
    exploding = _postflop_view(
        [("A", "s"), ("K", "s")],
        [("A", "c"), ("K", "d"), ("9", "h")],
        to_call=20,
        legal=[{"type": "fold"}, {"type": "call"}],
        hand_id=4007,
    )
    for view in (missing_hole, malformed_hole):
        bot = TournamentHybridBot("survival")
        action = bot.act(view)
        ok &= _valid_action(action, view.legal_actions)
        ok &= action.type == "fold"
        ok &= bot.last_decision.get("branch") == "postflop_no_cards"
    no_equity_bot = _NoEquityBot("survival")
    no_equity_action = no_equity_bot.act(no_equity)
    ok &= _valid_action(no_equity_action, no_equity.legal_actions)
    ok &= no_equity_action.type == "fold"
    ok &= no_equity_bot.last_decision.get("branch") == "postflop_equity_error"
    exploding_bot = _ExplodingPostflopBot("survival")
    exploding_action = exploding_bot.act(exploding)
    ok &= _valid_action(exploding_action, exploding.legal_actions)
    ok &= exploding_action.type == "fold"
    ok &= exploding_bot.last_decision.get("branch") == "postflop_exception"

    class _FixedEquityBot(TournamentHybridBot):
        def _postflop_equity(self, view, hole, board, n_opp, sims):
            return 0.30

    call_for_less = _postflop_view(
        [("A", "s"), ("7", "d")],
        [("K", "c"), ("7", "h"), ("2", "d")],
        to_call=100,
        pot=120,
        opponents=("V",),
        hero_stack=40,
        legal=[{"type": "fold"}, {"type": "call"}],
        hand_id=4008,
    )
    fixed_bot = _FixedEquityBot("aggro")
    fixed_action = fixed_bot.act(call_for_less)
    ok &= _valid_action(fixed_action, call_for_less.legal_actions)
    ok &= fixed_action.type == "call"
    ok &= fixed_bot.last_decision.get("pot_odds") == 0.25
    ok &= fixed_bot.last_decision.get("stack_state", {}).get("commit_frac") == 1.0

    # Strong made hand facing a small bet continues (call or raise).
    strong = _postflop_view(
        [("A", "s"), ("A", "d")],
        [("A", "c"), ("7", "d"), ("2", "h")],
        to_call=15,
        pot=60,
        hand_id=4003,
    )
    for profile in ("survival", "aggro"):
        bot = TournamentHybridBot(profile)
        action = bot.act(strong)
        ok &= _valid_action(action, strong.legal_actions)
        ok &= action.type in {"call", "raise"}

    # Determinism: identical postflop spot resolves identically.
    repeat_bot = TournamentHybridBot("aggro")
    first = repeat_bot.act(strong)
    second = repeat_bot.act(strong)
    ok &= first == second

    # Aggro applies at least as much postflop pressure as survival.
    rng = random.Random(424242)
    aggro_aggressive = 0
    survival_aggressive = 0
    deck = [(r, s) for r in "23456789TJQKA" for s in "cdhs"]
    for i in range(60):
        cards = rng.sample(deck, 5)
        hole, board = cards[:2], cards[2:]
        street = rng.choice(["flop", "turn", "river"])
        if street == "turn":
            board = board + [rng.choice([c for c in deck if c not in cards])]
        elif street == "river":
            extra = rng.sample([c for c in deck if c not in cards], 2)
            board = board + extra
        view = _postflop_view(hole, board, street=street, to_call=0, pot=80,
                              hand_id=5000 + i)
        a_action = TournamentHybridBot("aggro").act(view)
        s_action = TournamentHybridBot("survival").act(view)
        ok &= _valid_action(a_action, view.legal_actions)
        ok &= _valid_action(s_action, view.legal_actions)
        if a_action.type in ("bet", "raise"):
            aggro_aggressive += 1
        if s_action.type in ("bet", "raise"):
            survival_aggressive += 1
    ok &= aggro_aggressive >= survival_aggressive

    # Legality fuzz across random postflop spots and legal sets.
    for i in range(80):
        cards = rng.sample(deck, rng.choice([5, 6, 7]))
        hole = cards[:2]
        board_len = rng.choice([3, 4, 5])
        board = cards[2:2 + board_len]
        street = {3: "flop", 4: "turn", 5: "river"}[len(board)]
        n_opp = rng.randint(1, 4)
        opponents = [f"V{j}" for j in range(n_opp)]
        to_call = rng.choice([0, 0, 10, 40, 120, 400])
        pot = rng.randint(20, 400)
        scenario = rng.choice(["normal", "check_only", "no_raise", "all_in_raise", "empty"])
        if scenario == "check_only":
            legal = [{"type": "check"}]
            to_call = 0
        elif scenario == "no_raise":
            legal = [{"type": "fold"}, {"type": "call"}]
            to_call = max(10, to_call)
        elif scenario == "all_in_raise":
            legal = [
                {"type": "fold"},
                {"type": "call"},
                {"type": "raise", "min": 400, "max": 400, "all_in": True, "reopens": False},
            ]
            to_call = max(10, to_call)
        elif scenario == "empty":
            legal = []
        else:
            legal = None
        view = _postflop_view(hole, board, street=street, to_call=to_call, pot=pot,
                              opponents=opponents, legal=legal, hand_id=6000 + i)
        for profile in ("survival", "aggro"):
            bot = TournamentHybridBot(profile)
            try:
                action = bot.act(view)
            except Exception:
                ok = False
                continue
            enforceable = bool(_legal_types(view.legal_actions) & {"check", "call", "fold"})
            enforceable |= any(
                _has_valid_aggressive_spec(view.legal_actions, typ)
                for typ in ("bet", "raise")
            )
            if enforceable:
                ok &= _valid_action(action, view.legal_actions)
            else:
                ok &= action.type == "fold"
            ok &= bot.last_decision is not None
            ok &= bot.last_decision.get("path") in {"postflop", "passive", "fallback"}

    print(f"[CHECK 14] {'PASS' if ok else 'FAIL'} - postflop equity, value/fold, "
          "draw detection, aggro pressure, and legality fuzz")
    return ok


def _rank_view(stacks, *, hand_id=15000, street="preflop", legal=None):
    if legal is None:
        legal = [{"type": "check"}]
    pids = ["Hero"] + [pid for pid in stacks if pid != "Hero"]
    return _view(
        legal,
        street=street,
        stacks=dict(stacks),
        opponents=[pid for pid in pids if pid != "Hero"],
        acting_opponents=[pid for pid in pids if pid != "Hero"],
        all_in_opponents=[],
        seat_indices={pid: idx for idx, pid in enumerate(pids)},
        hand_id=hand_id,
    )


def _ranked_rfi_view(position, hole, *, chips, stacks, hand_id):
    view = _rfi_view(position, hole, chips=chips, hand_id=hand_id)
    pids = ["Hero"] + [pid for pid in stacks if pid != "Hero"]
    view.stacks = dict(stacks)
    view.opponents = [pid for pid in pids if pid != "Hero"]
    view.acting_opponents = list(view.opponents)
    view.all_in_opponents = []
    view.seat_indices = {pid: idx for idx, pid in enumerate(pids)}
    return view


def _postflop_with_bb(view, *, bb=10):
    view.history = [{
        "street": "preflop",
        "pid": "Hero",
        "type": "raise",
        "amount": 2 * bb,
        "to_call_before": bb,
        "pot_before": round(1.5 * bb),
    }]
    view.min_raise = bb
    return view


def _check_stack_rank_phase4():
    ok = True
    bot = TournamentHybridBot("survival")

    dominant = _rank_view({
        "Hero": 1800, "A": 1000, "B": 900, "C": 800, "D": 760, "E": 740,
    }, hand_id=15001)
    ok &= bot._rank_context(dominant).get("bucket") == "chip_leader"

    flat_top = _rank_view({
        "Hero": 1020, "A": 1010, "B": 1000, "C": 1000, "D": 990, "E": 980,
    }, hand_id=15002)
    flat_bottom = _rank_view({
        "A": 1020, "B": 1010, "C": 1000, "D": 1000, "E": 990, "Hero": 980,
    }, hand_id=15003)
    ok &= bot._rank_context(flat_top).get("bucket") in {"top_2", "middle"}
    ok &= bot._rank_context(flat_bottom).get("bucket") in {"top_2", "middle"}

    tie_top = _rank_view({
        "Hero": 1000, "A": 1001, "B": 1000, "C": 999, "D": 998, "E": 997,
    }, hand_id=15004)
    ok &= bot._rank_context(tie_top).get("bucket") != "chip_leader"

    side_pot = _postflop_view(
        [("A", "s"), ("K", "s")],
        [("A", "c"), ("7", "d"), ("2", "h")],
        to_call=0,
        opponents=("A", "B"),
        hero_stack=1000,
        legal=[{"type": "check"}],
        hand_id=15006,
    )
    side_pot.stacks = {"Hero": 1000, "A": 1000, "B": 0}
    side_pot.acting_opponents = ["A"]
    side_pot.all_in_opponents = ["B"]
    side_bot = TournamentHybridBot("survival")
    side_bot.postflop_base_sims = 20
    side_bot.postflop_sim_cap = 20
    side_action = side_bot.act(side_pot)
    side_rank = side_bot._rank_context(side_pot)
    ok &= _valid_action(side_action, side_pot.legal_actions)
    ok &= side_rank.get("players_left") == 3
    ok &= side_rank.get("bucket") != "hu_even"
    ok &= side_bot.last_decision.get("players_left") == 3
    ok &= side_bot.last_decision.get("heads_up_weight") == 0.0

    bottom_40bb = _ranked_rfi_view(
        "BTN",
        [("K", "s"), ("6", "s")],
        chips=400,
        stacks={"Hero": 400, "A": 1200, "B": 1100, "C": 1000, "D": 900, "E": 800},
        hand_id=15005,
    )
    acted = TournamentHybridBot("survival")
    action = acted.act(bottom_40bb)
    rank_ctx = acted._rank_context(bottom_40bb)
    ok &= rank_ctx.get("bucket") == "short_stack"
    ok &= action.type == "fold"
    ok &= acted.last_decision.get("band") == "deep"
    ok &= acted.last_decision.get("stack_state", {}).get("short_weight") == 0.0

    print(f"[CHECK 15] {'PASS' if ok else 'FAIL'} - stack-rank buckets, flat tables, "
          "tie tolerance, and non-desperate 40bb short rank")
    return ok


def _check_desperation_not_rank_phase4():
    ok = True
    short_stacks = {"Hero": 60, "CL": 80, "V1": 50, "V2": 40, "V3": 30, "V4": 20}
    shove_trace = None
    shove_action = None
    for hand_id in range(16000, 16200):
        view = _ranked_rfi_view(
            "BTN",
            [("K", "s"), ("6", "s")],
            chips=60,
            stacks=short_stacks,
            hand_id=hand_id,
        )
        bot = TournamentHybridBot("survival")
        action = bot.act(view)
        if action.type == "raise":
            shove_trace = bot.last_decision
            shove_action = action
            break

    ok &= shove_action is not None and shove_action.type == "raise"
    ok &= shove_trace is not None
    if shove_trace:
        ok &= shove_trace.get("band") == "short"
        ok &= shove_trace.get("branch") == "short_open_shove"
        ok &= shove_trace.get("base_range_hit") is False
        ok &= shove_trace.get("range_expansion_hit") is True
        ok &= shove_trace.get("future_edge_tax") == 0.0
        ok &= shove_trace.get("rank_ctx", {}).get("bucket") == "top_2"

    deep_last = _ranked_rfi_view(
        "BTN",
        [("K", "s"), ("6", "s")],
        chips=400,
        stacks={"Hero": 400, "A": 1200, "B": 1100, "C": 1000, "D": 900, "E": 800},
        hand_id=16201,
    )
    bot = TournamentHybridBot("survival")
    action = bot.act(deep_last)
    ok &= action.type == "fold"
    ok &= bot.last_decision.get("band") == "deep"
    ok &= bot.last_decision.get("rank_ctx", {}).get("bucket") == "short_stack"
    ok &= bot.last_decision.get("stack_state", {}).get("short_weight") == 0.0
    ok &= not bot._action_is_all_in(deep_last, action)

    print(f"[CHECK 16] {'PASS' if ok else 'FAIL'} - 6bb desperation overrides rank, "
          "while 40bb last place does not panic jam")
    return ok


def _rfi_pressure_count(profile, hole, stacks, *, start=17000, samples=200):
    count = 0
    last_trace = None
    for hand_id in range(start, start + samples):
        view = _ranked_rfi_view("BTN", hole, chips=2000, stacks=stacks, hand_id=hand_id)
        bot = TournamentHybridBot(profile)
        action = bot.act(view)
        if action.type == "raise":
            count += 1
        last_trace = bot.last_decision
    return count, last_trace


def _pressure_postflop_trace(profile, *, hand_id=17500):
    view = _postflop_view(
        [("7", "s"), ("2", "d")],
        [("A", "c"), ("K", "d"), ("9", "h")],
        to_call=0,
        pot=80,
        opponents=("A", "B", "C", "D", "E"),
        hero_stack=2000,
        hand_id=hand_id,
    )
    view = _postflop_with_bb(view)
    view.stacks = {"Hero": 2000, "A": 500, "B": 500, "C": 500, "D": 500, "E": 500}
    view.seat_indices = {pid: idx for idx, pid in enumerate(["Hero", "A", "B", "C", "D", "E"])}
    view.acting_opponents = ["A", "B", "C", "D", "E"]
    bot = TournamentHybridBot(profile)
    bot.postflop_base_sims = 30
    bot.postflop_sim_cap = 30
    bot.act(view)
    return bot.last_decision


def _check_chipleader_pressure_phase4():
    ok = True
    leader = {"Hero": 2000, "A": 500, "B": 500, "C": 500, "D": 500, "E": 500}
    flat = {"Hero": 2000, "A": 2000, "B": 2000, "C": 2000, "D": 2000, "E": 2000}

    pressure_hand = [("Q", "h"), ("8", "d")]
    survival_base, _ = _rfi_pressure_count(
        "survival", pressure_hand, flat, start=17000
    )
    survival_pressure, survival_trace = _rfi_pressure_count(
        "survival", pressure_hand, leader, start=17000
    )
    aggro_base, _ = _rfi_pressure_count(
        "aggro", pressure_hand, flat, start=17000
    )
    aggro_pressure, aggro_trace = _rfi_pressure_count(
        "aggro", pressure_hand, leader, start=17000
    )

    ok &= survival_pressure > survival_base
    ok &= aggro_pressure > aggro_base
    ok &= aggro_pressure >= survival_pressure
    ok &= survival_trace.get("range_nudge_pp", 0.0) <= 0.08
    ok &= aggro_trace.get("range_nudge_pp", 0.0) <= 0.12
    ok &= survival_trace.get("pressure_weight", 0.0) > 0.0
    ok &= aggro_trace.get("pressure_weight", 0.0) >= survival_trace.get("pressure_weight", 0.0)

    s_post = _pressure_postflop_trace("survival", hand_id=17501)
    a_post = _pressure_postflop_trace("aggro", hand_id=17502)
    ok &= s_post.get("freq_after", 0.0) > s_post.get("freq_before", 0.0)
    ok &= a_post.get("freq_after", 0.0) > a_post.get("freq_before", 0.0)
    ok &= a_post.get("freq_after", 0.0) >= s_post.get("freq_after", 0.0)
    ok &= s_post.get("value_threshold_before") == s_post.get("value_threshold_after")
    ok &= a_post.get("value_threshold_before") == a_post.get("value_threshold_after")
    ok &= s_post.get("freq_after", 0.0) <= min(
        s_post.get("freq_before", 0.0) * 1.25,
        s_post.get("freq_before", 0.0) + 0.15,
        0.92,
    ) + 1e-9
    ok &= a_post.get("freq_after", 0.0) <= min(
        a_post.get("freq_before", 0.0) * 1.40,
        a_post.get("freq_before", 0.0) + 0.22,
        0.96,
    ) + 1e-9

    print(f"[CHECK 17] {'PASS' if ok else 'FAIL'} - chip-leader pressure widens "
          "ranges/frequencies, aggro >= survival, value thresholds unchanged")
    return ok


def _hu_postflop_open_trace(profile, *, hand_id=18000):
    view = _postflop_view(
        [("7", "s"), ("2", "d")],
        [("A", "c"), ("K", "d"), ("9", "h")],
        to_call=0,
        pot=80,
        opponents=("V",),
        hero_stack=1000,
        hand_id=hand_id,
    )
    view = _postflop_with_bb(view)
    view.stacks = {"Hero": 1000, "V": 1000}
    view.seat_indices = {"Hero": 0, "V": 1}
    view.acting_opponents = ["V"]
    bot = TournamentHybridBot(profile)
    bot.postflop_base_sims = 30
    bot.postflop_sim_cap = 30
    bot.act(view)
    return bot.last_decision


def _hu_postflop_call_trace(profile, *, to_call=40, hand_id=18100):
    view = _postflop_view(
        [("A", "s"), ("7", "d")],
        [("K", "c"), ("7", "h"), ("2", "d")],
        to_call=to_call,
        pot=max(100, to_call),
        opponents=("V",),
        hero_stack=1000,
        legal=[{"type": "fold"}, {"type": "call"}],
        hand_id=hand_id,
    )
    view = _postflop_with_bb(view)
    view.stacks = {"Hero": 1000, "V": 1000}
    view.seat_indices = {"Hero": 0, "V": 1}
    view.acting_opponents = ["V"]
    bot = TournamentHybridBot(profile)
    bot.postflop_base_sims = 30
    bot.postflop_sim_cap = 30
    bot.act(view)
    return bot.last_decision


def _check_heads_up_phase4():
    ok = True
    for profile in ("survival", "aggro"):
        open_trace = _hu_postflop_open_trace(profile, hand_id=18000 + (0 if profile == "survival" else 1))
        ok &= open_trace.get("players_left") == 2
        ok &= open_trace.get("heads_up_weight", 0.0) > 0.0
        ok &= open_trace.get("freq_after", 0.0) > open_trace.get("freq_before", 0.0)
        ok &= open_trace.get("value_threshold_after", 1.0) < open_trace.get("value_threshold_before", 0.0)

        call_trace = _hu_postflop_call_trace(profile, to_call=40, hand_id=18100 + (0 if profile == "survival" else 1))
        ok &= call_trace.get("call_cushion_after", 1.0) < call_trace.get("call_cushion_before", 0.0)
        ok &= call_trace.get("future_edge_tax") == 0.0

        all_in_trace = _hu_postflop_call_trace(profile, to_call=1000, hand_id=18200 + (0 if profile == "survival" else 1))
        ok &= all_in_trace.get("spot_type") == "all_in_call"
        ok &= all_in_trace.get("call_cushion_after") == all_in_trace.get("call_cushion_before")
        ok &= 0.0 < all_in_trace.get("future_edge_tax", 0.0) <= (0.010 if profile == "survival" else 0.004)

    print(f"[CHECK 18] {'PASS' if ok else 'FAIL'} - heads-up aggression loosens "
          "non-all-in play while preserving a small deep stack-off premium")
    return ok


def _check_phase4_invariants():
    ok = True
    view = _ranked_rfi_view(
        "BTN",
        [("K", "s"), ("8", "s")],
        chips=2000,
        stacks={"Hero": 2000, "A": 500, "B": 500, "C": 500, "D": 500, "E": 500},
        hand_id=19001,
    )
    bot = TournamentHybridBot("survival")
    first = bot.act(view)
    second = bot.act(view)
    ok &= first == second
    ok &= bot.last_decision.get("rank_ctx", {}).get("bucket") == "chip_leader"

    cached_bot = TournamentHybridBot("survival")
    cache_first = _rank_view({"Hero": 2000, "A": 500, "B": 500}, hand_id=19002)
    cache_second = _rank_view({"Hero": 100, "A": 2000, "B": 0}, hand_id=19002)
    first_ctx = cached_bot._rank_context(cache_first)
    second_ctx = cached_bot._rank_context(cache_second)
    ok &= second_ctx == first_ctx
    ok &= second_ctx.get("hero_stack") == 2000

    clock_default = _rank_view({"Hero": 1000, "A": 1000}, hand_id=38)
    clock_custom = _rank_view({"Hero": 1000, "A": 1000}, hand_id=38)
    clock_disabled = _rank_view({"Hero": 1000, "A": 1000}, hand_id=38)
    clock_custom.blind_increase_every = 20
    clock_disabled.blind_increase_every = 0
    ok &= TournamentHybridBot._hands_until_blind_up(clock_default) == 12
    ok &= TournamentHybridBot._hands_until_blind_up(clock_custom) == 2
    ok &= TournamentHybridBot._hands_until_blind_up(clock_disabled) > 1_000_000

    rng = random.Random(20260620)
    deck = [(r, s) for r in "23456789TJQKA" for s in "cdhs"]
    for i in range(100):
        n = rng.randint(2, 6)
        positions = _positions(n)
        hero_pos = rng.choice(positions)
        chips = rng.randint(40, 1800)
        stacks = {"Hero": chips}
        for j in range(n - 1):
            stacks[f"V{j}"] = rng.randint(1, 2200)
        hole = rng.sample(deck, 2)
        if rng.choice([True, False]):
            fuzz = _ranked_rfi_view(hero_pos, hole, chips=chips, stacks=stacks, hand_id=19100 + i)
        else:
            board = rng.sample([c for c in deck if c not in hole], rng.choice([3, 4, 5]))
            fuzz = _postflop_view(
                hole,
                board,
                street={3: "flop", 4: "turn", 5: "river"}[len(board)],
                to_call=rng.choice([0, 10, 40, 120]),
                pot=rng.randint(20, 300),
                opponents=tuple(pid for pid in stacks if pid != "Hero"),
                hero_stack=chips,
                hand_id=19100 + i,
            )
            fuzz = _postflop_with_bb(fuzz, bb=10)
            fuzz.stacks = stacks
            fuzz.seat_indices = {pid: idx for idx, pid in enumerate(["Hero"] + [pid for pid in stacks if pid != "Hero"])}
            fuzz.acting_opponents = [pid for pid in stacks if pid != "Hero"]
        for profile in ("survival", "aggro"):
            active = TournamentHybridBot(profile)
            active.postflop_base_sims = 20
            active.postflop_sim_cap = 20
            try:
                action = active.act(fuzz)
            except Exception:
                ok = False
                continue
            ok &= _valid_action(action, fuzz.legal_actions)
            ok &= active.last_decision.get("path") in {"preflop", "postflop", "passive", "fallback"}
            ok &= "rank_ctx" in active.last_decision.get("context", {})

    reset_bot = TournamentHybridBot("survival")
    reset_bot.act(view)
    ok &= bool(reset_bot._rank_cache)
    reset_bot.reset_memory()
    ok &= reset_bot._rank_cache == {}

    print(f"[CHECK 19] {'PASS' if ok else 'FAIL'} - Phase 4 determinism, legality "
          "fuzz, trace path set, rank-cache reset, and blind-clock schedule")
    return ok


class _FixedEquityBot(TournamentHybridBot):
    def __init__(self, profile="survival", equity=0.10):
        super().__init__(profile)
        self.fixed_equity = equity
        self.postflop_base_sims = 1
        self.postflop_big_sims = 1
        self.postflop_allin_sims = 1
        self.postflop_sim_cap = 1

    def _postflop_equity(self, view, hole, board, n_opp, sims):
        return self.fixed_equity


def _p5_history_entry(street, pid, action_type, *, amount=None, to_call_before=0, pot_before=0):
    return {
        "street": street,
        "pid": pid,
        "type": action_type,
        "amount": amount,
        "to_call_before": to_call_before,
        "pot_before": pot_before,
    }


def _p5_counter_view(history, *, hand_id=20000, opponents=("V",), all_in=()):
    view = _postflop_view(
        [("7", "s"), ("2", "d")],
        [("A", "c"), ("K", "d"), ("9", "h")],
        to_call=0,
        pot=120,
        opponents=opponents,
        legal=[{"type": "check"}],
        hand_id=hand_id,
    )
    view.history = list(history)
    view.all_in_opponents = list(all_in)
    for pid in all_in:
        view.stacks[pid] = 0
    return view


def _p5_feed_station(bot, pid="V", *, start=21000, calls=30):
    for i in range(calls):
        history = [
            _p5_history_entry("flop", "Hero", "bet", amount=40, pot_before=80),
            _p5_history_entry("flop", pid, "call", amount=40, to_call_before=40, pot_before=120),
        ]
        bot.act(_p5_counter_view(history, hand_id=start + i, opponents=(pid,)))


def _p5_feed_overfolder(bot, pid="V", *, start=22000, folds=12):
    for i in range(folds):
        history = [
            _p5_history_entry("flop", "Hero", "bet", amount=40, pot_before=80),
            _p5_history_entry("flop", pid, "fold", to_call_before=40, pot_before=120),
        ]
        bot.act(_p5_counter_view(history, hand_id=start + i, opponents=(pid,)))


def _p5_feed_spewy(bot, pid="V", *, start=23000, short=False):
    amount = 80 if short else 150
    for i in range(3):
        history = [
            _p5_history_entry("preflop", pid, "raise", amount=30, to_call_before=10, pot_before=15),
            _p5_history_entry("flop", pid, "bet", amount=amount, pot_before=80),
        ]
        bot.act(_p5_counter_view(history, hand_id=start + i, opponents=(pid,), all_in=(pid,)))


def _p5_r2_view(*, hand_id=24000, opponents=("V",), to_call=1000, hero_stack=1000,
                 all_in=True):
    view = _postflop_view(
        [("A", "s"), ("7", "d")],
        [("K", "c"), ("7", "h"), ("2", "d")],
        to_call=to_call,
        pot=1000,
        opponents=opponents,
        hero_stack=hero_stack,
        legal=[{"type": "fold"}, {"type": "call"}],
        hand_id=hand_id,
    )
    view.history = [
        _p5_history_entry("preflop", "Hero", "raise", amount=20, to_call_before=10, pot_before=15),
        _p5_history_entry("flop", "V", "bet", amount=to_call, pot_before=1000),
    ]
    if all_in:
        view.all_in_opponents = ["V"]
        view.stacks["V"] = 0
    return view


def _strip_p5(trace):
    return {k: v for k, v in (trace or {}).items() if not str(k).startswith("p5_")}


def _check_p5_accumulation_dedup():
    bot = _FixedEquityBot()
    prefix = [
        _p5_history_entry("preflop", "V", "call", amount=10, to_call_before=10, pot_before=15),
    ]
    full = prefix + [
        _p5_history_entry("flop", "V", "call", amount=30, to_call_before=30, pot_before=60),
    ]
    bot.act(_p5_counter_view(prefix, hand_id=20001))
    bot.act(_p5_counter_view(full, hand_id=20001))
    snap = copy.deepcopy(bot._profiles.all_raw())
    bot.act(_p5_counter_view(full, hand_id=20001))
    raw = bot._profiles.raw("V")
    ok = (
        snap == bot._profiles.all_raw()
        and raw["preflop_action_seen"] == 1
        and raw["vpip_seen"] == 1
        and raw["postflop_pressure_call"] == 1
    )
    print(f"[CHECK 20] {'PASS' if ok else 'FAIL'} - P5 accumulation and dedup across overlapping prefixes")
    return ok


def _check_p5_once_per_hand_vpip():
    bot = _FixedEquityBot()
    history = [
        _p5_history_entry("preflop", "V", "call", amount=10, to_call_before=10, pot_before=15),
        _p5_history_entry("preflop", "Hero", "raise", amount=40, to_call_before=10, pot_before=25),
        _p5_history_entry("preflop", "V", "call", amount=30, to_call_before=30, pot_before=65),
        _p5_history_entry("preflop", "V", "raise", amount=90, to_call_before=0, pot_before=95),
    ]
    bot.act(_p5_counter_view(history, hand_id=21001))
    raw = bot._profiles.raw("V")
    ok = (
        raw["preflop_action_seen"] == 1
        and raw["vpip_seen"] == 1
        and raw["preflop_raise_seen"] == 1
    )
    print(f"[CHECK 21] {'PASS' if ok else 'FAIL'} - P5 once-per-hand VPIP and preflop aggression")
    return ok


def _check_p5_pressure_state_machine():
    bot = _FixedEquityBot()
    history = [
        _p5_history_entry("preflop", "V", "call", amount=10, to_call_before=10, pot_before=15),
        _p5_history_entry("preflop", "BB", "check", to_call_before=0, pot_before=25),
        _p5_history_entry("preflop", "Hero", "raise", amount=40, to_call_before=10, pot_before=25),
        _p5_history_entry("preflop", "V", "fold", to_call_before=30, pot_before=65),
        _p5_history_entry("flop", "V", "call", amount=20, to_call_before=20, pot_before=80),
    ]
    bot.act(_p5_counter_view(history, hand_id=22001, opponents=("V", "BB")))
    v = bot._profiles.raw("V")
    bb = bot._profiles.raw("BB")
    ok = (
        v["vpip_seen"] == 1
        and v["pressure_fold"] == 1
        and v["postflop_pressure_call"] == 1
        and bb["vpip_seen"] == 0
        and bb["pressure_fold"] == 0
        and bb["pressure_call"] == 0
        and bb["pressure_raise"] == 0
    )
    print(f"[CHECK 22] {'PASS' if ok else 'FAIL'} - P5 facing-pressure state machine")
    return ok


def _check_p5_censoring_discipline():
    ok = True
    bot = _FixedEquityBot()
    hero_bet_only = [_p5_history_entry("flop", "Hero", "bet", amount=50, pot_before=80)]
    bot.act(_p5_counter_view(hero_bet_only, hand_id=23001))
    ok &= bot._profiles.all_raw() == {}
    bot.act(_p5_counter_view(hero_bet_only, hand_id=23001))
    ok &= bot._profiles.all_raw() == {}

    folded = _FixedEquityBot()
    _p5_feed_overfolder(folded, folds=5)
    fstats = folded._profiles.stat_summary("V")
    ok &= fstats["station_response_n"] == 0
    ok &= folded._profiles.read_strength("V", "station_score", threshold=0.60, band=0.25) == 0.0

    caller = _FixedEquityBot()
    _p5_feed_station(caller, calls=5)
    cstats = caller._profiles.stat_summary("V")
    ok &= cstats["fold_to_pressure_hat"] < 0.58
    ok &= caller._profiles.read_strength("V", "fold_to_pressure", threshold=0.58, band=0.17) == 0.0

    print(f"[CHECK 23] {'PASS' if ok else 'FAIL'} - P5 censored-fold discipline")
    return ok


def _p5_trace_for_open(bot, *, is_pfr=False, equity=0.10, hand_id=24001):
    bot.fixed_equity = equity
    view = _postflop_view(
        [("7", "s"), ("2", "d")],
        [("A", "c"), ("K", "d"), ("9", "h")],
        to_call=0,
        pot=80,
        opponents=("V",),
        hero_stack=1000,
        hand_id=hand_id,
    )
    if is_pfr:
        view = _postflop_with_bb(view)
    bot.act(view)
    return bot.last_decision


def _check_p5_station_suppression_active():
    ok = True

    p4_cbet = _FixedEquityBot(equity=0.10)
    p5_cbet = _FixedEquityBot(equity=0.10)
    p5_cbet.p5_enabled = True
    _p5_feed_station(p4_cbet, start=24100)
    _p5_feed_station(p5_cbet, start=24100)
    c4 = _p5_trace_for_open(p4_cbet, is_pfr=True, hand_id=24201)
    c5 = _p5_trace_for_open(p5_cbet, is_pfr=True, hand_id=24201)
    ok &= c5["freq_after"] < c4["freq_after"]
    ok &= 0.70 <= (c5["freq_after"] / c4["freq_after"]) <= 0.90
    ok &= c5.get("p5_station_applied") is True

    p4_bluff = _FixedEquityBot(equity=0.10)
    p5_bluff = _FixedEquityBot(equity=0.10)
    p5_bluff.p5_enabled = True
    _p5_feed_station(p4_bluff, start=24300)
    _p5_feed_station(p5_bluff, start=24300)
    b4 = _p5_trace_for_open(p4_bluff, is_pfr=False, hand_id=24401)
    b5 = _p5_trace_for_open(p5_bluff, is_pfr=False, hand_id=24401)
    ok &= b5["freq_after"] < b4["freq_after"]
    ok &= 0.45 <= (b5["freq_after"] / b4["freq_after"]) <= 0.65

    p4_semi = _FixedEquityBot(equity=0.35)
    p5_semi = _FixedEquityBot(equity=0.35)
    p5_semi.p5_enabled = True
    _p5_feed_station(p4_semi, start=24500)
    _p5_feed_station(p5_semi, start=24500)
    semi_view = _postflop_view(
        [("K", "s"), ("Q", "s")],
        [("J", "s"), ("7", "s"), ("2", "d")],
        to_call=0,
        pot=80,
        opponents=("V",),
        hero_stack=1000,
        hand_id=24601,
    )
    p4_semi.act(semi_view)
    p5_semi.act(semi_view)
    ok &= p5_semi.last_decision.get("spot_type") == "semibluff"
    ok &= p5_semi.last_decision.get("freq_after") == p4_semi.last_decision.get("freq_after")
    ok &= p5_semi.last_decision.get("p5_station_applied") is False

    p5_raise = _FixedEquityBot(equity=0.10)
    p5_raise.p5_enabled = True
    _p5_feed_station(p5_raise, start=24700)
    raise_view = _postflop_view(
        [("7", "s"), ("2", "d")],
        [("A", "c"), ("K", "d"), ("9", "h")],
        to_call=40,
        pot=100,
        opponents=("V",),
        hero_stack=1000,
        hand_id=24801,
    )
    raise_action = p5_raise.act(raise_view)
    ok &= raise_action.type != "raise"

    p4_value = _FixedEquityBot(equity=0.90)
    p5_value = _FixedEquityBot(equity=0.90)
    p5_value.p5_enabled = True
    _p5_feed_station(p4_value, start=24900)
    _p5_feed_station(p5_value, start=24900)
    v4 = _p5_trace_for_open(p4_value, is_pfr=False, equity=0.90, hand_id=25001)
    v5 = _p5_trace_for_open(p5_value, is_pfr=False, equity=0.90, hand_id=25001)
    ok &= v5.get("branch") == "postflop_value_bet"
    ok &= v5.get("value_threshold_after") == v4.get("value_threshold_after")

    print(f"[CHECK 24] {'PASS' if ok else 'FAIL'} - P5 station suppression active only in safe directions")
    return ok


def _check_p5_neutral_default_reset():
    p4 = _FixedEquityBot(equity=0.10)
    p5 = _FixedEquityBot(equity=0.10)
    p5.p5_enabled = True
    view = _postflop_view(
        [("7", "s"), ("2", "d")],
        [("A", "c"), ("K", "d"), ("9", "h")],
        to_call=0,
        pot=80,
        opponents=("V",),
        hero_stack=1000,
        hand_id=26001,
    )
    a4 = p4.act(view)
    a5 = p5.act(view)
    _p5_feed_station(p5, start=26100, calls=4)
    p5.reset_memory()
    ok = (
        TournamentHybridBot().p5_enabled is False
        and a4 == a5
        and p5._profiles.all_raw() == {}
        and p5.last_decision is None
        and p5.p5_error_count == 0
    )
    print(f"[CHECK 25] {'PASS' if ok else 'FAIL'} - P5 neutral default and reset")
    return ok


def _check_p5_idempotent_prefix_replay():
    bot = _FixedEquityBot()
    prefix = [_p5_history_entry("preflop", "V", "call", amount=10, to_call_before=10, pot_before=15)]
    full = prefix + [_p5_history_entry("flop", "V", "call", amount=30, to_call_before=30, pot_before=60)]
    bot.act(_p5_counter_view(prefix, hand_id=27001))
    one = copy.deepcopy(bot._profiles.all_raw())
    bot.act(_p5_counter_view(full, hand_id=27001))
    two = copy.deepcopy(bot._profiles.all_raw())
    bot.act(_p5_counter_view(full, hand_id=27001))
    three = copy.deepcopy(bot._profiles.all_raw())
    raw = bot._profiles.raw("V")
    ok = (
        one != two
        and two == three
        and raw["vpip_seen"] == 1
        and raw["postflop_pressure_call"] == 1
    )
    print(f"[CHECK 26] {'PASS' if ok else 'FAIL'} - P5 idempotent prefix replay preserves counter bytes")
    return ok


def _check_p5_off_log_only_parity():
    ok = True
    views = [
        _rfi_view("BTN", [("J", "s"), ("8", "s")], hand_id=28001),
        _facing_open_view([("A", "s"), ("Q", "s")], opener="UTG", hand_id=28002),
        _limped_view([("6", "s"), ("5", "s")], hero_pos="BTN", limpers=2, hand_id=28003),
        _postflop_view([("7", "s"), ("2", "d")], [("A", "c"), ("K", "d"), ("9", "h")],
                       to_call=0, opponents=("V",), hand_id=28004),
        _postflop_view([("7", "s"), ("2", "d")], [("A", "c"), ("K", "d"), ("9", "h")],
                       to_call=40, opponents=("V",), legal=[{"type": "fold"}, {"type": "call"}],
                       hand_id=28005),
    ]
    malformed = _postflop_view(
        [("7", "s"), ("2", "d")],
        [("A", "c"), ("K", "d"), ("9", "h")],
        to_call=0,
        opponents=("V",),
        hand_id=28006,
    )
    malformed.history = [None, {"pid": "V", "type": "call"}, "bad"]
    views.append(malformed)

    for idx, view in enumerate(views):
        p4 = _FixedEquityBot(equity=0.10)
        off = _FixedEquityBot(equity=0.10)
        log = _FixedEquityBot(equity=0.10)
        log.p5_enabled = True
        log.p5_log_only = True
        _p5_feed_station(log, start=28100 + idx * 100)
        a4 = p4.act(view)
        a_off = off.act(view)
        a_log = log.act(view)
        p4_keys = set(_strip_p5(p4.last_decision).keys())
        ok &= a4 == a_off == a_log
        ok &= set(_strip_p5(off.last_decision).keys()) == p4_keys
        ok &= set(_strip_p5(log.last_decision).keys()) == p4_keys
        ok &= all(str(k).startswith("p5_") or k in p4.last_decision for k in off.last_decision)

    print(f"[CHECK 27] {'PASS' if ok else 'FAIL'} - P5 off/log-only Phase 4 parity and trace namespace")
    return ok


def _check_p5_log_only_overfolder():
    p4 = _FixedEquityBot(equity=0.10)
    p5 = _FixedEquityBot(equity=0.10)
    p5.p5_enabled = True
    _p5_feed_overfolder(p5, folds=20)
    view = _postflop_view(
        [("7", "s"), ("2", "d")],
        [("A", "c"), ("K", "d"), ("9", "h")],
        to_call=0,
        opponents=("V",),
        hand_id=29001,
    )
    a4 = p4.act(view)
    a5 = p5.act(view)
    ok = (
        a4 == a5
        and p5.last_decision.get("p5_overfolder_delta_log_only", 0.0) > 0.0
    )
    print(f"[CHECK 28] {'PASS' if ok else 'FAIL'} - P5 overfolder widening stays log-only")
    return ok


def _check_p5_log_only_passive_tight():
    p4 = _FixedEquityBot(equity=0.30)
    p5 = _FixedEquityBot(equity=0.30)
    p5.p5_enabled = True
    for i in range(8):
        history = [
            _p5_history_entry("preflop", "V", "fold", to_call_before=10, pot_before=15),
            _p5_history_entry("flop", "V", "check", pot_before=80),
        ]
        p5.act(_p5_counter_view(history, hand_id=30000 + i))
    view = _postflop_view(
        [("A", "s"), ("7", "d")],
        [("K", "c"), ("7", "h"), ("2", "d")],
        to_call=40,
        pot=100,
        opponents=("V",),
        legal=[{"type": "fold"}, {"type": "call"}],
        hand_id=30099,
    )
    a4 = p4.act(view)
    a5 = p5.act(view)
    ok = (
        a4 == a5
        and p5.last_decision.get("p5_passive_tight_delta_log_only", 0.0) > 0.0
    )
    print(f"[CHECK 29] {'PASS' if ok else 'FAIL'} - P5 passive/tight flip stays log-only")
    return ok


def _check_p5_strict_r2_scope():
    ok = True
    p4 = _FixedEquityBot(equity=0.575)
    p5 = _FixedEquityBot(equity=0.575)
    p5.p5_enabled = True
    _p5_feed_spewy(p5, start=31000)
    view = _p5_r2_view(hand_id=31100)
    a4 = p4.act(view)
    a5 = p5.act(view)
    t = p5.last_decision
    ok &= a4.type == "fold" and a5.type == "call"
    ok &= t.get("p5_r2_relief_applied") is True
    ok &= 0.0 < t.get("future_edge_tax", 0.0) < t.get("p5_r2_tax_before", 0.0)
    ok &= t.get("future_edge_tax") == t.get("p5_r2_tax_after_proposed")

    open_jam = _postflop_view(
        [("A", "s"), ("A", "d")],
        [("K", "c"), ("7", "h"), ("2", "d")],
        to_call=0,
        pot=1000,
        opponents=("V",),
        hero_stack=1000,
        legal=[{"type": "check"}, {"type": "bet", "min": 1000, "max": 1000, "all_in": True}],
        hand_id=31101,
    )
    p5.act(open_jam)
    ok &= p5.last_decision.get("p5_r2_relief_applied") is False

    p5.act(_p5_r2_view(hand_id=31102, to_call=500, all_in=False))
    ok &= p5.last_decision.get("p5_r2_relief_applied") is False

    p5.act(_p5_r2_view(hand_id=31103, opponents=("V", "W")))
    ok &= p5.last_decision.get("p5_r2_relief_applied") is False

    tight = _FixedEquityBot(equity=0.575)
    tight.p5_enabled = True
    tight.act(_p5_r2_view(hand_id=31104))
    ok &= tight.last_decision.get("p5_r2_relief_applied") is False

    short = _FixedEquityBot(equity=0.575)
    short.p5_enabled = True
    _p5_feed_spewy(short, start=31200, short=True)
    short.act(_p5_r2_view(hand_id=31299))
    ok &= short.last_decision.get("p5_r2_relief_applied") is False
    ok &= short.last_decision.get("p5_r2_short_jam_like_count", 0) > 0

    neg = _FixedEquityBot(equity=0.40)
    neg.p5_enabled = True
    _p5_feed_spewy(neg, start=31300)
    neg.act(_p5_r2_view(hand_id=31399))
    ok &= neg.last_decision.get("p5_r2_relief_applied") is False
    ok &= neg.last_decision.get("p5_r2_chip_ev_ok") is False

    print(f"[CHECK 30] {'PASS' if ok else 'FAIL'} - P5 strict R2 fires only in scoped all-in call spots")
    return ok


def _check_p5_reset_across_tournaments():
    bot = _FixedEquityBot()
    _p5_feed_station(bot, start=32000, calls=8)
    ok = bot._profiles.raw("V")["postflop_pressure_call"] == 8
    bot.act(_p5_counter_view([], hand_id=1))
    ok &= bot._profiles.all_raw() == {}
    fresh = _FixedEquityBot()
    ok &= fresh._profiles.all_raw() == {}
    print(f"[CHECK 31] {'PASS' if ok else 'FAIL'} - P5 reset/regression boundary prevents tournament leakage")
    return ok


def _check_p5_log_only_multi_read_precedence():
    p4 = _FixedEquityBot(equity=0.10)
    p5 = _FixedEquityBot(equity=0.10)
    p5.p5_enabled = True
    p5.p5_log_only = True
    _p5_feed_overfolder(p5, pid="F", start=33000, folds=20)
    _p5_feed_station(p5, pid="S", start=33100, calls=20)
    view = _postflop_view(
        [("7", "s"), ("2", "d")],
        [("A", "c"), ("K", "d"), ("9", "h")],
        to_call=0,
        opponents=("F", "S"),
        hand_id=33200,
    )
    a4 = p4.act(view)
    a5 = p5.act(view)
    ok = (
        a4 == a5
        and p5.last_decision.get("p5_overfolder_delta_log_only", 0.0) > 0.0
        and p5.last_decision.get("p5_station_value_delta_log_only", 0.0) > 0.0
        and p5.last_decision.get("p5_station_applied") is False
    )
    print(f"[CHECK 32] {'PASS' if ok else 'FAIL'} - P5 multi-read positive modules remain log-only")
    return ok


def _check_p5_fail_closed():
    class _ExplodingP5Bot(_FixedEquityBot):
        def _p5_station_read(self, view):
            raise RuntimeError("intentional p5 failure")

    p4 = _FixedEquityBot(equity=0.10)
    bad = _ExplodingP5Bot(equity=0.10)
    bad.p5_enabled = True
    view = _postflop_view(
        [("7", "s"), ("2", "d")],
        [("A", "c"), ("K", "d"), ("9", "h")],
        to_call=0,
        opponents=("V",),
        hand_id=34001,
    )
    a4 = p4.act(view)
    a_bad = bad.act(view)
    ok = (
        a4 == a_bad
        and bad.p5_error_count == 1
        and bad.last_decision.get("p5_error_count") == 1
    )
    print(f"[CHECK 33] {'PASS' if ok else 'FAIL'} - P5 fail-closed errors preserve Phase 4 action")
    return ok


def _check_p5_zero_data_neutral():
    profiles = TournamentHybridBot()._profiles
    stats = [
        ("vpip", 0.36, 0.24),
        ("preflop_aggression_rate", 0.18, 0.22),
        ("postflop_aggression_freq", 0.38, 0.25),
        ("fold_to_pressure", 0.58, 0.17),
        ("station_score", 0.60, 0.25),
        ("blind_fold_to_steal", 0.58, 0.17),
        ("fold_to_cbet", 0.58, 0.17),
        ("threebet_rate", 0.16, 0.20),
        ("fourbet_rate", 0.08, 0.12),
    ]
    ok = all(
        profiles.read_strength("Nobody", stat, threshold=thr, band=band) == 0.0
        for stat, thr, band in stats
    )
    print(f"[CHECK 34] {'PASS' if ok else 'FAIL'} - P5 zero-data reads are neutral")
    return ok


# --- Cross-process determinism of the decision path (CHECK 35) --------------
#
# In-process determinism (CHECK 13, 19) only proves the bot is reproducible
# within a single process. The stricter property is that the chosen action is
# identical across *separate* processes. Two distinct mechanisms can break it,
# and they need different guards:
#
#  1. Builtin ``hash()`` or unordered set/dict iteration -> output depends on
#     PYTHONHASHSEED. Caught behaviourally by diffing an action digest across
#     two PYTHONHASHSEED subprocesses.
#  2. A value keyed on object identity -- ``id(...)`` / a memory address. This
#     is what actually bit the exploitative bot here: a per-opponent dedup keyed
#     on ``id(entry)`` collided non-deterministically as freed history dicts had
#     their addresses reused, desyncing opponent profiles run-to-run. This does
#     NOT track PYTHONHASHSEED (two seeds can share an address layout), so the
#     behavioural diff misses it. It is caught reliably by a source audit that
#     forbids builtin ``id(`` / ``hash(`` in the bot decision path.
#
# CHECK 35 runs both. Mark a deliberate, determinism-safe use with a trailing
# ``# det-ok`` comment to exempt it from the audit.

_TRACE_ARG = "__determinism_trace__"


def _audit_decision_path_for_nondeterminism() -> list[str]:
    """Scan bots/*.py for builtin ``id(`` / ``hash(`` feeding the decision path.

    ``\\bid\\s*\\(`` / ``\\bhash\\s*\\(`` match the builtins but not ``hand_id``,
    ``session_id`` (the ``id`` is not at a word boundary) nor ``_stable_hash`` /
    ``hashlib`` (same). Inline comments are stripped; ``# det-ok`` opts a line
    out. Returns a list of ``file:lineno: source`` violations (empty == clean).
    """
    import glob
    import re

    forbidden = re.compile(r"\b(?:id|hash)\s*\(")
    violations: list[str] = []
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "bots", "*.py"))):
        with open(path, encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                if "det-ok" in raw:
                    continue
                code = raw.split("#", 1)[0]
                if forbidden.search(code):
                    violations.append(
                        f"{os.path.basename(path)}:{lineno}: {raw.strip()}"
                    )
    return violations


def _determinism_trace_digest() -> str:
    """Replay fixed seeded tournaments; return ``<n_actions> <sha256>``.

    Mirrors eval_final_bot's loop (same pool, per-tournament global-RNG seed,
    deck RNG, dealer rotation) so the trace exercises the exact spots the eval
    does. A handful of tournaments are played because process-dependent leaks
    (e.g. a dedup keyed on ``id()``, or hash-randomized iteration) only flip an
    action in specific accumulated states -- a single short tournament can miss
    them.
    """
    import hashlib

    base_seed = 20260619
    pool = ["smart", "mc200", "gto", "icm", "exploitative"]
    specs = [("HERO", "final_survival:p4")]
    specs += [(f"P{i}", b) for i, b in enumerate(pool)]
    n_players = len(specs)
    actions: list[str] = []

    def wrap(pid, bot):
        orig = bot.act

        def traced(view):
            action = orig(view)
            actions.append(
                f"{getattr(view, 'hand_id', None)}|{getattr(view, 'street', None)}"
                f"|{getattr(view, 'pot', None)}|{getattr(view, 'to_call', None)}"
                f"|{pid}|{getattr(action, 'type', None)}|{getattr(action, 'amount', None)}"
            )
            return action

        bot.act = traced
        return bot

    for t in range(2):
        random.seed(base_seed + t + 1 + 7_000_003)
        seats = [Seat(player_id=pid, chips=1000) for pid, _ in specs]
        bots = {pid: wrap(pid, create_bot(spec)) for pid, spec in specs}
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_tournament(
                seats,
                bots,
                small_blind=5,
                big_blind=10,
                blind_increase_every=50,
                max_hands=40,
                dealer_index=t % n_players,
                dealer_rotation="full_table",
                winner_resolution="chip_count_on_max_hands",
                rng=random.Random(base_seed + t + 1),
                suppress_output=True,
                log_decisions=False,
            )
    digest = hashlib.sha256("\n".join(actions).encode()).hexdigest()
    return f"{len(actions)} {digest}"


def _check_cross_process_determinism():
    import subprocess

    # (1) Source audit -- the reliable guard for id()/hash() leaks.
    violations = _audit_decision_path_for_nondeterminism()
    audit_ok = not violations

    # (2) Behavioural diff across two PYTHONHASHSEED subprocesses -- guards
    #     against hash-randomized set/dict iteration order.
    def trace(hashseed: str) -> str:
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), _TRACE_ARG],
            capture_output=True, text=True, env=env, cwd=REPO_ROOT,
        )
        if proc.returncode != 0:
            return f"ERR rc={proc.returncode} {proc.stderr.strip()[-300:]}"
        return proc.stdout.strip()

    a = trace("0")
    b = trace("1")
    n_actions = int(a.split()[0]) if a[:1].isdigit() else 0
    behaviour_ok = a == b and n_actions > 0

    ok = audit_ok and behaviour_ok
    if not audit_ok:
        detail = f"forbidden id()/hash() in bot decision path: {violations}"
    elif not behaviour_ok:
        detail = f"PYTHONHASHSEED=0 -> {a!r}; =1 -> {b!r}"
    else:
        detail = f"audit clean, replay digest stable across hashseeds (actions={n_actions})"
    print(f"[CHECK 35] {'PASS' if ok else 'FAIL'} - cross-process "
          f"determinism of the decision path ({detail})")
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
    PASS &= _check_postflop()
    PASS &= _check_stack_rank_phase4()
    PASS &= _check_desperation_not_rank_phase4()
    PASS &= _check_chipleader_pressure_phase4()
    PASS &= _check_heads_up_phase4()
    PASS &= _check_phase4_invariants()
    PASS &= _check_p5_accumulation_dedup()
    PASS &= _check_p5_once_per_hand_vpip()
    PASS &= _check_p5_pressure_state_machine()
    PASS &= _check_p5_censoring_discipline()
    PASS &= _check_p5_station_suppression_active()
    PASS &= _check_p5_neutral_default_reset()
    PASS &= _check_p5_idempotent_prefix_replay()
    PASS &= _check_p5_off_log_only_parity()
    PASS &= _check_p5_log_only_overfolder()
    PASS &= _check_p5_log_only_passive_tight()
    PASS &= _check_p5_strict_r2_scope()
    PASS &= _check_p5_reset_across_tournaments()
    PASS &= _check_p5_log_only_multi_read_precedence()
    PASS &= _check_p5_fail_closed()
    PASS &= _check_p5_zero_data_neutral()
    PASS &= _check_cross_process_determinism()
    print("=" * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED [PASS]' if PASS else 'SOME CHECKS FAILED [FAIL]'}")
    return PASS


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == _TRACE_ARG:
        # Subprocess entry point for CHECK 35: emit the action digest and exit.
        print(_determinism_trace_digest())
        sys.exit(0)
    sys.exit(0 if run() else 1)
