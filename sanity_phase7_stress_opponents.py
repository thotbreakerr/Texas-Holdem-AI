#!/usr/bin/env python3
"""Deterministic checks for Phase 7 stress-opponent archetypes."""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from bots import create_bot
from bots.archetype_bot import ARCHETYPE_CONFIGS, ArchetypeBot
from bots.tournament_hybrid_bot import OpponentProfiles
from core.bot_api import Action, PlayerView
from eval_final_bot import _build_eval_specs, _hero_relative_offsets


def _history_entry(street, pid, action_type, *, amount=None, to_call_before=0, pot_before=0):
    return {
        "street": street,
        "pid": pid,
        "type": action_type,
        "amount": amount,
        "to_call_before": to_call_before,
        "pot_before": pot_before,
    }


def _view(history, *, hand_id=70000, opponents=("V",), all_in=(), street="flop", min_raise=10):
    return PlayerView(
        me="Hero",
        street=street,
        position="BTN",
        hole_cards=[("A", "s"), ("K", "s")],
        board=[("2", "c"), ("7", "d"), ("J", "h")] if street != "preflop" else [],
        pot=180,
        to_call=0,
        min_raise=min_raise,
        max_raise=1000,
        legal_actions=[{"type": "check"}],
        stacks={"Hero": 1000, **{pid: 1000 for pid in opponents}},
        opponents=list(opponents),
        history=list(history),
        hand_id=hand_id,
        seat_indices={"Hero": 0, **{pid: i + 1 for i, pid in enumerate(opponents)}},
        acting_opponents=list(opponents),
        all_in_opponents=list(all_in),
    )


def _profile_for(history, *, all_in=(), hand_id=70000):
    profiles = OpponentProfiles()
    profiles.ingest(_view(history, all_in=all_in, hand_id=hand_id))
    return profiles.raw("V"), profiles.stat_summary("V", confidence_w=10.0)


def _legal_types(legal_actions):
    return tuple(
        spec.get("type")
        for spec in legal_actions
        if isinstance(spec, dict) and isinstance(spec.get("type"), str)
    )


def _valid_action(action, view):
    legal = _legal_types(view.legal_actions)
    if action.type not in legal:
        return False
    if action.type in ("bet", "raise"):
        for spec in view.legal_actions:
            if spec.get("type") == action.type:
                return int(spec["min"]) <= int(action.amount) <= int(spec["max"])
        return False
    return action.amount is None


def _sample_view(alias, *, hand_id=71000):
    street = "flop"
    legal = [{"type": "check"}, {"type": "bet", "min": 10, "max": 300}]
    to_call = 0
    history = [
        _history_entry("preflop", "P0", "raise", amount=30, to_call_before=10, pot_before=15),
        _history_entry("preflop", "Hero", "call", amount=30, to_call_before=30, pot_before=45),
    ]
    if alias in ("calling_station", "loose_passive", "nit", "folder"):
        legal = [{"type": "fold"}, {"type": "call"}, {"type": "raise", "min": 80, "max": 300}]
        to_call = 40
        history.append(_history_entry("flop", "Hero", "bet", amount=40, pot_before=80))
    return PlayerView(
        me="P0",
        street=street,
        position="UTG",
        hole_cards=[("A", "h"), ("Q", "h")],
        board=[("2", "c"), ("7", "d"), ("J", "h")],
        pot=120,
        to_call=to_call,
        min_raise=10,
        max_raise=300,
        legal_actions=legal,
        stacks={"Hero": 1000, "P0": 300, "P1": 900},
        opponents=["Hero", "P1"],
        history=history,
        hand_id=hand_id,
        seat_indices={"Hero": 0, "P0": 1, "P1": 2},
        acting_opponents=["Hero", "P1"],
        all_in_opponents=[],
    )


def _check_registry_and_legal_actions():
    ok = True
    aliases = (
        "maniac",
        "maniac_trigger",
        "maniac_mixed",
        "overbet_merchant",
        "calling_station",
        "nit",
        "folder",
        "loose_passive",
        "minraise",
        "minraiser",
        "baseline_sane",
        "pressure_filler",
    )
    for idx, alias in enumerate(aliases):
        adapter = create_bot(alias)
        core = getattr(adapter, "bot", None)
        ok &= isinstance(core, ArchetypeBot)
        view = _sample_view(alias, hand_id=71000 + idx)
        a1 = adapter.act(view)
        a2 = create_bot(alias).act(view)
        ok &= _valid_action(a1, view)
        ok &= (a1.type, a1.amount) == (a2.type, a2.amount)
    try:
        create_bot("__phase7_unknown__")
    except ValueError as exc:
        ok &= "calling_station" in str(exc) and "pressure_filler" in str(exc)
    else:
        ok = False
    print(f"[P7 CHECK 1] {'PASS' if ok else 'FAIL'} - registry aliases are legal and deterministic")
    return ok


def _check_classifier_targets():
    ok = True

    raw, _ = _profile_for([
        _history_entry("preflop", "V", "raise", amount=30, to_call_before=10, pot_before=15),
        _history_entry("flop", "V", "bet", amount=160, pot_before=100),
    ], all_in=("V",), hand_id=72001)
    ok &= raw["jam_like_count"] == 1 and raw["short_jam_like_count"] == 0

    raw, _ = _profile_for([
        _history_entry("preflop", "V", "raise", amount=30, to_call_before=10, pot_before=15),
        _history_entry("flop", "V", "bet", amount=95, pot_before=100),
    ], hand_id=72002)
    ok &= raw["large_bet_count"] == 1 and raw["jam_like_count"] == 0

    raw, stats = _profile_for([
        _history_entry("flop", "Hero", "bet", amount=40, pot_before=80),
        _history_entry("flop", "V", "call", amount=40, to_call_before=40, pot_before=120),
    ], hand_id=72003)
    ok &= raw["postflop_pressure_call"] == 1 and stats["station_response_n"] == 1

    raw, _ = _profile_for([
        _history_entry("flop", "Hero", "bet", amount=40, pot_before=80),
        _history_entry("flop", "V", "fold", to_call_before=40, pot_before=120),
    ], hand_id=72004)
    ok &= raw["postflop_pressure_fold"] == 1 and raw["pressure_fold"] == 1

    raw, stats = _profile_for([
        _history_entry("preflop", "V", "call", amount=10, to_call_before=10, pot_before=15),
        _history_entry("flop", "Hero", "bet", amount=30, pot_before=60),
        _history_entry("flop", "V", "call", amount=30, to_call_before=30, pot_before=90),
    ], hand_id=72005)
    ok &= raw["vpip_seen"] == 1
    ok &= raw["preflop_raise_seen"] == 0
    ok &= raw["jam_like_count"] == 0 and raw["large_bet_count"] == 0
    ok &= stats["preflop_aggression_rate_hat"] < 0.18

    raw, _ = _profile_for([
        _history_entry("preflop", "V", "raise", amount=20, to_call_before=10, pot_before=15),
        _history_entry("flop", "V", "raise", amount=45, to_call_before=20, pot_before=120),
    ], hand_id=72006)
    ok &= raw["preflop_raise_seen"] == 1
    ok &= raw["jam_like_count"] == 0 and raw["large_bet_count"] == 0

    raw, _ = _profile_for([
        _history_entry("preflop", "V", "call", amount=10, to_call_before=10, pot_before=15),
        _history_entry("flop", "V", "bet", amount=30, pot_before=120),
    ], hand_id=72007)
    ok &= raw["postflop_bet_raise"] == 1
    ok &= raw["jam_like_count"] == 0 and raw["large_bet_count"] == 0

    profiles = OpponentProfiles()
    profiles.ingest(_view([
        _history_entry("preflop", "V", "raise", amount=30, to_call_before=10, pot_before=15),
        _history_entry("flop", "V", "raise", amount=1461, to_call_before=50, pot_before=140),
    ], all_in=("V",), hand_id=72008, min_raise=2822))
    raw = profiles.raw("V")
    ok &= raw["jam_like_count"] == 1 and raw["short_jam_like_count"] == 0

    print(f"[P7 CHECK 2] {'PASS' if ok else 'FAIL'} - classifier targets fire for all policy roles")
    return ok


def _check_policy_roster_shape():
    expected = {
        "maniac",
        "maniac_trigger",
        "maniac_mixed",
        "overbet_merchant",
        "calling_station",
        "nit",
        "folder",
        "loose_passive",
        "minraise",
        "minraiser",
        "baseline_sane",
        "pressure_filler",
    }
    policy_names = {ARCHETYPE_CONFIGS[name].policy_name for name in expected}
    ok = expected.issubset(ARCHETYPE_CONFIGS)
    ok &= policy_names == {
        "maniac",
        "overbet",
        "station",
        "nit",
        "loose_passive",
        "minraise",
        "baseline_sane",
        "pressure_filler",
    }
    ok &= "station" not in ARCHETYPE_CONFIGS
    print(f"[P7 CHECK 3] {'PASS' if ok else 'FAIL'} - one wrapper with separate policy classes")
    return ok


def _bot_base_type(bot_type):
    return str(bot_type or "").strip().lower().split(":", 1)[0]


def _check_seat_geometry_harness():
    candidate = "final_aggro:p5"
    pool = ["maniac_trigger", "maniac_trigger", "pressure_filler", "calling_station", "nit"]

    before_after = _build_eval_specs(candidate, pool, "target-before-after")
    target_after = _build_eval_specs(candidate, pool, "target-after-hero")
    bracketing = _build_eval_specs(candidate, pool, "bracketing")

    is_maniac = lambda bot_type: _bot_base_type(bot_type).startswith("maniac")
    ok = True
    ok &= [pid for pid, _ in before_after] == ["P0", "HERO", "P1", "P2", "P3", "P4"]
    ok &= _hero_relative_offsets(before_after, is_maniac) == [1, 5]
    ok &= [pid for pid, _ in target_after] == ["P2", "P0", "HERO", "P1", "P3", "P4"]
    ok &= _hero_relative_offsets(target_after, is_maniac) == [1, 5]
    ok &= [pid for pid, _ in bracketing] == ["P0", "P2", "P3", "P1", "HERO", "P4"]
    ok &= _hero_relative_offsets(bracketing, is_maniac) == [2, 5]

    pressure_pool = ["maniac_trigger", "calling_station", "pressure_filler", "loose_passive", "nit"]
    pressure = _build_eval_specs(candidate, pressure_pool, "pressure-nit-hero")
    ok &= [pid for pid, _ in pressure] == ["P0", "P1", "P3", "P2", "P4", "HERO"]
    ok &= _hero_relative_offsets(pressure, lambda bot_type: _bot_base_type(bot_type) == "pressure_filler") == [4]
    ok &= _hero_relative_offsets(pressure, lambda bot_type: _bot_base_type(bot_type) == "nit") == [5]

    try:
        _build_eval_specs(candidate, ["pressure_filler", "calling_station"], "target-before-after")
    except ValueError:
        missing_roles_rejected = True
    else:
        missing_roles_rejected = False
    ok &= missing_roles_rejected

    print(f"[P7 CHECK 4] {'PASS' if ok else 'FAIL'} - deterministic seat geometry offsets")
    return ok


def run():
    ok = True
    ok &= _check_registry_and_legal_actions()
    ok &= _check_classifier_targets()
    ok &= _check_policy_roster_shape()
    ok &= _check_seat_geometry_harness()
    print("=" * 60)
    print(f"PHASE 7: {'ALL CHECKS PASSED [PASS]' if ok else 'SOME CHECKS FAILED [FAIL]'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
