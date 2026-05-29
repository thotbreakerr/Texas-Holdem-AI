#!/usr/bin/env python3
"""
probe_deep_cfr.py — action-distribution probe for a Deep CFR checkpoint.

This is diagnostic only. It does not train and does not modify checkpoints.
"""
from __future__ import annotations

import argparse
import io
import random
from collections import Counter
from contextlib import redirect_stdout

from core.bot_api import PlayerView
from core.engine import _FULL_DECK
from bots.deep_cfr_bot import DeepCFRBot


POSITIONS = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
RANKS = "23456789TJQKA"


def _load_bot(weights_path: str, disable_search: bool) -> DeepCFRBot:
    with redirect_stdout(io.StringIO()):
        bot = DeepCFRBot(weights_path=weights_path, inference_mode=True)
    if disable_search:
        bot.search_depth = 0
    return bot


def _deck_without(cards):
    used = set(cards)
    deck = [c for c in _FULL_DECK if c not in used]
    random.shuffle(deck)
    return deck


def _random_hole():
    deck = list(_FULL_DECK)
    random.shuffle(deck)
    return [deck.pop(), deck.pop()]


def _random_board(exclude, n=3):
    deck = _deck_without(exclude)
    return [deck.pop() for _ in range(n)]


def _view(hole, *, board=None, street="preflop", position="BTN",
          pot=15, to_call=10, stack=500, n_opponents=5):
    board = list(board or [])
    opponents = [f"opp{i}" for i in range(1, n_opponents + 1)]
    stacks = {"hero": stack}
    for op in opponents:
        stacks[op] = stack

    if to_call > 0:
        legal = [
            {"type": "fold"},
            {"type": "call"},
            {"type": "raise", "min": max(to_call + 10, 20), "max": stack},
        ]
    else:
        legal = [
            {"type": "check"},
            {"type": "bet", "min": 10, "max": stack},
        ]

    return PlayerView(
        me="hero",
        street=street,
        position=position,
        hole_cards=list(hole),
        board=board,
        pot=pot,
        to_call=to_call,
        min_raise=legal[-1].get("min", 0),
        max_raise=legal[-1].get("max", 0),
        legal_actions=legal,
        stacks=stacks,
        opponents=opponents,
        history=[],
    )


def _is_vpip(action) -> bool:
    return action.type in ("call", "bet", "raise", "all_in")


def _is_raise(action) -> bool:
    return action.type in ("bet", "raise", "all_in")


def _is_all_in(action, view) -> bool:
    if action.type == "all_in":
        return True
    if action.type not in ("bet", "raise"):
        return False
    return action.amount is not None and action.amount >= view.max_raise


def _record(stats, action, view):
    stats["total"] += 1
    stats["actions"][action.type] += 1
    if _is_vpip(action):
        stats["vpip"] += 1
    if _is_raise(action):
        stats["pfr"] += 1
    if _is_all_in(action, view):
        stats["all_in"] += 1
    if action.type == "fold":
        stats["fold"] += 1
    if action.type in ("bet", "raise") and action.amount is not None:
        stats["sizes"].append(action.amount / max(view.pot, 1))


def _new_stats():
    return {
        "total": 0,
        "actions": Counter(),
        "vpip": 0,
        "pfr": 0,
        "all_in": 0,
        "fold": 0,
        "sizes": [],
    }


def _pct(num, den):
    return num / den * 100 if den else 0.0


def _print_stats(title, stats, *, healthy=None):
    n = stats["total"]
    avg_size = sum(stats["sizes"]) / len(stats["sizes"]) if stats["sizes"] else 0.0
    print(title)
    print("-" * len(title))
    print(f"Samples:          {n}")
    print(f"Actions:          {dict(stats['actions'])}")
    print(f"VPIP:             {_pct(stats['vpip'], n):5.1f}%")
    print(f"PFR/bet freq:     {_pct(stats['pfr'], n):5.1f}%")
    print(f"All-in freq:      {_pct(stats['all_in'], n):5.1f}%")
    print(f"Fold freq:        {_pct(stats['fold'], n):5.1f}%")
    print(f"Avg bet/raise:    {avg_size:5.2f}x pot")
    if healthy:
        for label, ok, detail in healthy(stats):
            mark = "OK" if ok else "FLAG"
            print(f"{mark}: {label} — {detail}")
    print()


def _preflop_health(stats):
    n = stats["total"]
    vpip = _pct(stats["vpip"], n)
    pfr = _pct(stats["pfr"], n)
    all_in = _pct(stats["all_in"], n)
    fold = _pct(stats["fold"], n)
    return [
        ("VPIP", 20 <= vpip <= 40, "healthy rough target 20-40%"),
        ("PFR", 15 <= pfr <= 25, "healthy rough target 15-25%"),
        ("all-in", all_in < 5, "healthy rough target <5%"),
        ("fold", 60 <= fold <= 80, "healthy rough target 60-80%"),
    ]


def run_probe(args):
    random.seed(args.seed)
    bot = _load_bot(args.weights, args.disable_search)

    pre = _new_stats()
    for _ in range(args.samples):
        hole = _random_hole()
        view = _view(
            hole,
            position=random.choice(POSITIONS),
            pot=15,
            to_call=10,
            stack=args.chips,
        )
        action = bot.act(view)
        _record(pre, action, view)

    post = _new_stats()
    for _ in range(args.samples):
        hole = _random_hole()
        board = _random_board(hole, 3)
        pot = random.choice([30, 50, 80, 120, 200])
        to_call = random.choice([0, 10, 20, 40, 80])
        to_call = min(to_call, pot)
        view = _view(
            hole,
            board=board,
            street="flop",
            position=random.choice(POSITIONS),
            pot=pot,
            to_call=to_call,
            stack=args.chips,
        )
        action = bot.act(view)
        _record(post, action, view)

    strong = _new_stats()
    strong_hands = [
        [("A", "h"), ("A", "s")],
        [("K", "h"), ("K", "s")],
        [("A", "h"), ("K", "h")],
    ]
    for i in range(50):
        view = _view(
            strong_hands[i % len(strong_hands)],
            position=random.choice(["CO", "BTN"]),
            pot=15,
            to_call=10,
            stack=args.chips,
        )
        action = bot.act(view)
        _record(strong, action, view)

    trash = _new_stats()
    trash_hands = [
        [("7", "h"), ("2", "c")],
        [("3", "h"), ("2", "c")],
    ]
    for i in range(50):
        view = _view(
            trash_hands[i % len(trash_hands)],
            position=random.choice(["UTG", "MP"]),
            pot=15,
            to_call=10,
            stack=args.chips,
        )
        action = bot.act(view)
        _record(trash, action, view)

    print("=" * 72)
    print("Deep CFR Action-Distribution Probe")
    print("=" * 72)
    print(f"Weights:        {args.weights}")
    print(f"Search:         {'disabled' if args.disable_search else 'enabled'}")
    print(f"Samples:        {args.samples} random preflop, {args.samples} random flop")
    print(f"Seed:           {args.seed}")
    print("=" * 72)
    print()

    _print_stats("A. Preflop random spots", pre, healthy=_preflop_health)
    _print_stats("B. Postflop random flop spots", post)

    strong_play = _pct(strong["vpip"], strong["total"])
    _print_stats("C. Strong-hand sanity (AA/KK/AKs late position)", strong)
    print(f"Strong-hand continue/raise/call rate: {strong_play:5.1f}%")
    print(("OK" if strong_play >= 80 else "FLAG") +
          ": target is >80% continuing strong hands")
    print()

    trash_fold = _pct(trash["fold"], trash["total"])
    _print_stats("D. Trash-hand sanity (72o/32o early position)", trash)
    print(f"Trash-hand fold rate: {trash_fold:5.1f}%")
    print(("OK" if trash_fold >= 70 else "FLAG") +
          ": target is >70% folding trash early")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe a Deep CFR checkpoint's action distribution."
    )
    parser.add_argument("--weights", default="models/deep_cfr_v1.pt")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--chips", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-search", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_probe(parse_args())
