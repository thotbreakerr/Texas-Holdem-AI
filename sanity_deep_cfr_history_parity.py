"""
sanity_deep_cfr_history_parity.py — Deep CFR tree vs engine history parity (I3/B1)

The history GRU is trained on the tree's ``history_events`` and queried at play
time on ``extract_history(engine PlayerView)``.  Those two encoders must agree
or the network is trained on input prefixes that never occur in real play.

This gate plays out EQUIVALENT hands in both worlds:

  * the real engine (``Table.play_hand``) driven by scripted bots that convert
    the same abstract action indices to concrete Actions via
    ``_abstract_to_concrete`` (the exact mapping inference uses), and
  * the Deep CFR tree (``_DeepCFRGameState.apply_action``) driven by the same
    abstract action indices,

then asserts the two produce identical (street, seat, action, amount,
pot_before) sequences.  Scenarios cover the three spots the review flagged:

  1. a shove (abstract ``all_in``) — pre-fix the tree labeled it "all_in",
     an action type the engine can never produce, so this gate FAILS pre-fix;
  2. a postflop raise whose pot-fraction target clamps up to the min-raise;
  3. a short-stack call-for-less (call amount = chips actually paid).

A heads-up scenario is included so the n=2 blind/order conventions are
exercised as well as the multiway ones.
"""
from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout

# Repo root = this file's directory (gates live at the repo root), so the
# gate runs in any clone — never hard-code an absolute machine path here.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random

from core.engine import Table, Seat
from core.action_history import extract_history
from core.bot_api import Action, BotAdapter, PlayerView
from core.table_order import street_action_order
from bots.deep_cfr_bot import (
    ABSTRACT_ACTIONS,
    _abstract_to_concrete,
    _DeepCFRGameState,
    _FULL_DECK,
)

BIG_BLIND = 10
SMALL_BLIND = 5


# ─────────────────────────────────────────────────────────────────────────────
#  Engine side: scripted bots that play a fixed list of abstract actions
# ─────────────────────────────────────────────────────────────────────────────

class ScriptedAbstractBot(BotAdapter):
    """Pops abstract action indices off a queue and converts each one to a
    concrete Action with the same mapping Deep CFR inference uses."""

    def __init__(self, script: list[int], observed_views: list[PlayerView]):
        self.script = list(script)
        self.observed_views = observed_views  # shared list — collects every view

    def act(self, view: PlayerView) -> Action:
        # Record the view BEFORE acting; its .history holds everything so far.
        self.observed_views.append(view)
        if not self.script:
            raise AssertionError(
                f"script for {view.me} exhausted but engine asked for an action"
            )
        abstract_idx = self.script.pop(0)
        return _abstract_to_concrete(
            abstract_idx,
            view.legal_actions,
            view.pot,
            street=view.street,
            big_blind=BIG_BLIND,
        )


def play_engine_hand(stacks: list[int], scripts: dict[int, list[int]]):
    """Run one engine hand with seat ``i`` holding ``stacks[i]`` chips and
    playing ``scripts[i]`` (a list of abstract action indices, in turn order).

    Returns the longest PlayerView observed during the hand — its ``history``
    contains every action up to (excluding) the final scripted decision.
    """
    n = len(stacks)
    seats = [Seat(player_id=f"P{i}", chips=stacks[i]) for i in range(n)]
    observed: list[PlayerView] = []
    bot_for = {
        f"P{i}": ScriptedAbstractBot(scripts.get(i, []), observed)
        for i in range(n)
    }
    table = Table(rng=random.Random(20260611))
    # dealer_index=0 → ring order [0, 1, ..., n-1] with seat 0 on the button,
    # matching the tree's ring_order convention.
    with redirect_stdout(io.StringIO()):
        table.play_hand(seats, SMALL_BLIND, BIG_BLIND, dealer_index=0,
                        bot_for=bot_for)
    if not observed:
        raise AssertionError("engine hand produced no PlayerViews")
    # The view with the longest history has seen the most actions.
    return max(observed, key=lambda v: len(v.history or []))


# ─────────────────────────────────────────────────────────────────────────────
#  Tree side: drive _DeepCFRGameState with the same abstract actions
# ─────────────────────────────────────────────────────────────────────────────

def build_tree_state(stacks: list[int]) -> _DeepCFRGameState:
    """Fresh preflop tree state mirroring the engine hand above.

    Blind seats follow the engine: heads-up the button (seat 0) posts the SB
    and seat 1 the BB; 3+ handed seat 1 posts SB and seat 2 BB.  Preflop
    action order comes from the shared core.table_order helper.
    """
    n = len(stacks)
    deck = list(_FULL_DECK)
    rng = random.Random(20260611)
    rng.shuffle(deck)
    hole_cards = {seat: (deck.pop(), deck.pop()) for seat in range(n)}

    stacks = list(stacks)
    committed = [0] * n
    if n == 2:
        sb_seat, bb_seat = 0, 1  # heads-up: the button is the small blind
    else:
        sb_seat, bb_seat = 1, 2
    sb_amt = min(SMALL_BLIND, stacks[sb_seat])
    bb_amt = min(BIG_BLIND, stacks[bb_seat])
    stacks[sb_seat] -= sb_amt
    committed[sb_seat] = sb_amt
    stacks[bb_seat] -= bb_amt
    committed[bb_seat] = bb_amt

    ring = list(range(n))
    seat_order = [s for s in street_action_order("preflop", ring)
                  if stacks[s] > 0]

    return _DeepCFRGameState(
        pot=sb_amt + bb_amt,
        stacks=stacks,
        committed_per_seat=committed,
        alive=[True] * n,
        street="preflop",
        board=[],
        hole_cards=hole_cards,
        seat_order=seat_order,
        action_idx=0,
        history_events=[],
        deck_remaining=deck,
        big_blind=BIG_BLIND,
        ring_order=ring,
    )


def drive_tree(state: _DeepCFRGameState, scripts: dict[int, list[int]]):
    """Apply per-seat abstract scripts to an existing tree state.

    Returns the final state's history_events.  Stops once every script is
    exhausted (the matching engine hand ends at the same point by
    construction).  Also reused by sanity_deep_cfr_curriculum.py to drive
    states built by the PRODUCTION builder (train_deep_cfr.build_initial_state).
    """
    scripts = {seat: list(script) for seat, script in scripts.items()}
    random.seed(20260611)  # advance_street deals random cards; irrelevant here

    # Safety bound so a parity bug cannot loop forever.
    for _ in range(200):
        if all(len(s) == 0 for s in scripts.values()):
            break
        if state.is_terminal():
            break
        if state.is_chance_node():
            state = state.advance_street()
            continue
        seat = state.seat_to_act()
        if not scripts.get(seat):
            raise AssertionError(
                f"tree asked seat {seat} to act but its script is empty — "
                f"action order diverged from the engine"
            )
        abstract_idx = scripts[seat].pop(0)
        state = state.apply_action(seat, abstract_idx)
    return state.history_events


def play_tree_hand(stacks: list[int], scripts: dict[int, list[int]]):
    """Build a fresh tree state for ``stacks`` and apply the scripts."""
    return drive_tree(build_tree_state(stacks), scripts)


# ─────────────────────────────────────────────────────────────────────────────
#  Comparison helper
# ─────────────────────────────────────────────────────────────────────────────

def as_tuples(events) -> list[tuple]:
    return [(e.street, e.seat, e.action, int(e.amount), int(e.pot_before))
            for e in events]


def compare_scenario(name: str, stacks: list[int],
                     scripts: dict[int, list[int]],
                     must_contain: list[tuple]) -> bool:
    """Play the scenario in both worlds and diff the histories.

    ``must_contain`` lists (street, action, amount) tuples that must appear in
    the ENGINE history — a self-check that the scenario actually produced the
    spot it claims to cover (shove / clamp / call-for-less).
    """
    ok = True
    print(f"  Scenario: {name}")

    engine_view = play_engine_hand(stacks, {k: list(v) for k, v in scripts.items()})
    engine_events = as_tuples(extract_history(engine_view))
    tree_events = as_tuples(play_tree_hand(stacks, scripts))

    # The final observed view excludes the hand's very last action, so compare
    # the tree prefix of the same length.  Scenarios are built so every
    # interesting action lands inside this prefix.
    n_compare = len(engine_events)
    tree_prefix = tree_events[:n_compare]

    if n_compare == 0:
        print("    [FAIL] — engine produced an empty history")
        return False

    if tree_prefix == engine_events:
        print(f"    [PASS] — {n_compare} events identical "
              f"(street, seat, action, amount, pot_before)")
    else:
        ok = False
        print(f"    [FAIL] — histories diverge")
        for i in range(max(len(tree_prefix), n_compare)):
            t = tree_prefix[i] if i < len(tree_prefix) else "<missing>"
            e = engine_events[i] if i < n_compare else "<missing>"
            marker = "  " if t == e else "✗ "
            print(f"      {marker}tree={t}  engine={e}")

    # Self-check: the scenario really contains the spots it advertises.
    flat = [(street, action, amount)
            for (street, _seat, action, amount, _pot) in engine_events]
    for needed in must_contain:
        if needed in flat:
            print(f"    [PASS] — scenario exercises {needed}")
        else:
            ok = False
            print(f"    [FAIL] — scenario never produced {needed}; "
                  f"engine history={flat}")

    # The engine can never emit an "all_in" action type; the tree must not
    # either (this is the line that fails pre-fix).
    bad = [t for t in tree_prefix if t[2] == "all_in"]
    if bad:
        ok = False
        print(f"    [FAIL] — tree emitted engine-impossible 'all_in' events: {bad}")
    else:
        print("    [PASS] — tree emitted no 'all_in' action labels")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
#  Scenarios
# ─────────────────────────────────────────────────────────────────────────────

IDX = {label: i for i, label in enumerate(ABSTRACT_ACTIONS)}


def run() -> bool:
    PASS = True

    print("=" * 60)
    print("Check 1: 3-handed — shove + call-for-less (preflop)")
    print("=" * 60)
    # Seats: 0=BTN (1000), 1=SB (150), 2=BB (60).  BTN opens to 2.5BB, the SB
    # shoves 150 (abstract all_in → engine 'raise'), the BB calls all-in for
    # less (pays its remaining 50 against a 140 to_call), action returns to
    # the BTN — whose view contains all three events — and the BTN folds.
    PASS &= compare_scenario(
        name="BTN opens, SB shoves, BB calls for less, BTN folds",
        stacks=[1000, 150, 60],
        scripts={
            0: [IDX["bet_33"], IDX["fold"]],  # open to 2.5BB=25, then fold
            1: [IDX["all_in"]],               # shove to 150
            2: [IDX["check_call"]],           # call all-in for less (pays 50)
        },
        must_contain=[
            ("preflop", "raise", 25),    # the open
            ("preflop", "raise", 150),   # the shove — engine label is 'raise'
            ("preflop", "call", 50),     # call-for-less records chips PAID
        ],
    )
    print()

    print("=" * 60)
    print("Check 2: heads-up — limp, flop bet, clamped raise")
    print("=" * 60)
    # Heads-up 1000/1000.  BTN/SB limps, BB checks.  Flop (BB first): BB bets
    # pot (20); BTN raises bet_33 — its 0.33*pot=13 target clamps UP to the
    # min-raise total of 40 on both sides; BB's next view holds everything,
    # then BB folds.
    PASS &= compare_scenario(
        name="HU limp pot, flop pot-bet, bet_33 raise clamps to min-raise",
        stacks=[1000, 1000],
        scripts={
            0: [IDX["check_call"], IDX["bet_33"]],   # limp; clamped flop raise
            1: [IDX["check_call"], IDX["bet_100"], IDX["fold"]],
        },
        must_contain=[
            ("preflop", "call", 5),     # limp completes the SB
            ("flop", "bet", 20),        # pot-sized lead
            ("flop", "raise", 40),      # 13 clamped up to min-raise 40
        ],
    )
    print()

    print("=" * 60)
    print("Check 3: heads-up — shove over a flop bet (postflop all-in)")
    print("=" * 60)
    # Same shape but the response to the flop bet is an abstract all_in, so a
    # postflop shove is covered too (engine label 'raise', amount = stack).
    PASS &= compare_scenario(
        name="HU flop bet, BTN shoves, BB folds",
        stacks=[500, 500],
        scripts={
            0: [IDX["check_call"], IDX["all_in"]],
            1: [IDX["check_call"], IDX["bet_100"], IDX["fold"]],
        },
        must_contain=[
            ("flop", "bet", 20),
            ("flop", "raise", 490),     # whole stack: 500 - 10 preflop
        ],
    )
    print()

    print("=" * 60)
    if PASS:
        print("ALL CHECKS PASSED [PASS]")
    else:
        print("SOME CHECKS FAILED [FAIL]")
    print("=" * 60)
    return PASS


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
