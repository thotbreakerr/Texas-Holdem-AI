"""
sanity_mccfr_es_update.py — external-sampling MCCFR estimator correctness.

Phase 3 (2026-06-10). Two checks on CFRBot._cfr_recurse:

1. REGRET WEIGHT — opponent reach must be counted ONCE (by sampling),
   not twice. We build a tiny river state where the opponent acts first
   with a frozen 50/50 check-or-bet strategy, and the hero then faces
   fold-vs-call with deterministic (patched) leaf values. The expected
   per-traversal regret increment at the hero's facing-a-bet node is

       p_bet * (action_util - node_util)        <- correct (single p)

   The old buggy update multiplied by cf_reach again, giving p_bet^2 —
   exactly half the correct magnitude here. A Monte-Carlo average over
   N traversals separates the two by ~30 sigma.

2. STRATEGY-SUM PLACEMENT — textbook external sampling accumulates the
   average strategy at OPPONENT nodes (unweighted); hero nodes get none.
   Phase-3.1 exception: the hero TRAVERSAL-ROOT node (depth 0, reach
   exactly 1) also accumulates — covered by sanity_cfr_root_coverage.py.
   Here the root is the OPPONENT (seat_order=[1, 0]), so the hero node
   sits at depth 1 and must still receive zero strategy_sum.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bots.cfr_bot as cfr_bot
from bots.cfr_bot import CFRBot, _GameState

N_TRAVERSALS = 4000
BOARD = [("2", "h"), ("7", "d"), ("9", "s"), ("J", "c"), ("Q", "h")]
HOLES = {0: (("A", "s"), ("A", "d")), 1: (("K", "s"), ("K", "d"))}


def make_root():
    """River, 2 seats. Opponent (seat 1) acts first, hero (seat 0) second.

    Hero stack 10 == any bet's to_call, so hero's only options facing a
    bet are fold/call (raise needs chips > to_call). Pot 10, BB 10.
    Opponent options: check (1), bet 10 (2), all-in 100 (7).
    """
    return _GameState(
        pot=10, stacks=[10, 100], committed_per_seat=[0, 0],
        alive=[True, True], street="river", board=list(BOARD),
        hole_cards=dict(HOLES), seat_order=[1, 0], action_idx=0,
        history_tokens="", deck_remaining=[], hero_seat=0,
        real_contributions=True, ring_order=[0, 1],
        position_labels={0: "BTN", 1: "BB"}, big_blind=10,
    )


def fake_leaf(state, hero_seat):
    """Deterministic leaf values (replaces the MC equity evaluator)."""
    if not state.alive[hero_seat]:
        return 0.0                      # hero folded
    c_hero = state.committed_per_seat[hero_seat]
    c_opp = state.committed_per_seat[1 - hero_seat]
    if c_hero > 0 and c_hero == c_opp:
        return 30.0                     # hero called
    return 10.0                         # check lines / everything else
    # NOTE: values are arbitrary; only their differences matter.


def run() -> bool:
    ok = True
    random.seed(1234)

    # Deterministic card bucket so info-set keys are stable across calls
    # (the real _postflop_bucket Monte-Carlos equity and can flip bins).
    original_bucket = cfr_bot._postflop_bucket
    cfr_bot._postflop_bucket = lambda hole, board, n_opponents: 7

    try:
        bot = CFRBot(iterations=1, profile_path=None, use_average=True)
        bot._rng = random.Random(99)
        bot._leaf_value = fake_leaf     # instance-level override

        root = make_root()

        # Identify the two nodes by computing their keys directly.
        opp_key = bot._info_key_for_state(root, 1)
        facing_bet_state = root.apply_action(1, 2)   # opp bets 10
        hero_key = bot._info_key_for_state(facing_bet_state, 0)

        opp_legal = root.legal_abstract_actions()            # [1, 2, 7]
        hero_legal = facing_bet_state.legal_abstract_actions()  # [0, 1]
        print(f"opp legal: {opp_legal}   hero legal: {hero_legal}")
        if opp_legal != [1, 2, 7] or hero_legal != [0, 1]:
            print("[FAIL] unexpected legal masks — test setup invalid")
            return False

        # Frozen strategies via preset regrets:
        #   opponent: 50% check, 50% bet-10, 0% all-in
        #   hero:     75% fold, 25% call
        OPP_PRESET = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        HERO_PRESET = [3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Expected hero values at the facing-a-bet node:
        #   fold: leaf 0  - cost 0  = 0
        #   call: leaf 30 - cost 10 = 20
        #   node_util = 0.75*0 + 0.25*20 = 5
        #   increments: fold -5, call +15 ... weighted by p_bet = 0.5
        P_BET = 0.5
        EXPECTED_FOLD = P_BET * (0.0 - 5.0)    # -2.5
        EXPECTED_CALL = P_BET * (20.0 - 5.0)   # +7.5
        BUGGY_CALL = P_BET ** 2 * 15.0         # +3.75 (cf_reach double count)

        sum_fold = 0.0
        sum_call = 0.0
        opp_node = bot._get_node(opp_key)
        hero_node = bot._get_node(hero_key)

        for _ in range(N_TRAVERSALS):
            # Re-freeze strategies every traversal.
            opp_node.regret_sum = list(OPP_PRESET)
            hero_node.regret_sum = list(HERO_PRESET)
            before = list(hero_node.regret_sum)
            bot._cfr_recurse(make_root(), 0, depth=0)
            sum_fold += hero_node.regret_sum[0] - before[0]
            sum_call += hero_node.regret_sum[1] - before[1]

        mean_fold = sum_fold / N_TRAVERSALS
        mean_call = sum_call / N_TRAVERSALS
        # SE(mean_call) = 7.5/sqrt(N) ~ 0.12 at N=4000; use 6-sigma bands.
        TOL = 0.75
        print(f"mean regret increment  fold: {mean_fold:+.3f} "
              f"(expect {EXPECTED_FOLD:+.2f})")
        print(f"mean regret increment  call: {mean_call:+.3f} "
              f"(expect {EXPECTED_CALL:+.2f}; buggy would be "
              f"{BUGGY_CALL:+.2f})")

        if abs(mean_call - EXPECTED_CALL) <= TOL:
            print("[PASS] regret update weighted by single power of "
                  "opponent reach")
        else:
            print("[FAIL] regret update weight wrong (double-counted "
                  "opponent reach?)")
            ok = False
        if abs(mean_fold - EXPECTED_FOLD) <= TOL:
            print("[PASS] fold regret increment matches expectation")
        else:
            print("[FAIL] fold regret increment off")
            ok = False

        # ── Check 2: strategy-sum placement ──────────────────────────
        opp_node.regret_sum = list(OPP_PRESET)
        hero_node.regret_sum = list(HERO_PRESET)
        opp_node.strategy_sum = [0.0] * len(opp_node.strategy_sum)
        hero_node.strategy_sum = [0.0] * len(hero_node.strategy_sum)

        bot._cfr_recurse(make_root(), 0, depth=0)

        opp_ss = opp_node.strategy_sum
        expect_opp = [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        if all(abs(opp_ss[i] - expect_opp[i]) < 1e-9 for i in range(8)):
            print("[PASS] opponent node strategy_sum += current strategy "
                  "(unweighted)")
        else:
            print(f"[FAIL] opponent strategy_sum {opp_ss} != {expect_opp}")
            ok = False

        if all(v == 0.0 for v in hero_node.strategy_sum):
            print("[PASS] hero (traverser) node accumulates no "
                  "strategy_sum")
        else:
            print(f"[FAIL] hero strategy_sum should stay zero, got "
                  f"{hero_node.strategy_sum}")
            ok = False
    finally:
        cfr_bot._postflop_bucket = original_bucket

    print("\nOVERALL:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
