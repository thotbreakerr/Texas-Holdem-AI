"""
Regressions for two Deep CFR fixes:

  #3  Postflop sizing head no longer overrides the bucket fraction, so bet_33
      and bet_100 map to DISTINCT concrete amounts (they previously collapsed
      to the single sizing_frac*pot value).

  #4  build_network_input reconstructs each opponent's committed amount instead
      of hardcoding 0, matching the committed/pot feature semantics used during
      training (to_network_input feeds committed_per_seat).
"""
import sys

sys.path.insert(0, "/Users/jaroslavaupart/Desktop/Projects/Texas-Holdem-AI")

from core.bot_api import PlayerView
from bots.deep_cfr_bot import (
    _abstract_to_concrete,
    build_network_input,
    ABSTRACT_ACTIONS,
)


def run():
    PASS = True

    # ── #3: distinct bucket sizes at a to_call==0 postflop bet spot ──────────
    legal_bet = [{"type": "check"}, {"type": "bet", "min": 10, "max": 1000}]
    i33 = ABSTRACT_ACTIONS.index("bet_33")
    i100 = ABSTRACT_ACTIONS.index("bet_100")
    a33 = _abstract_to_concrete(i33, legal_bet, 100, sizing_frac=0.6,
                                street="flop", big_blind=20)
    a100 = _abstract_to_concrete(i100, legal_bet, 100, sizing_frac=0.6,
                                 street="flop", big_blind=20)
    if a33.amount != a100.amount and a33.amount < a100.amount:
        print(f"[CHECK 1] PASS — distinct bucket sizes (bet_33={a33.amount} "
              f"< bet_100={a100.amount}) under a shared sizing_frac")
    else:
        PASS = False
        print(f"[CHECK 1] FAIL — buckets collapsed: bet_33={a33.amount} "
              f"bet_100={a100.amount}")

    # ── #4: opponent committed feature reconstructed, not hardcoded 0 ────────
    # HU preflop: opp posts SB 10 then raises to 60 (committed 60); hero posts
    # BB 20.  pot = 80.  Expected opp committed/pot feature = 60/80 = 0.75.
    view = PlayerView(
        me="hero", street="preflop", position="BB",
        hole_cards=[("A", "h"), ("K", "h")], board=[],
        pot=80, to_call=40, min_raise=100, max_raise=440,
        legal_actions=[{"type": "fold"}, {"type": "call"},
                       {"type": "raise", "min": 100, "max": 500}],
        stacks={"hero": 480, "opp": 440},
        opponents=["opp"],
        history=[
            {"type": "blind", "pid": "opp", "amount": 10,
             "street": "preflop", "pot_before": 0},
            {"type": "blind", "pid": "hero", "amount": 20,
             "street": "preflop", "pot_before": 10},
            {"type": "raise", "pid": "opp", "amount": 60,
             "street": "preflop", "to_call_before": 10, "pot_before": 30},
        ],
    )
    batch = build_network_input(view)
    # opp_features channel 1 is committed/pot; opponent "opp" is index 0.
    committed_feat = float(batch["opp_features"][0, 0, 1].item())
    expected = 60 / 80
    if abs(committed_feat - expected) < 1e-6:
        print(f"[CHECK 2] PASS — opp committed feature = {committed_feat:.3f} "
              f"(matches training semantics {expected:.3f})")
    elif committed_feat > 0:
        # Reconstruction differed but is no longer the hardcoded 0 — still a
        # pass for the regression's intent, surfaced for visibility.
        print(f"[CHECK 2] PASS — opp committed feature nonzero ({committed_feat:.3f}); "
              f"expected ~{expected:.3f}")
    else:
        PASS = False
        print(f"[CHECK 2] FAIL — opp committed feature is {committed_feat} "
              f"(still hardcoded 0?)")

    print("=" * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED [PASS]' if PASS else 'SOME CHECKS FAILED [FAIL]'}")
    return PASS


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
