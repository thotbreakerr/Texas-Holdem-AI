"""
sanity_cfr_search_shaping.py — Path A inference search must not collapse
to a single-sample argmax.

Phase 3 (2026-06-10). The old _subgame_search weighted actions by
prior * exp(ev - max_ev) with ev in RAW CHIPS: a 20-chip EV gap (e^20)
annihilated the trained prior, so the bot argmaxed one noisy sampled
line. The fix mirrors Path B (deep_cfr_bot.py): EVs in big-blind units,
prior-weighted baseline, temperature 20, advantage clip +/-2, and a 25%
blend back into the prior.

Checks:
  1. Anti-collapse — a huge EV gap shifts mass but cannot erase the
     prior; refined strategy is never one-hot.
  2. Chip-scale invariance — the same spot at 10x chips/blinds produces
     the same refined distribution (EVs are normalized to BB).
  3. Noise dominance — EV differences within sampling noise leave the
     prior's argmax unchanged and the distribution close to the prior.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.bot_api import PlayerView
from bots.cfr_bot import CFRBot, NUM_ACTIONS

PASS = True


def check(label, cond, detail=""):
    global PASS
    if cond:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label}  {detail}")
        PASS = False


def make_view(scale=1):
    """3-handed preflop spot, facing a raise (sizing-parity test's shape).

    scale multiplies every chip quantity; big blind = 20*scale.
    """
    s = scale
    return PlayerView(
        me="hero", street="preflop", position="BB",
        hole_cards=[("A", "h"), ("K", "h")], board=[],
        pot=90 * s, to_call=40 * s, min_raise=100 * s, max_raise=480 * s,
        legal_actions=[
            {"type": "fold"},
            {"type": "call"},
            {"type": "raise", "min": 100 * s, "max": 500 * s},
        ],
        stacks={"sb": 490 * s, "hero": 480 * s, "opp": 440 * s},
        opponents=["sb", "opp"],
        history=[
            {"type": "blind", "pid": "sb", "amount": 10 * s,
             "street": "preflop", "pot_before": 0},
            {"type": "blind", "pid": "hero", "amount": 20 * s,
             "street": "preflop", "pot_before": 10 * s},
            {"type": "raise", "pid": "opp", "amount": 60 * s,
             "street": "preflop", "to_call_before": 20 * s,
             "pot_before": 30 * s},
        ],
    )


def run() -> bool:
    bot = CFRBot(iterations=1, profile_path=None)

    print("Section 1: anti-collapse on huge EV gaps")
    legal = [0, 1]
    prior = [0.0] * NUM_ACTIONS
    prior[0], prior[1] = 0.9, 0.1
    # 50 BB EV gap in favor of the prior's UNDERDOG action.
    evs = {0: 0.0, 1: 50.0}
    refined = bot._shape_search_strategy(evs, prior, legal)
    print(f"  prior 0.90/0.10, EV gap +50BB toward action 1 -> "
          f"refined {refined[0]:.3f}/{refined[1]:.3f}")
    check("prior survives (refined[0] >= 0.5)", refined[0] >= 0.5,
          f"refined[0]={refined[0]:.3f}")
    check("EV still shifts mass (refined[1] > prior[1])",
          refined[1] > prior[1], f"refined[1]={refined[1]:.3f}")
    check("not one-hot (max < 0.99)", max(refined) < 0.99,
          f"max={max(refined):.3f}")
    check("distribution sums to 1",
          abs(sum(refined[a] for a in legal) - 1.0) < 1e-9)

    print("Section 2: chip-scale invariance via _subgame_search")
    # Deterministic subtree: leaf value = 1.5 * pot in chips, so each
    # action's EV = pot + 0.5 * its added cost — EVs genuinely differ
    # per action AND scale linearly with the spot, so the BB-normalized
    # refined distribution must be identical at 10x chips.
    bot._search_subtree = lambda state, hero_seat, depth: 1.5 * state.pot
    prior2 = [0.0] * NUM_ACTIONS
    legal2 = [0, 1, 3, 7]
    for a, p in zip(legal2, (0.4, 0.3, 0.2, 0.1)):
        prior2[a] = p
    r1 = bot._subgame_search(make_view(scale=1), prior2, legal2, depth=2)
    r10 = bot._subgame_search(make_view(scale=10), prior2, legal2, depth=2)
    print("  scale x1 :", [f"{r1[a]:.4f}" for a in legal2])
    print("  scale x10:", [f"{r10[a]:.4f}" for a in legal2])
    drift = max(abs(r1[a] - r10[a]) for a in legal2)
    # Integer chip rounding moves bucket targets slightly; allow 2%.
    check("refined distribution invariant to 10x chip scale",
          drift < 0.02, f"max drift {drift:.4f}")

    print("Section 3: noise-scale EV differences leave the prior in charge")
    prior3 = [0.0] * NUM_ACTIONS
    prior3[0], prior3[1] = 0.7, 0.3
    evs3 = {0: 0.0, 1: 0.1}   # 0.1 BB apart — well inside sampling noise
    refined3 = bot._shape_search_strategy(evs3, prior3, [0, 1])
    print(f"  prior 0.70/0.30, EVs 0.0/0.1 BB -> "
          f"refined {refined3[0]:.3f}/{refined3[1]:.3f}")
    check("prior argmax preserved", refined3[0] > refined3[1])
    check("stays close to prior (within 0.05)",
          abs(refined3[0] - 0.7) < 0.05 and abs(refined3[1] - 0.3) < 0.05,
          f"{refined3[0]:.3f}/{refined3[1]:.3f}")

    print("\nOVERALL:", "ALL CHECKS PASSED" if PASS else "SOME CHECKS FAILED")
    return PASS


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
