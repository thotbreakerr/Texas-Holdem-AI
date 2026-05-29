"""
Regression for the ML/RL preflop hand-strength normalization bug.

Bug (pre-fix): _estimate_hand_strength divided the preflop rank-sum heuristic
(max = AA = 64) by EVAL_HAND_MAX (~134M), so every preflop hand scored ~4e-7.
The fallback strategy therefore folded even AA to any nonzero pot odds and never
raised preflop.

This test checks both bots:
  - preflop ordering AA > AKs > 72o
  - AA strength is well above ordinary pot odds
  - the fallback does NOT fold AA facing an ordinary preflop bet
"""
import sys

sys.path.insert(0, "/Users/jaroslavaupart/Desktop/Projects/Texas-Holdem-AI")

from core.bot_api import PlayerView
from bots.ml_bot import MLBot
from bots.rl_bot import RLBot

AA = [("A", "s"), ("A", "h")]
AKs = [("A", "s"), ("K", "s")]
T72o = [("7", "d"), ("2", "c")]


def _view(hole, to_call, pot):
    legal = [{"type": "fold"}, {"type": "call"},
             {"type": "raise", "min": to_call * 2, "max": 500}]
    if to_call == 0:
        legal = [{"type": "check"}, {"type": "bet", "min": 20, "max": 500}]
    return PlayerView(
        me="HERO", street="preflop", position="BTN",
        hole_cards=hole, board=[], pot=pot, to_call=to_call,
        min_raise=to_call * 2, max_raise=500, legal_actions=legal,
        stacks={"HERO": 500, "OPP": 500}, opponents=["OPP"], history=[],
    )


def check_bot(name, bot):
    ok = True

    s_aa = bot._estimate_hand_strength(AA, [])
    s_ak = bot._estimate_hand_strength(AKs, [])
    s_72 = bot._estimate_hand_strength(T72o, [])

    if s_aa > s_ak > s_72:
        print(f"[{name}] PASS — ordering AA({s_aa:.3f}) > AKs({s_ak:.3f}) > 72o({s_72:.3f})")
    else:
        ok = False
        print(f"[{name}] FAIL — ordering broken: AA={s_aa:.3f} AKs={s_ak:.3f} 72o={s_72:.3f}")

    # AA must be a strong signal, not ~0.
    if s_aa > 0.6:
        print(f"[{name}] PASS — AA strength {s_aa:.3f} > 0.6")
    else:
        ok = False
        print(f"[{name}] FAIL — AA strength {s_aa:.3f} collapsed near 0")

    # Fallback must not fold AA to an ordinary preflop bet (pot_odds ~0.25).
    action = bot._fallback_strategy(_view(AA, to_call=20, pot=60))
    if action.type != "fold":
        print(f"[{name}] PASS — fallback did not fold AA (chose {action.type})")
    else:
        ok = False
        print(f"[{name}] FAIL — fallback folded AA to ordinary pot odds")

    return ok


def run():
    PASS = True
    PASS &= check_bot("MLBot", MLBot(use_fallback=True))
    PASS &= check_bot("RLBot", RLBot(use_fallback=True))
    print("=" * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED [PASS]' if PASS else 'SOME CHECKS FAILED [FAIL]'}")
    return PASS


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
