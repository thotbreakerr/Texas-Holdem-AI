"""
sanity_smart_bot_river.py — regression for the empty-candidate-set crash.

The heuristic Monte-Carlo bots (SmartBot, GTOBot, ExploitativeBot) estimate
equity by sampling `num_opponents` random opponent hands and taking
`max(eval_hand(opp, board) for opp in opp_hands)`.  When every remaining
opponent is already all-in (or folded), the caller derived num_opponents=0
from `acting_opponents_for(...)`, leaving `opp_hands` empty and crashing the
`max()` on the river ("max() iterable argument is empty").

This gate proves:
  1. Each MC equity function returns a finite value in [0, 1] for
     num_opponents=0 instead of crashing.
  2. A river PlayerView in which all opponents are all-in (so
     acting_opponents == []) drives each bot's act() to a valid Action.
  3. Normal behavior is preserved when candidates DO exist: a premium hand
     out-rates trash, and num_opponents=1 is unchanged.
"""
import random
import sys

sys.path.insert(0, ".")

from core.bot_api import Action, PlayerView
from bots.poker_mind_bot import SmartBot
from bots.gto_bot import GTOBot
from bots.exploitative_bot import ExploitativeBot

PASS = True


def _fail(msg):
    global PASS
    PASS = False
    print(f"  [FAIL] — {msg}")


def _ok(msg):
    print(f"  [PASS] — {msg}")


# Each entry: (name, instance, equity-method-name)
BOTS = [
    ("SmartBot", SmartBot(), "_estimate_equity"),
    ("GTOBot", GTOBot(), "_hand_strength"),
    ("ExploitativeBot", ExploitativeBot(), "_hand_strength"),
]

RIVER_BOARD = [("K", "h"), ("9", "d"), ("4", "c"), ("J", "s"), ("2", "h")]
AA = [("A", "h"), ("A", "s")]
TRASH = [("7", "c"), ("3", "d")]


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 1 — equity function with num_opponents=0 must not crash
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: MC equity with num_opponents=0 (no candidates)")
print("=" * 60)

for name, bot, method_name in BOTS:
    method = getattr(bot, method_name)
    try:
        random.seed(7)
        eq0 = method(AA, RIVER_BOARD, num_opponents=0, sims=80)
        if isinstance(eq0, float) and 0.0 <= eq0 <= 1.0:
            _ok(f"{name}.{method_name}(num_opponents=0) = {eq0:.4f} (finite, no crash)")
        else:
            _fail(f"{name}.{method_name}(num_opponents=0) returned {eq0!r}")
    except Exception as e:
        _fail(f"{name}.{method_name}(num_opponents=0) raised {type(e).__name__}: {e}")

print()


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 2 — river act() with all opponents all-in (acting_opponents == [])
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 2: river act() with every opponent all-in")
print("=" * 60)

VALID_TYPES = {"fold", "check", "call", "bet", "raise"}


def _river_view(to_call, legal_actions):
    """River view where v1/v2 are non-folded but all-in (acting set empty)."""
    return PlayerView(
        me="hero",
        street="river",
        position="BTN",
        hole_cards=AA,
        board=RIVER_BOARD,
        pot=600,
        to_call=to_call,
        min_raise=0,
        max_raise=0,
        legal_actions=legal_actions,
        stacks={"hero": 500, "v1": 0, "v2": 0},
        opponents=["v1", "v2"],          # still in the hand
        history=[],
        acting_opponents=[],             # nobody left to act → num_opponents=0
        all_in_opponents=["v1", "v2"],
    )


scenarios = [
    ("check-down (to_call=0)", _river_view(0, [{"type": "check"}])),
    ("facing all-in (to_call>0)",
     _river_view(120, [{"type": "fold"}, {"type": "call"}])),
]

for scen_name, view in scenarios:
    for name, bot, _ in BOTS:
        try:
            random.seed(11)
            action = bot.act(view)
            if isinstance(action, Action) and action.type in VALID_TYPES:
                _ok(f"{name} [{scen_name}] → {action.type}"
                    f"{'' if action.amount is None else f'({action.amount})'}")
            else:
                _fail(f"{name} [{scen_name}] returned invalid action {action!r}")
        except Exception as e:
            _fail(f"{name} [{scen_name}] raised {type(e).__name__}: {e}")

print()


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 3 — normal behavior preserved when candidates exist
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 3: candidates-exist path unchanged (premium > trash)")
print("=" * 60)

for name, bot, method_name in BOTS:
    method = getattr(bot, method_name)
    try:
        random.seed(99)
        eq_aa = sum(method(AA, RIVER_BOARD, num_opponents=1, sims=120)
                    for _ in range(10)) / 10
        random.seed(99)
        eq_trash = sum(method(TRASH, RIVER_BOARD, num_opponents=1, sims=120)
                       for _ in range(10)) / 10
        if eq_aa > eq_trash:
            _ok(f"{name}: AA ({eq_aa:.3f}) > trash ({eq_trash:.3f}) at 1 opp")
        else:
            _fail(f"{name}: AA ({eq_aa:.3f}) !> trash ({eq_trash:.3f}) at 1 opp")
    except Exception as e:
        _fail(f"{name}.{method_name} (1 opp) raised {type(e).__name__}: {e}")

print()


# ═══════════════════════════════════════════════════════════════════════════
#  Overall
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
if PASS:
    print("ALL CHECKS PASSED [PASS]")
else:
    print("SOME CHECKS FAILED [FAIL]")
    sys.exit(1)
print("=" * 60)
