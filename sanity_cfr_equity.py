"""
Sanity check: CFR _quick_equity and _estimate_action_value must produce
sensible values for both postflop AND preflop scenarios.

Tests:
  1. Postflop: AQ on A72 rainbow — equity must decrease with more opponents.
  2. Preflop: AA, KK, AKs, AQo, 72o — equity must decrease with more
     opponents, and premium hands must NOT produce fold-favoring signals.
  3. Action value categories:
     - Premium hands must not fold.
     - Trash hands must fold.
     - Marginal hands must not prefer all-in.
"""
import sys
sys.path.insert(0, ".")

from bots.cfr_bot import (
    CFRBot, ABSTRACT_ACTIONS,
    _position_bucket, _spr_bucket, _info_set_key,
)
from core.bot_api import PlayerView

# Build a fresh CFR bot (no loaded profile — we only need equity/value functions)
bot = CFRBot(iterations=0, inference_mode=True)

PASS = True
rounds = 50  # average over 50 calls for stable MC numbers (100 sims each)


def avg_equity(hole, board, n_opp):
    """Run many batches and average for stability (each call = 100 sims)."""
    total = sum(bot._quick_equity(hole, board, n_opponents=n_opp)
                for _ in range(rounds))
    return total / rounds


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 1 — Postflop: AQ on A72 rainbow (existing test)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: Postflop equity (AQ on A72 rainbow)")
print("=" * 60)

hole_aq = [("A", "h"), ("Q", "s")]
board_a72 = [("A", "s"), ("7", "d"), ("2", "c")]

postflop_equities = {}
for n in [1, 2, 5]:
    eq = avg_equity(hole_aq, board_a72, n)
    postflop_equities[n] = eq
    print(f"  n_opponents={n}  →  equity = {eq:.4f}")

vals = [postflop_equities[n] for n in [1, 2, 5]]
if all(vals[i] > vals[i+1] for i in range(len(vals) - 1)):
    print("  [PASS] — postflop equities are monotonically decreasing\n")
else:
    print(f"  [FAIL] — postflop equities NOT monotonically decreasing: {vals}\n")
    PASS = False


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 2 — Preflop equity: multiple hands at 1 and 5 opponents
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 2: Preflop equity (AA, KK, AKs, AQo, 72o)")
print("=" * 60)

preflop_hands = {
    "AA":  [("A", "h"), ("A", "s")],
    "KK":  [("K", "h"), ("K", "s")],
    "AKs": [("A", "h"), ("K", "h")],
    "AQo": [("A", "h"), ("Q", "s")],
    "72o": [("7", "h"), ("2", "s")],
}

preflop_equities = {}  # name -> {n_opp: equity}

for name, hole in preflop_hands.items():
    preflop_equities[name] = {}
    for n in [1, 5]:
        eq = avg_equity(hole, [], n)
        preflop_equities[name][n] = eq
        print(f"  {name:4s}  n_opponents={n}  →  equity = {eq:.4f}")

    # Check monotonic: 1-opp > 5-opp
    e1 = preflop_equities[name][1]
    e5 = preflop_equities[name][5]
    if e1 > e5:
        print(f"  {name:4s}  [PASS] monotonic (1-opp > 5-opp)")
    else:
        print(f"  {name:4s}  [FAIL] — 1-opp ({e1:.4f}) <= 5-opp ({e5:.4f})")
        PASS = False

print()

# ── Absolute bounds for premiums (catches the **N regression) ─────────
print("Absolute bound checks:")

aa_5 = preflop_equities["AA"][5]
if aa_5 > 0.45:
    print(f"  AA  at 5-opp: {aa_5:.4f} > 0.45  [PASS]")
else:
    print(f"  AA  at 5-opp: {aa_5:.4f} <= 0.45  [FAIL] (true ~0.50)")
    PASS = False

kk_5 = preflop_equities["KK"][5]
if kk_5 > 0.40:
    print(f"  KK  at 5-opp: {kk_5:.4f} > 0.40  [PASS]")
else:
    print(f"  KK  at 5-opp: {kk_5:.4f} <= 0.40  [FAIL] (true ~0.43)")
    PASS = False

trash_5 = preflop_equities["72o"][5]
if trash_5 < 0.20:
    print(f"  72o at 5-opp: {trash_5:.4f} < 0.20  [PASS]")
else:
    print(f"  72o at 5-opp: {trash_5:.4f} >= 0.20  [FAIL] (true ~0.09)")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 3 — Action values by hand category
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 3: Action values by hand category")
print("=" * 60)

pot = 30
call_amount = 10
hero_stack = 1000
n_opp = 5

premium_hands = [
    ("AA", [("A", "h"), ("A", "s")]),
    ("KK", [("K", "h"), ("K", "s")]),
    ("QQ", [("Q", "h"), ("Q", "s")]),
]

trash_hands = [
    ("72o", [("7", "h"), ("2", "s")]),
    ("32o", [("3", "h"), ("2", "s")]),
]

marginal_hands = [
    ("AKs", [("A", "h"), ("K", "h")]),
    ("AQo", [("A", "h"), ("Q", "s")]),
    ("KQo", [("K", "h"), ("Q", "s")]),
    ("JTs", [("J", "h"), ("T", "h")]),
    ("99",  [("9", "h"), ("9", "s")]),
]


def evaluate_action_values(hand_name, hole):
    """Print all abstract action values and return the best action label."""
    # Compute equity (use preflop_equities if we already have it, else compute)
    if hand_name in preflop_equities and 5 in preflop_equities[hand_name]:
        eq = preflop_equities[hand_name][5]
    else:
        eq = avg_equity(hole, [], n_opp)

    print(
        f"\n  {hand_name}  "
        f"(equity={eq:.4f}, pot={pot}, call={call_amount}, stack={hero_stack}):"
    )

    action_values = {}
    for i, label in enumerate(ABSTRACT_ACTIONS):
        val = bot._estimate_action_value(
            i,
            pot,
            eq,
            n_opponents=n_opp,
            call_amount=call_amount,
            hero_stack=hero_stack,
        )
        action_values[label] = val
        print(f"    {label:12s}  value = {val:+.4f}")

    best_action = max(action_values, key=action_values.get)
    print(f"    → best = {best_action}")
    return best_action


print("\nPREMIUM — must not fold")
for hand_name, hole in premium_hands:
    best_action = evaluate_action_values(hand_name, hole)
    if best_action == "fold":
        print(f"    [FAIL] — {hand_name}'s best action is FOLD!")
        PASS = False
    else:
        print(f"    [PASS] — {hand_name} does NOT fold")

print("\nTRASH — must fold")
for hand_name, hole in trash_hands:
    best_action = evaluate_action_values(hand_name, hole)
    if best_action == "fold":
        print(f"    [PASS] — {hand_name} folds")
    else:
        print(f"    [FAIL] — {hand_name}'s best action is {best_action}, not fold!")
        PASS = False

print("\nMARGINAL — must not prefer all-in")
for hand_name, hole in marginal_hands:
    best_action = evaluate_action_values(hand_name, hole)
    if best_action == "all_in":
        print(f"    [FAIL] — {hand_name}'s best action is ALL-IN!")
        PASS = False
    else:
        print(f"    [PASS] — {hand_name} does not prefer all-in")

print()


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 4 — Position bucket helper
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 4: Position bucket helper")
print("=" * 60)

pos_cases = [
    ("UTG",  "early"),
    ("BTN",  "late"),
    ("CO",   "late"),
    ("MP",   "middle"),
    ("BB",   "blinds"),
    ("SB",   "blinds"),
    ("Foo",  "middle"),   # safe fallback for unknown labels
]

for label, expected in pos_cases:
    result = _position_bucket(label)
    if result == expected:
        print(f"  _position_bucket({label!r:8s}) == {expected!r:10s}  [PASS]")
    else:
        print(f"  _position_bucket({label!r:8s}) == {result!r:10s}  [FAIL] expected {expected!r}")
        PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 5 — SPR bucket helper
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 5: SPR bucket helper")
print("=" * 60)

spr_cases = [
    # (hero_stack, pot, opp_stacks, expected)
    (1000, 30,  [1000], "high"),   # SPR ≈ 33
    (1000, 100, [1000], "mid"),    # SPR = 10
    (100,  50,  [1000], "low"),    # effective=100, SPR=2
    (100,  0,   [1000], "high"),   # pot=0 → safe default
    (1000, 30,  [],     "high"),   # no opponents → hero_stack/pot
]

for hero, p, opps, expected in spr_cases:
    result = _spr_bucket(hero, p, opps)
    tag = f"_spr_bucket({hero}, {p}, {opps})"
    if result == expected:
        print(f"  {tag:40s} == {expected!r:8s}  [PASS]")
    else:
        print(f"  {tag:40s} == {result!r:8s}  [FAIL] expected {expected!r}")
        PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 6 — Info-set key includes position and SPR
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 6: Info-set key includes position and SPR")
print("=" * 60)

# Helper to build a minimal PlayerView for act()
def _make_view(position="BTN", hero_stack=1000, pot=30, opponents=None):
    """Build a minimal PlayerView for testing info-key generation."""
    if opponents is None:
        opponents = ["opp1"]
    return PlayerView(
        me="hero",
        street="preflop",
        position=position,
        hole_cards=[("A", "h"), ("A", "s")],
        board=[],
        pot=pot,
        to_call=10,
        min_raise=20,
        max_raise=hero_stack,
        legal_actions=[
            {"type": "fold"},
            {"type": "call"},
            {"type": "raise", "min": 20, "max": hero_stack},
        ],
        stacks={"hero": hero_stack, **{o: 1000 for o in opponents}},
        opponents=opponents,
        history=[],
    )

# 6a: Same hand, different positions → keys must differ
view_utg = _make_view(position="UTG")
view_btn = _make_view(position="BTN")

# Run act() in inference mode — the bot won't modify regrets, but we can
# inspect the info key it would have built. We reconstruct the key using
# the same helpers the bot uses internally.
from bots.cfr_bot import (
    _preflop_bucket, _abstract_history, _info_set_key as _isk,
)

def _build_key(view):
    """Replicate the info-key construction from act() for testing."""
    bucket = _preflop_bucket(view.hole_cards)
    hist_key = _abstract_history(view.history or [], view.pot)
    n_opp = max(1, len(view.opponents) if view.opponents else 1)
    pos_b = _position_bucket(view.position)
    hero_stack = int(view.stacks.get(view.me, 0))
    opp_stacks = [
        int(view.stacks.get(o, 0))
        for o in (view.opponents or [])
        if int(view.stacks.get(o, 0)) > 0
    ]
    spr_b = _spr_bucket(hero_stack, view.pot, opp_stacks)
    return _isk(view.street, bucket, hist_key,
                n_opponents=n_opp, position_bucket=pos_b,
                spr_bucket=spr_b)

key_utg = _build_key(view_utg)
key_btn = _build_key(view_btn)
print(f"  UTG key: {key_utg}")
print(f"  BTN key: {key_btn}")

if key_utg != key_btn:
    print("  [PASS] — different positions produce different info-set keys")
else:
    print("  [FAIL] — UTG and BTN produced the SAME key!")
    PASS = False

# 6b: Same position, different SPR → keys must differ
view_deep = _make_view(hero_stack=1000, pot=30)   # SPR ≈ 33 → high
view_shallow = _make_view(hero_stack=100, pot=30)  # SPR ≈ 3.3 → low

key_deep = _build_key(view_deep)
key_shallow = _build_key(view_shallow)
print(f"  Deep key:    {key_deep}")
print(f"  Shallow key: {key_shallow}")

if key_deep != key_shallow:
    print("  [PASS] — different SPRs produce different info-set keys")
else:
    print("  [FAIL] — deep and shallow produced the SAME key!")
    PASS = False


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 7 — Gate 2A: AIVAT leaf evaluator integration
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 7: AIVAT leaf evaluator integration")
print("=" * 60)

from bots.cfr_bot import (
    _GameState, _FULL_DECK, CFRBot, _build_game_state_from_view,
    _legal_abstract_actions,
)
from core.aivat import Snapshot as _Snapshot, value as _aivat_value
from core.equity import equity as _vanilla_equity
import random as _random_mod

# Build a GameState by hand
gs = _GameState(
    pot=100,
    stacks=[200, 200],
    committed_per_seat=[50, 50],
    alive=[True, True],
    street="turn",
    board=[("K","h"), ("7","d"), ("2","c"), ("J","s")],
    hole_cards={0: (("A","h"), ("A","s")), 1: (("7","c"), ("2","d"))},
    seat_order=[0, 1],
    action_idx=0,
    history_tokens="",
    deck_remaining=[c for c in _FULL_DECK if c not in
        {("K","h"),("7","d"),("2","c"),("J","s"),("A","h"),("A","s"),("7","c"),("2","d")}],
    real_contributions=True,
)

# Build corresponding Snapshot manually
snap = _Snapshot(
    hole_cards={0: (("A","h"), ("A","s")), 1: (("7","c"), ("2","d"))},
    board=(("K","h"), ("7","d"), ("2","c"), ("J","s")),
    pot=100,
    stacks=(200, 200),
    alive=(True, True),
    to_call=0,
    hero_committed=50,
    committed_per_seat=(50, 50),
)

bot_test = CFRBot(inference_mode=True)
leaf_val = bot_test._leaf_value(gs, hero_seat=0)
snap_val = _aivat_value(snap, 0, mode="chip_ev", n_sims=200)

print(f"  _leaf_value: {leaf_val:.2f}")
print(f"  aivat_value: {snap_val:.2f}")
if abs(leaf_val - snap_val) < 30:  # generous tolerance for MC
    print("  [PASS] — real-contribution _leaf_value ≈ aivat_value")
else:
    print(f"  [FAIL] — delta = {abs(leaf_val - snap_val):.2f}")
    PASS = False

print()
print("  Testing estimated-contribution inference fallback...")
gs_est = _GameState(
    pot=100,
    stacks=[200, 200],
    committed_per_seat=[0, 0],  # bogus but explicitly estimated
    alive=[True, True],
    street="river",
    board=[("K","h"),("7","d"),("2","c"),("J","s"),("3","h")],
    hole_cards={0: (("A","h"), ("A","s")), 1: (("7","c"), ("2","d"))},
    seat_order=[0, 1],
    real_contributions=False,
)
_random_mod.seed(123)
leaf_est = bot_test._leaf_value(gs_est, hero_seat=0)
_random_mod.seed(123)
eq_est = _vanilla_equity(
    [("A","h"), ("A","s")],
    [("K","h"),("7","d"),("2","c"),("J","s"),("3","h")],
    n_opponents=1,
    n_sims=200,
) * 100
print(f"  fallback leaf: {leaf_est:.2f}")
print(f"  equity * pot:  {eq_est:.2f}")
if abs(leaf_est - eq_est) < 1e-9:
    print("  [PASS] — estimated contributions use vanilla equity fallback")
else:
    print("  [FAIL] — estimated path did not use vanilla equity fallback")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 8 — Gate 2A: Opponent-stat bucket in info-set key
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 8: Opponent-stat bucket in info-set key")
print("=" * 60)

# Two keys with different opp_stat_bucket values should differ
key_ta = _isk("preflop", 10, "CC", n_opponents=1,
              position_bucket="late", spr_bucket="high",
              opp_stat_bucket="TA")
key_lp = _isk("preflop", 10, "CC", n_opponents=1,
              position_bucket="late", spr_bucket="high",
              opp_stat_bucket="LP")
print(f"  Key with TA: {key_ta}")
print(f"  Key with LP: {key_lp}")
if key_ta != key_lp:
    print("  [PASS] — different opp_stat_buckets produce different keys")
else:
    print("  [FAIL] — keys are identical!")
    PASS = False

# Verify 7-field format (6 colons)
if key_ta.count(":") == 6:
    print(f"  [PASS] — key has 6 colons (7 fields)")
else:
    print(f"  [FAIL] — key has {key_ta.count(':')} colons, expected 6")
    PASS = False

# Old-format profile rejection
print()
print("  Testing old-format profile key rejection...")
import pickle, tempfile, os as _os
old_profile = {
    "nodes": {
        # Old 5-colon keys (pre-Gate-2A)
        "preflop:1:late:high:10:CC": {"regret_sum": [0.0]*8, "strategy_sum": [1.0]*8},
        "flop:2:middle:mid:5:SQ": {"regret_sum": [0.0]*8, "strategy_sum": [1.0]*8},
        # Valid 6-colon key
        "preflop:1:late:high:TA:10:CC": {"regret_sum": [0.0]*8, "strategy_sum": [1.0]*8},
    },
    "hands_played": 100,
    "total_iterations": 500,
}
with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
    pickle.dump(old_profile, f)
    tmp_path = f.name

test_bot = CFRBot(inference_mode=True)
test_bot.load(tmp_path)
_os.unlink(tmp_path)

if len(test_bot._nodes) == 1:
    print(f"  [PASS] — kept 1 valid key, dropped 2 old-format keys")
else:
    print(f"  [FAIL] — expected 1 valid key, got {len(test_bot._nodes)}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 9 — Gate 2A: Finer card buckets (50 buckets)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 9: Finer card buckets (50)")
print("=" * 60)

from core.equity import equity_bucket as _eb

# Sweep a variety of hand × board combos and check that 50 distinct
# bucket values are reachable.
hands = [
    [("A","h"),("A","s")], [("K","h"),("K","d")], [("Q","s"),("Q","c")],
    [("J","h"),("T","h")], [("9","d"),("8","d")], [("7","c"),("6","c")],
    [("5","s"),("4","s")], [("3","h"),("2","h")], [("A","h"),("K","h")],
    [("A","d"),("2","c")], [("K","s"),("Q","d")], [("T","c"),("9","s")],
]
boards = [
    [("A","d"),("7","s"),("2","d")],
    [("K","c"),("Q","h"),("J","d")],
    [("T","d"),("9","h"),("8","s")],
    [("4","h"),("3","d"),("2","s")],
    [("A","c"),("K","d"),("5","h")],
]

bucket_set = set()
for h in hands:
    for b in boards:
        # Skip if cards overlap
        if any(c in b for c in h):
            continue
        bkt = _eb(h, b, n_opponents=1, n_buckets=50, n_sims=200)
        bucket_set.add(bkt)

print(f"  Distinct buckets found: {len(bucket_set)} / 50")
if len(bucket_set) >= 10:
    print("  [PASS] — at least 10 distinct buckets (not collapsed)")
else:
    print("  [FAIL] — buckets collapsed")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 10 — Gate 2A: Real-time subgame search
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 10: Real-time subgame search")
print("=" * 60)

import time as _time_mod

# Premium hand: AA with high pot — search should favor bet/raise
view_aa = PlayerView(
    me="hero",
    street="flop",
    position="BTN",
    hole_cards=[("A","h"), ("A","s")],
    board=[("K","d"), ("7","c"), ("2","h")],
    pot=100,
    to_call=20,
    min_raise=40,
    max_raise=500,
    legal_actions=[
        {"type": "fold"},
        {"type": "call"},
        {"type": "raise", "min": 40, "max": 500},
    ],
    stacks={"hero": 500, "opp1": 500},
    opponents=["opp1"],
    history=[],
)

search_bot = CFRBot(inference_mode=True, search_depth=3)
uniform = [0.0] * 8
legal_mask_aa = [0, 1, 2, 3, 4, 5, 6, 7]
for a in legal_mask_aa:
    uniform[a] = 1.0 / len(legal_mask_aa)

t0 = _time_mod.monotonic()
strategy_aa = search_bot._subgame_search(view_aa, uniform, legal_mask_aa, depth=3)
elapsed_aa = _time_mod.monotonic() - t0

# Fold weight should be lowest for AA
fold_w = strategy_aa[0]
bet_w = sum(strategy_aa[i] for i in range(2, 8))
print(f"  AA search: fold={fold_w:.3f}, bet/raise={bet_w:.3f}, time={elapsed_aa:.2f}s")
if bet_w > fold_w:
    print("  [PASS] — bet/raise > fold for AA")
else:
    print("  [FAIL] — fold preferred over bet for AA")
    PASS = False

if elapsed_aa < 2.0:
    print(f"  [PASS] — search completed in {elapsed_aa:.2f}s (< 2s budget)")
else:
    print(f"  [FAIL] — search took {elapsed_aa:.2f}s (> 2s budget)")
    PASS = False

# Counter-test: 72o on connected high board — should favor fold
view_72 = PlayerView(
    me="hero",
    street="flop",
    position="UTG",
    hole_cards=[("7","s"), ("2","d")],
    board=[("A","d"), ("K","c"), ("Q","h")],
    pot=100,
    to_call=50,
    min_raise=100,
    max_raise=500,
    legal_actions=[
        {"type": "fold"},
        {"type": "call"},
        {"type": "raise", "min": 100, "max": 500},
    ],
    stacks={"hero": 500, "opp1": 500},
    opponents=["opp1"],
    history=[],
)

t0 = _time_mod.monotonic()
strategy_72 = search_bot._subgame_search(view_72, uniform, legal_mask_aa, depth=3)
elapsed_72 = _time_mod.monotonic() - t0

fold_w_72 = strategy_72[0]
bet_w_72 = sum(strategy_72[i] for i in range(2, 8))
print(f"  72o search: fold={fold_w_72:.3f}, bet/raise={bet_w_72:.3f}, time={elapsed_72:.2f}s")
if abs(sum(strategy_72) - 1.0) < 1e-6 and all(w >= 0 for w in strategy_72):
    print("  [PASS] — 72o search returned a valid strategy distribution")
else:
    print("  [FAIL] — 72o search returned an invalid strategy distribution")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 11 — Gate 2A: Smoke tournament (3 tournaments, no crashes)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 11: Smoke tournament (3 tournaments)")
print("=" * 60)

import io
from contextlib import redirect_stdout

from core.engine import Table, Seat
from bots import create_bot, escalate_blinds

smoke_pass = True
search_times = []

for tournament_i in range(1, 4):
    try:
        # Build 6 bots: 1 CFR with an explicit in-memory empty profile + 5 others.
        # The create_bot("cfr") factory now fails loudly when the default
        # inference profile is missing.
        pids = [f"P{i}" for i in range(1, 7)]
        bots = {}
        bots["P1"] = CFRBot(iterations=0, inference_mode=True)
        for pid in pids[1:]:
            bots[pid] = create_bot("smart")

        seats = [Seat(player_id=pid, chips=500) for pid in pids]
        table = Table()
        dealer_index = 0
        hand_count = 0

        buf = io.StringIO()
        with redirect_stdout(buf):
            while True:
                active = [s for s in seats if s.chips > 0]
                if len(active) <= 1:
                    break
                hand_count += 1
                sb, bb = escalate_blinds(hand_count, 5, 10, 50)
                active_bots = {s.player_id: bots[s.player_id] for s in active}

                t0_h = _time_mod.monotonic()
                table.play_hand(
                    seats=active,
                    small_blind=sb,
                    big_blind=bb,
                    dealer_index=dealer_index % len(active),
                    bot_for=active_bots,
                    on_event=None,
                    log_decisions=False,
                )
                elapsed_h = _time_mod.monotonic() - t0_h
                search_times.append(elapsed_h)

                dealer_index += 1
                if hand_count >= 500:  # safety cap for smoke
                    break

        winner = max(seats, key=lambda s: s.chips).player_id
        print(f"  Tournament {tournament_i}: {hand_count} hands, winner={winner}  [PASS]")

    except Exception as e:
        print(f"  Tournament {tournament_i}: CRASHED — {e}")
        smoke_pass = False
        PASS = False

if smoke_pass:
    avg_time = sum(search_times) / len(search_times) if search_times else 0
    max_time = max(search_times) if search_times else 0
    print(f"  All 3 tournaments completed without crashes [PASS]")
    print(f"  Hand timings: avg={avg_time:.3f}s, max={max_time:.3f}s, "
          f"total hands={len(search_times)}")
else:
    print("  [FAIL] — smoke tournament crashed")

print()


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 12 — Import smoke: CFRBot with empty profile on synthetic view
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 12: Import smoke — empty profile act()")
print("=" * 60)

from bots.cfr_bot import CFRBot as _CFRBotCheck

test_b = _CFRBotCheck(inference_mode=True)
view_smoke = _make_view(position="BTN", hero_stack=500, pot=20)
action_out = test_b.act(view_smoke)
print(f"  Returned action: {action_out.type} (amount={getattr(action_out, 'amount', None)})")
if action_out.type in ("fold", "check", "call", "bet", "raise"):
    print("  [PASS] — valid action with empty profile")
else:
    print(f"  [FAIL] — invalid action type: {action_out.type}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 13 — Gate 2A: Training-path wiring
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 13: Training-path wiring")
print("=" * 60)

def _training_view(hole, board=None, large_pot=False):
    if board is None:
        board = [("K","h"), ("7","d"), ("2","c"), ("J","s")]
    flop_bet = 455 if large_pot else 20
    turn_pot_before = 970 if large_pot else 100
    pot_now = 1000 if large_pot else 130
    hero_stack = 1000 if large_pot else 200
    opp_stack = 0 if large_pot else 170
    history = [
        {"street": "preflop", "pid": "hero", "type": "raise",
         "amount": 30, "to_call_before": 10, "pot_before": 15},
        {"street": "preflop", "pid": "opp", "type": "call",
         "amount": None, "to_call_before": 20, "pot_before": 40},
        {"street": "flop", "pid": "opp", "type": "check",
         "amount": None, "to_call_before": 0, "pot_before": 60},
        {"street": "flop", "pid": "hero", "type": "bet",
         "amount": flop_bet, "to_call_before": 0, "pot_before": 60},
        {"street": "flop", "pid": "opp", "type": "call",
         "amount": None, "to_call_before": flop_bet,
         "pot_before": 60 + flop_bet},
        {"street": "turn", "pid": "opp", "type": "bet",
         "amount": 30, "to_call_before": 0, "pot_before": turn_pot_before},
    ]
    return PlayerView(
        me="hero",
        street="turn",
        position="BTN",
        hole_cards=hole,
        board=board,
        pot=pot_now,
        to_call=30,
        min_raise=60,
        max_raise=hero_stack,
        legal_actions=[
            {"type": "fold"},
            {"type": "call"},
            {"type": "raise", "min": 60, "max": 200},
        ],
        stacks={"hero": hero_stack, "opp": opp_stack},
        opponents=["opp"],
        history=history,
    )

_random_mod.seed(2026)  # seed BEFORE constructor so bot._rng cascades from this
train_bot = CFRBot(iterations=1, inference_mode=False)
view_train_aa = _training_view(
    [("A","h"), ("A","s")],
    board=[("A","d"), ("A","c"), ("K","h"), ("Q","s")],
    large_pot=True,
)
for _ in range(50):
    train_bot.act(view_train_aa)
root_state = train_bot._build_training_game_state(
    view_train_aa,
    hero_hole=view_train_aa.hole_cards,
    opp_hands=[(("Q","c"), ("T","c"))],
    board=view_train_aa.board,
)
root_key = train_bot._info_key_for_state(root_state, root_state.hero_seat)
node = train_bot._nodes.get(root_key)
valid_key_count = sum(1 for k in train_bot._nodes if k.count(":") == 6)
if node and valid_key_count > 0:
    print(f"  Root key: {root_key}")
    print(f"  Regrets: fold={node.regret_sum[0]:+.2f}, "
          f"bet_67={node.regret_sum[4]:+.2f}, "
          f"bet_100={node.regret_sum[6]:+.2f}")
    if node.regret_sum[4] > 0 and node.regret_sum[6] > 0 and node.regret_sum[0] <= 0:
        print("  [PASS] — AA training regrets reward aggressive actions")
    else:
        print("  [FAIL] — AA root regrets did not move toward aggression")
        PASS = False
    print(f"  [PASS] — wrote {valid_key_count} Gate-2A-format keys")
else:
    print("  [FAIL] — recursive training did not populate root Gate-2A key")
    PASS = False

_random_mod.seed(2027)  # seed BEFORE constructor so bot._rng cascades from this
trash_bot = CFRBot(iterations=1, inference_mode=False)
view_train_72 = PlayerView(
    me="hero",
    street="preflop",
    position="BB",
    hole_cards=[("7","s"), ("2","d")],
    board=[],
    pot=130,
    to_call=110,
    min_raise=220,
    max_raise=490,
    legal_actions=[
        {"type": "fold"},
        {"type": "call"},
        {"type": "raise", "min": 220, "max": 500},
    ],
    stacks={"opp": 380, "hero": 490},
    opponents=["opp"],
    history=[{
        "street": "preflop",
        "pid": "opp",
        "type": "raise",
        "amount": 120,
        "to_call_before": 0,
        "pot_before": 15,
    }],
)
for _ in range(50):
    trash_bot.act(view_train_72)
trash_state = trash_bot._build_training_game_state(
    view_train_72,
    hero_hole=view_train_72.hole_cards,
    opp_hands=[(("A","c"), ("A","d"))],
    board=view_train_72.board,
)
trash_key = trash_bot._info_key_for_state(trash_state, trash_state.hero_seat)
trash_node = trash_bot._nodes.get(trash_key)
# Invariant: air (72o that cannot improve vs AA) prefers folding over calling
# to showdown — fold regret exceeds call regret.  This holds regardless of bet
# sizing.  (The prior `fold regret > 0` check assumed fold was the single best
# action, which only held while the bucket-collapse bug disabled the
# intermediate bluff-raise sizings; with correct pot-relative sizing the bot
# may bluff-raise, but folding still dominates calling.)
if trash_node and trash_node.regret_sum[0] > trash_node.regret_sum[1]:
    print(f"  [PASS] — 72o prefers folding over calling "
          f"({trash_node.regret_sum[0]:+.1f} > {trash_node.regret_sum[1]:+.1f})")
else:
    got = (trash_node.regret_sum[0], trash_node.regret_sum[1]) if trash_node else None
    print(f"  [FAIL] — 72o fold regret does not exceed call regret ({got})")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 14 — Gate 2A: Search tree action ordering
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 14: Search tree action ordering")
print("=" * 60)

order_view = PlayerView(
    me="Hero",
    street="preflop",
    position="UTG+1",
    hole_cards=[("A","h"), ("K","h")],
    board=[],
    pot=15,
    to_call=10,
    min_raise=20,
    max_raise=500,
    legal_actions=[
        {"type": "fold"},
        {"type": "call"},
        {"type": "raise", "min": 20, "max": 500},
    ],
    stacks={"P0": 500, "P1": 495, "P2": 490, "Hero": 500, "P4": 500, "P5": 500},
    opponents=["P0", "P1", "P2", "P4", "P5"],
    history=[],
)
order_state, order_hero = _build_game_state_from_view(order_view)
expected_order = [3, 4, 5, 0, 1, 2]
print(f"  seat_order={order_state.seat_order}, action_idx={order_state.action_idx}")
if order_state.seat_order[:len(expected_order)] == expected_order and order_state.action_idx == 0:
    print("  [PASS] — inference search starts at the hero decision root")
else:
    print(f"  [FAIL] — expected order prefix {expected_order}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 15 — Gate 2A: Legal action consistency
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 15: Legal action consistency")
print("=" * 60)

legal_state = _GameState(
    pot=100,
    stacks=[200, 200],
    committed_per_seat=[20, 50],
    alive=[True, True],
    street="turn",
    board=[("A","h"), ("K","d"), ("2","c"), ("J","s")],
    hole_cards={0: (("Q","h"), ("Q","s")), 1: (("T","c"), ("9","c"))},
    seat_order=[0],
    action_idx=0,
    real_contributions=True,
    big_blind=10,
)
concrete_legal = [
    {"type": "fold"},
    {"type": "call"},
    {"type": "raise", "min": 60, "max": 220},
]
# Both entry points must be given the same betting context (the production
# act() and _GameState paths always supply it).  Here current_bet=50 (seat 1's
# committed) and to_call=30 (50 - hero's 20).
from_state = legal_state.legal_abstract_actions()
from_view = _legal_abstract_actions(concrete_legal, 100, to_call=30,
                                    current_bet=50)
print(f"  GameState legal: {from_state}")
print(f"  View legal:      {from_view}")
if from_state == from_view:
    print("  [PASS] — GameState legality matches top-level helper")
else:
    print("  [FAIL] — GameState legality drifted from top-level helper")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 16 — Gate 2A: Opp-stat tracker live update
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 16: Opp-stat tracker live update")
print("=" * 60)

stats_bot = CFRBot(iterations=0, inference_mode=True)
for hand_i in range(10):
    v = PlayerView(
        me="hero",
        street="preflop",
        position="BTN",
        hole_cards=[("A","h"), ("Q","h")],
        board=[],
        pot=35,
        to_call=20,
        min_raise=40,
        max_raise=500,
        legal_actions=[
            {"type": "fold"},
            {"type": "call"},
            {"type": "raise", "min": 40, "max": 500},
        ],
        stacks={"hero": 500, "villain": 480},
        opponents=["villain"],
        history=[{
            "street": "preflop",
            "pid": "villain",
            "type": "raise",
            "amount": 30,
            "to_call_before": 0,
            "pot_before": 15 + hand_i,
        }],
    )
    stats_bot.act(v)
bucket = stats_bot._opp_stats.bucket(1)
print(f"  villain bucket after repeated raises: {bucket}")
if bucket == "LA":
    print("  [PASS] — tracker bucket updates from live observations")
else:
    print("  [FAIL] — bucket stayed default or wrong")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 17 — Gate 2A: Search latency at production depth
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 17: Search latency at production depth")
print("=" * 60)

lat_bot = CFRBot(inference_mode=True, search_depth=3)
lat_view = PlayerView(
    me="hero",
    street="flop",
    position="BTN",
    hole_cards=[("A","h"), ("A","s")],
    board=[("K","d"), ("7","c"), ("2","h")],
    pot=100,
    to_call=20,
    min_raise=40,
    max_raise=500,
    legal_actions=[
        {"type": "fold"},
        {"type": "call"},
        {"type": "raise", "min": 40, "max": 500},
    ],
    stacks={"hero": 500, "opp1": 500, "opp2": 500},
    opponents=["opp1", "opp2"],
    history=[],
)
lat_legal = _legal_abstract_actions(lat_view.legal_actions, lat_view.pot)
lat_prior = [0.0] * len(ABSTRACT_ACTIONS)
for a in lat_legal:
    lat_prior[a] = 1.0 / len(lat_legal)
lat_gs, lat_hero = _build_game_state_from_view(lat_view)
after_call = lat_gs.apply_action(lat_hero, 1)
if not after_call.is_chance_node():
    opp_key = lat_bot._info_key_for_state(after_call, after_call.seat_to_act())
    opp_node = lat_bot._get_node(opp_key)
    opp_node.strategy_sum = [0.2, 5.0, 1.0, 0.5, 0.5, 0.2, 0.2, 0.1]

leaf_counter = {"n": 0}
orig_leaf = lat_bot._leaf_value
def _counting_leaf(gs, hero_seat):
    leaf_counter["n"] += 1
    return orig_leaf(gs, hero_seat)
lat_bot._leaf_value = _counting_leaf
times = []
for _ in range(10):
    t0 = _time_mod.monotonic()
    lat_bot._subgame_search(lat_view, lat_prior, lat_legal, depth=3)
    times.append(_time_mod.monotonic() - t0)
mean_latency = sum(times) / len(times)
max_latency = max(times)
print(f"  depth=3 mean={mean_latency:.4f}s max={max_latency:.4f}s "
      f"leaf_calls={leaf_counter['n']}")
if leaf_counter["n"] > 0 and mean_latency < 2.0:
    print("  [PASS] — depth-3 search meets production latency budget")
else:
    print("  [FAIL] — depth-3 search exceeded budget or short-circuited")
    PASS = False
if mean_latency >= 1.0:
    print("  [DIAGNOSTIC] depth-3 search is above the 1s soft edge")

print()


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 18 — Reopen semantics + cumulative-contribution invariant
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 18: Reopen-action semantics + total_committed_per_seat")
print("=" * 60)

# Build a 3-seat preflop _GameState mid-action: hero=P0, blinds posted,
# P0 is about to act first in seat_order [0, 1, 2].
reopen_gs = _GameState(
    pot=15,                              # SB=5 + BB=10
    stacks=[195, 190, 200],              # P0 paid 5, P1 paid 10, P2 unposted
    committed_per_seat=[5, 10, 0],       # per-street (preflop) contributions
    alive=[True, True, True],
    street="preflop",
    board=[],
    hole_cards={0: (("A","h"), ("A","s")),
                1: (("K","c"), ("K","d")),
                2: (("Q","h"), ("Q","s"))},
    seat_order=[0, 1, 2],
    action_idx=0,
    deck_remaining=[c for c in _FULL_DECK if c not in
        {("A","h"),("A","s"),("K","c"),("K","d"),("Q","h"),("Q","s")}],
    real_contributions=True,
    ring_order=[0, 1, 2],
    big_blind=10,
)

# Step 1: P0 raises (bet_50 → abstract_idx 3).
gs1 = reopen_gs.apply_action(0, 3)
expected1 = [1, 2]
if gs1.seat_order == expected1 and gs1.action_idx == 0:
    print(f"  [PASS] — P0 raise: seat_order={gs1.seat_order}, action_idx=0")
else:
    print(f"  [FAIL] — P0 raise: got seat_order={gs1.seat_order} "
          f"action_idx={gs1.action_idx}, expected seat_order={expected1} action_idx=0")
    PASS = False

# Step 2: P1 raises (bet_67 → abstract_idx 4).
gs2 = gs1.apply_action(1, 4)
expected2 = [2, 0]
if gs2.seat_order == expected2 and gs2.action_idx == 0:
    print(f"  [PASS] — P1 raise: seat_order={gs2.seat_order}, action_idx=0")
else:
    print(f"  [FAIL] — P1 raise: got seat_order={gs2.seat_order} "
          f"action_idx={gs2.action_idx}, expected seat_order={expected2}")
    PASS = False

# Step 3: P2 folds (abstract_idx 0). Seat_order doesn't rebuild; action_idx advances.
gs3 = gs2.apply_action(2, 0)
if not gs3.alive[2] and gs3.action_idx == 1 and gs3.seat_order == expected2:
    print(f"  [PASS] — P2 fold: alive[2]=False, action_idx=1, seat_order unchanged")
else:
    print(f"  [FAIL] — P2 fold: alive[2]={gs3.alive[2]} "
          f"action_idx={gs3.action_idx} seat_order={gs3.seat_order}")
    PASS = False

# Step 4: P0 calls (check_call → abstract_idx 1). Total contributions accumulate.
total_before_call = list(gs3.total_committed_per_seat)
gs4 = gs3.apply_action(0, 1)
if gs4.total_committed_per_seat[0] > total_before_call[0]:
    print(f"  [PASS] — P0 call accumulated into total: "
          f"{total_before_call} → {gs4.total_committed_per_seat}")
else:
    print(f"  [FAIL] — P0 call did not accumulate: "
          f"{total_before_call} → {gs4.total_committed_per_seat}")
    PASS = False

# Step 5: advance_street to flop. committed_per_seat resets, total preserved.
gs_flop = gs4.advance_street()
preflop_total = list(gs4.total_committed_per_seat)
if (gs_flop.committed_per_seat == [0, 0, 0]
        and gs_flop.total_committed_per_seat == preflop_total):
    print(f"  [PASS] — advance_street: per-street reset to {gs_flop.committed_per_seat}, "
          f"total preserved as {gs_flop.total_committed_per_seat}")
else:
    print(f"  [FAIL] — advance_street: committed_per_seat={gs_flop.committed_per_seat}, "
          f"total_committed_per_seat={gs_flop.total_committed_per_seat} "
          f"(expected total={preflop_total})")
    PASS = False

# Step 6: leaf-value invariant — sum(total_committed_per_seat) >= pot - 1.
leaf_sum = sum(gs_flop.total_committed_per_seat)
if leaf_sum >= gs_flop.pot - 1:
    print(f"  [PASS] — leaf invariant holds: sum(total)={leaf_sum} >= pot-1={gs_flop.pot - 1}")
else:
    print(f"  [FAIL] — leaf invariant violated: sum(total)={leaf_sum} < pot-1={gs_flop.pot - 1}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 19 — Per-instance RNG independence + dealer rotation fingerprint
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 19: Per-instance RNG + dealer rotation regression")
print("=" * 60)

import random as _rnd_mod
from run_tournament_stats import _advance_dealer as _advance_dealer_ts
from run_local_match import _advance_dealer as _advance_dealer_lm

# Step 1: bot-independence — two CFRBots in same process must have different RNG.
rng_bot_a = CFRBot(iterations=0, inference_mode=True)
rng_bot_b = CFRBot(iterations=0, inference_mode=True)
a_sample = rng_bot_a._rng.random()
b_sample = rng_bot_b._rng.random()
if a_sample != b_sample:
    print(f"  ✅ PASS — two CFRBots produce different RNG samples "
          f"({a_sample:.6f} vs {b_sample:.6f})")
else:
    print(f"  ❌ FAIL — bots collided on first sample (both {a_sample})")
    PASS = False

# Step 2: bot-seeding determinism — same global seed → same bot RNG.
_rnd_mod.seed(123)
seeded_a = CFRBot(iterations=0, inference_mode=True)
x = seeded_a._rng.random()
_rnd_mod.seed(123)
seeded_b = CFRBot(iterations=0, inference_mode=True)
y = seeded_b._rng.random()
if x == y:
    print(f"  ✅ PASS — same global seed → same bot RNG sample ({x:.6f})")
else:
    print(f"  ❌ FAIL — seed cascade broken: {x:.6f} vs {y:.6f}")
    PASS = False

# Step 3: dealer rotation fingerprint via the orchestrator helpers.
# Scenario: 6 seats, P5 busted from start → active_count = 5.
# Buggy fingerprint (with % 6): [0,1,2,3,4,0,0,1,2,3] — note double-0.
# Fixed fingerprint  (with % 5): [0,1,2,3,4,0,1,2,3,4] — clean rotation.
expected_fixed = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
active_count = 5

# Verify run_tournament_stats._advance_dealer
ts_di = 0
ts_seq = [ts_di % active_count]
for _ in range(9):
    ts_di = _advance_dealer_ts(ts_di, active_count)
    ts_seq.append(ts_di % active_count)
if ts_seq == expected_fixed:
    print(f"  ✅ PASS — run_tournament_stats _advance_dealer rotation = {ts_seq}")
else:
    print(f"  ❌ FAIL — run_tournament_stats got {ts_seq}, expected {expected_fixed}")
    PASS = False

# Verify run_local_match._advance_dealer
lm_di = 0
lm_seq = [lm_di % active_count]
for _ in range(9):
    lm_di = _advance_dealer_lm(lm_di, active_count)
    lm_seq.append(lm_di % active_count)
if lm_seq == expected_fixed:
    print(f"  ✅ PASS — run_local_match _advance_dealer rotation = {lm_seq}")
else:
    print(f"  ❌ FAIL — run_local_match got {lm_seq}, expected {expected_fixed}")
    PASS = False

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
