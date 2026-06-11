"""
sanity_cfr_allin_call_reconstruction.py — short all-in CALL history parity.

Phase 3.1 (2026-06-11). The engine used to record amount=None for every
call; downstream reconstruction (CFR contribution rebuild) substituted the
full to_call_before. For a short-stack call-for-less the actual paid amount
is min(stack, to_call), so the rebuilt contributions exceeded the public
pot, _reconstruct_contributions_from_view() flagged the view unreliable,
and _build_training_game_state() returned None — every MCCFR traversal at
that decision was silently skipped. Fixed by recording the chips ACTUALLY
paid for calls in engine history.

Sections (all on REAL engine hands, not synthetic histories):
  1. Engine history records the actual paid amount for a short all-in call
  2. Reconstructed contributions match the public pot (is_real=True)
  3. _build_training_game_state() builds a state (traversals not skipped)
  4. Full (non-short) call still records/reconstructs exactly
  5. Legacy amount=None histories still reconstruct via to_call_before
"""
import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bots.cfr_bot import CFRBot, _reconstruct_contributions_from_view
from core.engine import Table, Seat, InProcessBot

PASS = True


class RaiseToBot:
    """Raises to a fixed total once, then checks/calls."""
    def __init__(self, target):
        self.target = target
        self.done = False

    def act(self, state):
        legal = state.legal_actions
        if not self.done:
            for a in legal:
                if a["type"] == "raise":
                    self.done = True
                    amt = max(a["min"], min(a["max"], self.target))
                    return {"type": "raise", "amount": amt}
        for t in ("check", "call", "fold"):
            for a in legal:
                if a["type"] == t:
                    return {"type": t}
        return {"type": legal[0]["type"]}


class CallBot:
    def act(self, state):
        for t in ("call", "check", "fold"):
            for a in state.legal_actions:
                if a["type"] == t:
                    return {"type": t}
        return {"type": state.legal_actions[0]["type"]}


class RecordingFoldBot:
    """Captures every PlayerView it is offered, then folds/checks."""
    def __init__(self):
        self.views = []

    def act(self, state):
        self.views.append(state)
        for t in ("fold", "check", "call"):
            for a in state.legal_actions:
                if a["type"] == t:
                    return {"type": t}
        return {"type": state.legal_actions[0]["type"]}


def play_short_call_hand(short_stack: int, raise_to: int):
    """4 seats, dealer_index=0: A=BTN(short), B=SB, C=BB, D=UTG.

    Preflop order: D(UTG) raises to `raise_to`; A(BTN, short) calls — all-in
    for less when short_stack < raise_to; B(SB) is captured next, folds.
    Returns (SB's first PlayerView, net chip dict).
    """
    seats = [Seat("A", short_stack), Seat("B", 1000), Seat("C", 1000),
             Seat("D", 1000)]
    sb_recorder = RecordingFoldBot()
    bot_for = {
        "A": InProcessBot(CallBot()),
        "B": InProcessBot(sb_recorder),
        "C": InProcessBot(RecordingFoldBot()),
        "D": InProcessBot(RaiseToBot(raise_to)),
    }
    table = Table()
    with redirect_stdout(io.StringIO()):
        net = table.play_hand(seats=seats, small_blind=5, big_blind=10,
                              dealer_index=0, bot_for=bot_for)
    if not sb_recorder.views:
        raise RuntimeError("SB never received a decision — hand script broke")
    return sb_recorder.views[0], net


print("=" * 70)
print("sanity_cfr_allin_call_reconstruction.py — short all-in call parity")
print("=" * 70)
print()

# ═══════════════════════════════════════════════════════════════════════════════
#  Sections 1-3 — short all-in call (60 chips into a 200 raise)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 1: engine records the ACTUAL paid amount for a short call")
print("=" * 60)

view = None
try:
    view, net = play_short_call_hand(short_stack=60, raise_to=200)

    call_entries = [e for e in view.history if e.get("type") == "call"]
    print(f"  history at SB's decision: {len(view.history)} entries")
    for e in view.history:
        print(f"    {e}")

    if len(call_entries) == 1:
        print("  [PASS] — exactly one call entry (the short all-in call)")
    else:
        print(f"  [FAIL] — expected 1 call entry, got {len(call_entries)}")
        PASS = False

    if call_entries:
        amt = call_entries[0].get("amount")
        tcb = call_entries[0].get("to_call_before")
        # A posted nothing (BTN), had 60 behind, faced to_call 200.
        if amt == 60:
            print(f"  [PASS] — call amount records chips actually paid "
                  f"({amt}, not None / not to_call_before={tcb})")
        else:
            print(f"  [FAIL] — call amount={amt!r}; expected 60 "
                  f"(min(stack, to_call)); to_call_before={tcb}")
            PASS = False

    if sum(net.values()) == 0:
        print("  [PASS] — hand settled with chips conserved")
    else:
        print(f"  [FAIL] — chip conservation broken: net={net}")
        PASS = False

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()

print("=" * 60)
print("Section 2: reconstructed contributions equal the public pot")
print("=" * 60)

try:
    if view is None:
        raise RuntimeError("Section 1 captured no view")
    street_c, total_c, is_real = _reconstruct_contributions_from_view(view)
    print(f"  view.pot (public)        = {view.pot}")
    print(f"  reconstructed sum(total) = {sum(total_c)}  per-seat={total_c}")
    print(f"  is_real                  = {is_real}")

    if abs(sum(total_c) - int(view.pot)) <= 1:
        print("  [PASS] — pot/contribution parity holds")
    else:
        print(f"  [FAIL] — mismatch of {sum(total_c) - int(view.pot)} chips "
              f"(short call rebuilt as full to_call?)")
        PASS = False
    if is_real:
        print("  [PASS] — reconstruction marked reliable")
    else:
        print("  [FAIL] — reconstruction marked UNRELIABLE")
        PASS = False

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()

print("=" * 60)
print("Section 3: _build_training_game_state() does not skip the traversal")
print("=" * 60)

try:
    if view is None:
        raise RuntimeError("Section 1 captured no view")
    bot = CFRBot(iterations=1, profile_path=None, inference_mode=False)
    opp_hands = bot._sample_opponent_hands(view.hole_cards, view.board,
                                           len(view.opponents))
    gs = bot._build_training_game_state(
        view, hero_hole=view.hole_cards, opp_hands=opp_hands,
        board=view.board)

    if gs is not None:
        print(f"  [PASS] — training state built (pot={gs.pot}, "
              f"total_committed={gs.total_committed_per_seat})")
        if abs(sum(gs.total_committed_per_seat) - gs.pot) <= 1:
            print("  [PASS] — state pot matches per-seat contributions")
        else:
            print(f"  [FAIL] — state pot {gs.pot} != contributions "
                  f"{sum(gs.total_committed_per_seat)}")
            PASS = False
    else:
        print("  [FAIL] — _build_training_game_state returned None: "
              "MCCFR traversals at this decision are silently skipped")
        PASS = False

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()

# ═══════════════════════════════════════════════════════════════════════════════
#  Section 4 — full (non-short) call still records / reconstructs exactly
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 4: full call regression (deep stack calls the raise)")
print("=" * 60)

try:
    view4, _net4 = play_short_call_hand(short_stack=1000, raise_to=200)
    call_entries = [e for e in view4.history if e.get("type") == "call"]
    if call_entries and call_entries[0].get("amount") == 200:
        print(f"  [PASS] — full call records its full paid amount (200)")
    else:
        amt = call_entries[0].get("amount") if call_entries else "<none>"
        print(f"  [FAIL] — full call amount={amt!r}, expected 200")
        PASS = False

    _s, total4, real4 = _reconstruct_contributions_from_view(view4)
    if real4 and abs(sum(total4) - int(view4.pot)) <= 1:
        print(f"  [PASS] — full-call reconstruction parity "
              f"(pot={view4.pot}, sum={sum(total4)})")
    else:
        print(f"  [FAIL] — full-call reconstruction broke: pot={view4.pot}, "
              f"sum={sum(total4)}, is_real={real4}")
        PASS = False

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()

# ═══════════════════════════════════════════════════════════════════════════════
#  Section 5 — legacy amount=None histories keep the to_call_before fallback
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 5: legacy amount=None call entries still reconstruct")
print("=" * 60)

try:
    # A deep stack calling 200 recorded legacy-style (amount=None) must
    # still rebuild as 200 paid via the to_call_before fallback.
    class _ViewShim:
        """view4 (full-call hand) with its call amounts stripped to None."""
    view4, _ = play_short_call_hand(short_stack=1000, raise_to=200)
    shim = _ViewShim()
    for attr in ("me", "street", "position", "hole_cards", "board", "pot",
                 "to_call", "min_raise", "max_raise", "legal_actions",
                 "stacks", "opponents", "seat_indices"):
        setattr(shim, attr, getattr(view4, attr, None))
    shim.history = [
        {**e, "amount": None} if e.get("type") == "call" else e
        for e in view4.history
    ]

    _s5, total5, real5 = _reconstruct_contributions_from_view(shim)
    if real5 and abs(sum(total5) - int(view4.pot)) <= 1:
        print(f"  [PASS] — legacy None-amount full call falls back to "
              f"to_call_before (pot={view4.pot}, sum={sum(total5)})")
    else:
        print(f"  [FAIL] — legacy fallback broke: pot={view4.pot}, "
              f"sum={sum(total5)}, is_real={real5}")
        PASS = False

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()

# ═══════════════════════════════════════════════════════════════════════════════
#  OVERALL
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
if PASS:
    print("OVERALL: ALL CHECKS PASSED")
else:
    print("OVERALL: SOME CHECKS FAILED")
print("=" * 60)
sys.exit(0 if PASS else 1)
