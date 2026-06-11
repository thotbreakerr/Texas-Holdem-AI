"""
sanity_cfr_token_parity.py — tree/live history-token round-trip parity.

Phase 3 (2026-06-10). _GameState.apply_action must append the SAME token
that live play would produce when core.action_history.tokenize() sees the
realized engine history entry (amount = actor's resulting total street
contribution, pot_before = pot before the action's chips).

The old code appended a fixed token per sizing bucket (bet_33 -> "S"...),
which diverged from realized-ratio tokenization for raises, min-raise
clamps, and all-in clamps — orphaning tree-trained info sets at inference.

Covers: opening bets (each bucket), raises facing a bet, min-raise clamp,
sizing clamped to all-in, short-stack all-in-for-less call, fold/check/call
tokens, plus a randomized property sweep.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bots.cfr_bot import _GameState
from core.action_history import ActionEvent, sizing_token, tokenize

BOARD = [("2", "h"), ("7", "d"), ("9", "s")]
HOLES = {0: (("A", "s"), ("A", "d")), 1: (("K", "s"), ("K", "d"))}

PASS = True
QUIET = False  # section 6 sets this: only failures are printed


def check(label, cond, detail=""):
    global PASS
    if cond:
        if not QUIET:
            print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label}  {detail}")
        PASS = False


def make_state(pot, stacks, committed, last_raise_size=None, street="flop",
               seat_order=None):
    return _GameState(
        pot=pot, stacks=list(stacks), committed_per_seat=list(committed),
        alive=[True] * len(stacks), street=street, board=list(BOARD),
        hole_cards=dict(HOLES),
        seat_order=list(seat_order) if seat_order else [0, 1],
        action_idx=0, history_tokens="", deck_remaining=[], hero_seat=0,
        real_contributions=True, ring_order=list(range(len(stacks))),
        position_labels={0: "BTN", 1: "BB"}, big_blind=10,
        last_raise_size=last_raise_size,
    )


def assert_parity(state, seat, abstract_idx, label):
    """Apply a tree action; compare its token to canonical tokenization."""
    nxt = state.apply_action(seat, abstract_idx)
    appended = nxt.history_tokens[len(state.history_tokens):]
    if appended == "":
        return  # action was a no-op (not legal); nothing to compare

    if abstract_idx == 0:
        check(label, appended == "F", f"got {appended!r}, want 'F'")
        return
    if abstract_idx == 1:
        want = "K" if state.to_call_for(seat) == 0 else "C"
        check(label, appended == want, f"got {appended!r}, want {want!r}")
        return

    # Sizing action: live play tokenizes the realized engine entry.
    realized_total = nxt.committed_per_seat[seat]
    event = ActionEvent(seat=seat, street=state.street, action="raise",
                        amount=realized_total, pot_before=state.pot)
    live = tokenize([event])
    helper = sizing_token(realized_total, state.pot)
    check(label, appended == live == helper,
          f"tree={appended!r} live={live!r} helper={helper!r} "
          f"(total={realized_total}, pot_before={state.pot})")


def run() -> bool:
    random.seed(42)

    print("Section 1: opening bets, deep stacks (pot 100)")
    for idx in (2, 3, 4, 5, 6, 7):
        st = make_state(pot=100, stacks=[1000, 1000], committed=[0, 0])
        assert_parity(st, 0, idx, f"open bucket {idx}")

    print("Section 2: raises facing a bet (pot 150, bet 50 outstanding)")
    # Seat 1 already bet 50 this street; seat 0 raises.
    for idx in (2, 3, 4, 5, 6, 7):
        st = make_state(pot=150, stacks=[1000, 950], committed=[0, 50],
                        last_raise_size=50)
        assert_parity(st, 0, idx, f"raise bucket {idx}")

    print("Section 3: min-raise clamp (large last raise forces clamp up)")
    for idx in (2, 3, 4):
        st = make_state(pot=500, stacks=[2000, 1700], committed=[0, 300],
                        last_raise_size=300)
        assert_parity(st, 0, idx, f"clamped raise bucket {idx}")

    print("Section 4: sizing clamped down to all-in (short stack)")
    for idx in (2, 6, 7):
        # Stack 120 facing bet 100 into pot 300: every bucket hits max.
        st = make_state(pot=300, stacks=[120, 900], committed=[0, 100],
                        last_raise_size=100)
        assert_parity(st, 0, idx, f"all-in-clamped bucket {idx}")

    print("Section 5: short-stack all-in-for-less call + fold/check/call")
    st = make_state(pot=300, stacks=[40, 900], committed=[0, 100],
                    last_raise_size=100)
    assert_parity(st, 0, 1, "all-in-for-less call -> 'C'")
    assert_parity(st, 0, 0, "fold -> 'F'")
    st2 = make_state(pot=100, stacks=[500, 500], committed=[0, 0])
    assert_parity(st2, 0, 1, "free check -> 'K'")

    print("Section 6: randomized property sweep (200 spots, failures only)")
    global QUIET
    QUIET = True
    n_checked = 0
    for trial in range(200):
        pot = random.randint(10, 500)
        opp_bet = random.choice([0, 0, random.randint(10, pot)])
        stacks = [random.randint(20, 1500), random.randint(20, 1500)]
        st = make_state(pot=pot + opp_bet, stacks=stacks,
                        committed=[0, opp_bet],
                        last_raise_size=max(10, opp_bet) if opp_bet else None)
        for idx in st.legal_abstract_actions():
            assert_parity(st, 0, idx, f"random spot {trial} action {idx}")
            n_checked += 1
    QUIET = False
    print(f"  swept {n_checked} (spot, action) pairs")

    print("\nOVERALL:", "ALL CHECKS PASSED" if PASS else "SOME CHECKS FAILED")
    return PASS


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
