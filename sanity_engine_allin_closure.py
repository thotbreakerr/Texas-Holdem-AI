"""
Regression for the all-in betting-round closure bug.

Bug (pre-fix): all_live_equal() returned True as soon as len(live) <= 1, so when
one player shoved all-in heads-up the lone live opponent was never asked to act —
they were dragged to showdown without ever calling/folding the shove.

This test plays a heads-up hand where the button/SB shoves all-in preflop and
asserts the big blind actually RECEIVES a call/fold decision against the shove,
and that chips are conserved.
"""
import sys

sys.path.insert(0, "/Users/jaroslavaupart/Desktop/Projects/Texas-Holdem-AI")

from core.engine import Table, Seat, InProcessBot


class ShoveBot:
    """Shoves all-in at the first preflop opportunity, else checks/calls."""
    def act(self, state):
        legal = state.legal_actions
        street = state.street
        if street == "preflop":
            for a in legal:
                if a["type"] == "raise":
                    return {"type": "raise", "amount": a["max"]}
        for t in ("check", "call", "fold"):
            for a in legal:
                if a["type"] == t:
                    return {"type": t}
        return {"type": legal[0]["type"]}


class RecordingCallBot:
    """Calls everything, and records the decisions it was actually offered."""
    def __init__(self):
        self.decisions = []  # list of (street, to_call, legal_types)

    def act(self, state):
        legal = state.legal_actions
        self.decisions.append(
            (state.street, state.to_call, {a["type"] for a in legal})
        )
        for t in ("call", "check", "fold"):
            for a in legal:
                if a["type"] == t:
                    return {"type": t}
        return {"type": legal[0]["type"]}


def run_multiway():
    """3-handed: two all-ins at different stack depths, lone live caller.

    Exercises the new closure path together with multi-level side-pot
    construction — A shoves 200, B shoves 500, C (1000) is the lone live player
    who must be offered the call/fold decision; then chips must conserve through
    _showdown_and_settle's side pots.
    """
    table = Table()
    seats = [Seat("A", 200), Seat("B", 500), Seat("C", 1000)]
    c_bot = RecordingCallBot()
    bot_for = {
        "A": InProcessBot(ShoveBot()),
        "B": InProcessBot(ShoveBot()),
        "C": InProcessBot(c_bot),
    }
    net = table.play_hand(
        seats=seats, small_blind=10, big_blind=20,
        dealer_index=0, bot_for=bot_for,
    )

    PASS = True
    faced = any(to_call > 0 and {"fold", "call"} <= types
                for _street, to_call, types in c_bot.decisions)
    if faced:
        print("[CHECK 4] PASS — lone live player C faced the all-in decision")
    else:
        PASS = False
        print(f"[CHECK 4] FAIL — C never offered call/fold; {c_bot.decisions}")

    if sum(net.values()) == 0:
        print(f"[CHECK 5] PASS — chips conserved across side pots (net={net})")
    else:
        PASS = False
        print(f"[CHECK 5] FAIL — side-pot settlement not conserved: net={net}")

    chips_total = sum(s.chips for s in seats)
    if chips_total == 1700:
        print(f"[CHECK 6] PASS — table chips intact ({chips_total})")
    else:
        PASS = False
        print(f"[CHECK 6] FAIL — table chips drifted: {chips_total}")
    return PASS


def run():
    table = Table()
    seats = [Seat("BTN", 1000), Seat("BB", 1000)]
    bb_bot = RecordingCallBot()
    bot_for = {
        "BTN": InProcessBot(ShoveBot()),
        "BB": InProcessBot(bb_bot),
    }

    net = table.play_hand(
        seats=seats,
        small_blind=10,
        big_blind=20,
        dealer_index=0,
        bot_for=bot_for,
    )

    PASS = True

    # CHECK 1: BB was offered a preflop decision facing the all-in (to_call>0
    # with both fold and call available).
    faced_shove = any(
        street == "preflop" and to_call > 0 and {"fold", "call"} <= types
        for street, to_call, types in bb_bot.decisions
    )
    if faced_shove:
        print("[CHECK 1] PASS — BB faced the all-in with fold/call offered")
    else:
        PASS = False
        print(f"[CHECK 1] FAIL — BB never got a call/fold decision vs the shove; "
              f"decisions={bb_bot.decisions}")

    # CHECK 2: chips are conserved (sum of net deltas == 0).
    total = sum(net.values())
    if total == 0:
        print(f"[CHECK 2] PASS — chips conserved (net={net})")
    else:
        PASS = False
        print(f"[CHECK 2] FAIL — chips not conserved: net={net} sum={total}")

    # CHECK 3: total chips on the table still 2000.
    chips_total = sum(s.chips for s in seats)
    if chips_total == 2000:
        print(f"[CHECK 3] PASS — table chips intact ({chips_total})")
    else:
        PASS = False
        print(f"[CHECK 3] FAIL — table chips drifted: {chips_total}")

    PASS = run_multiway() and PASS

    print("=" * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED [PASS]' if PASS else 'SOME CHECKS FAILED [FAIL]'}")
    return PASS


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
