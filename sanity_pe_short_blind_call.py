"""
Regression for the short-all-in-blind CALL(amount=0) violation.

Bug (pre-fix): Poker-Engine's compute_legal_actions() runs to_call through
_match_target() — a player can never be forced to match a bet no live opponent
can cover, so a nominal short-all-in blind shrinks the effective to_call,
possibly to zero (only check is then legal, never call). The _PEAgent adapter
in core/pe_engine.py instead built the legacy legal-action list from the RAW
Observation.to_call (current_bet - street_committed), so in that corner it
offered "call"; the bot called, Poker-Engine rejected the action
(``[violation] ... Action(type=CALL, amount=0)``) and substituted a check.

This test plays a heads-up hand where the big blind can only post 3 of the
10-chip blind (all-in for less). The button/SB then faces a raw to_call of 5
but an effective to_call of 0. It asserts the adapter offers check (not call),
no violation is emitted, and chips are conserved.
"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.engine import Seat
from core.pe_engine import PokerEngineTable


class RecordingCallBot:
    """Calls whenever offered, else checks/folds; records offered decisions."""

    def __init__(self):
        self.decisions = []  # (street, to_call, tuple of legal types)

    def act(self, view):
        legal_types = tuple(a["type"] for a in view.legal_actions)
        self.decisions.append((view.street, view.to_call, legal_types))
        for t in ("call", "check", "fold"):
            if t in legal_types:
                return {"type": t}
        return {"type": legal_types[0]}


def main():
    hero = RecordingCallBot()
    shorty = RecordingCallBot()
    seats = [Seat(player_id="Hero", chips=1000), Seat(player_id="Shorty", chips=3)]
    bot_for = {"Hero": hero, "Shorty": shorty}

    table = PokerEngineTable()
    stderr_capture = io.StringIO()
    # Heads-up, dealer_index=0: Hero is button/SB (posts 5), Shorty posts the
    # big blind all-in for 3 of 10. Nominal current_bet stays 10, so Hero's
    # raw to_call is 5 — but no live opponent can match beyond Hero's own 5,
    # so the effective to_call is 0 and Poker-Engine forbids CALL.
    with contextlib.redirect_stderr(stderr_capture):
        net = table.play_hand(seats, small_blind=5, big_blind=10,
                              dealer_index=0, bot_for=bot_for)
    stderr_text = stderr_capture.getvalue()

    # 1. The engine must not have rejected any adapter action.
    assert "[violation]" not in stderr_text, (
        f"adapter produced an illegal action:\n{stderr_text}")

    # 2. Hero's preflop decision must show the effective price, not the
    #    nominal one: to_call == 0 and no "call" row offered.
    hero_preflop = [d for d in hero.decisions if d[0] == "preflop"]
    assert hero_preflop, f"Hero never got a preflop decision: {hero.decisions}"
    street, to_call, legal_types = hero_preflop[0]
    assert to_call == 0, (
        f"Hero's to_call should be capped to 0 vs a short all-in blind, "
        f"got {to_call} (legal: {legal_types})")
    assert "call" not in legal_types, (
        f"'call' must not be offered when the effective to_call is 0, "
        f"got legal: {legal_types}")
    assert "check" in legal_types, f"'check' missing from legal: {legal_types}"

    # 3. Chips conserved: net deltas sum to zero on a 1003-chip table.
    assert sum(net.values()) == 0, f"chips leaked: {net}"
    assert sum(s.chips for s in seats) == 1003, \
        f"stack total changed: {[(s.player_id, s.chips) for s in seats]}"

    print("PASS  sanity_pe_short_blind_call")
    print(f"  Hero preflop view: to_call={to_call}, legal={legal_types}")
    print(f"  net deltas: {net}")


if __name__ == "__main__":
    main()
