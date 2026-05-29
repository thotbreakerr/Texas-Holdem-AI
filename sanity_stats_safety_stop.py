"""
sanity_stats_safety_stop.py -- run_tournament_stats regression coverage.

Covers:
- survivor finalization at max-hands safety stops
- no multi-way position-1 ties for unfinished tournaments
- unseeded Table RNG variation across fresh tournament tables
"""
import random
import sys

sys.path.insert(0, ".")

from bots import parse_players
from core.engine import InProcessBot, Seat, Table
from run_tournament_stats import finalize_finish_order, run_silent_tournament


PASS = True


print("=" * 60)
print("STATS: safety-stop survivor ranking")
print("=" * 60)

seats = [
    Seat("P1", 120),
    Seat("P2", 0),
    Seat("P3", 80),
    Seat("P4", 30),
]
finish_order = [("P2", 4, 5, 0)]
finalize_finish_order(seats, finish_order, total_players=4, hand_count=10)
positions = {pid: pos for pid, pos, _, _ in finish_order}
print(f"  Finalized positions: {positions}")

expected = {"P1": 1, "P3": 2, "P4": 3, "P2": 4}
if positions == expected:
    print("  PASS -- partial safety stop chip-ranks survivors distinctly")
else:
    print(f"  FAIL -- expected {expected}")
    PASS = False

print()

print("=" * 60)
print("STATS: max-hands no four-way winner tie")
print("=" * 60)

specs = parse_players("random,random,random,random")
res = run_silent_tournament((specs, 80, 1, 2, 0, 1, 123, False))
pos_ones = [pid for pid, pos, _, _ in res["finish_order"] if pos == 1]
print(f"  finish_order={res['finish_order']}")
print(f"  position-1 players={pos_ones}")
if len(pos_ones) <= 1:
    print("  PASS -- no multi-way position-1 tie")
else:
    print("  FAIL -- multiple players were marked position 1")
    PASS = False

print()

print("=" * 60)
print("STATS: unseeded deck RNG variation")
print("=" * 60)


class CheckCallSpyBot:
    def __init__(self, seen):
        self.seen = seen

    def act(self, state):
        if state.street == "flop" and state.board and not self.seen:
            self.seen.append(tuple(state.board[0]))
        for typ in ("check", "call"):
            for action in state.legal_actions:
                if action["type"] == typ:
                    return {"type": typ}
        return {"type": state.legal_actions[0]["type"]}


def first_flop_card():
    seen = []
    seats = [Seat(f"P{i}", 100) for i in range(4)]
    bots = {s.player_id: InProcessBot(CheckCallSpyBot(seen)) for s in seats}
    Table(rng=random.Random()).play_hand(
        seats=seats,
        small_blind=1,
        big_blind=2,
        dealer_index=0,
        bot_for=bots,
    )
    return seen[0] if seen else None


flops = [first_flop_card() for _ in range(3)]
print(f"  first flop cards={flops}")
if len(set(flops)) >= 2:
    print("  PASS -- at least 2 of 3 unseeded tables saw different first flop cards")
else:
    print("  FAIL -- unseeded tables reused the same first flop card")
    PASS = False

print()
print("=" * 60)
if PASS:
    print("ALL CHECKS PASSED")
else:
    print("SOME CHECKS FAILED")
    sys.exit(1)
print("=" * 60)
