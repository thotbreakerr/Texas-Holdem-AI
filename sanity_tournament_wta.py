"""Phase 2 gate: WTA tournament accounting and default-off antes."""

from __future__ import annotations

import random
import sys

sys.path.insert(0, ".")

from core.bot_api import Action
from core.engine import InProcessBot, Seat, Table
from core.tournament import run_tournament
from run_eval import aggregate_results


class FoldBot:
    def act(self, view):
        for action_type in ("fold", "check", "call"):
            if any(a["type"] == action_type for a in view.legal_actions):
                return Action(action_type)
        return Action(view.legal_actions[0]["type"])


class CheckCallBot:
    def act(self, view):
        for action_type in ("check", "call", "fold"):
            if any(a["type"] == action_type for a in view.legal_actions):
                return Action(action_type)
        return Action(view.legal_actions[0]["type"])


def _wrapped(bot):
    return InProcessBot(bot)


def _play_fold_hand(ante_arg=None):
    table = Table(rng=random.Random(77))
    seats = [Seat("BTN", 100), Seat("SB", 100), Seat("BB", 100)]
    bots = {seat.player_id: _wrapped(FoldBot()) for seat in seats}
    kwargs = {}
    if ante_arg is not None:
        kwargs["ante"] = ante_arg
    net = table.play_hand(
        seats=seats,
        small_blind=1,
        big_blind=2,
        dealer_index=0,
        bot_for=bots,
        **kwargs,
    )
    return net, {seat.player_id: seat.chips for seat in seats}


def _check(label, condition, details=""):
    print(f"[{label}] {'PASS' if condition else 'FAIL'}")
    if not condition and details:
        print(f"  {details}")
    return condition


def run():
    passed = True

    no_ante = _play_fold_hand()
    explicit_zero = _play_fold_hand(0)
    passed &= _check(
        "ante=0 parity",
        no_ante == explicit_zero,
        f"omitted={no_ante} explicit_zero={explicit_zero}",
    )

    net, chips = _play_fold_hand(5)
    expected_chips = {"BTN": 95, "SB": 94, "BB": 111}
    passed &= _check(
        "fixed ante collection",
        chips == expected_chips,
        f"chips={chips} expected={expected_chips}",
    )
    passed &= _check(
        "ante chip conservation",
        sum(net.values()) == 0 and sum(chips.values()) == 300,
        f"net={net} chips={chips}",
    )

    ante_events: list[int] = []
    schedule_result = run_tournament(
        [Seat("A", 1000), Seat("B", 1000)],
        {"A": _wrapped(CheckCallBot()), "B": _wrapped(CheckCallBot())},
        small_blind=10,
        big_blind=20,
        blind_increase_every=1,
        max_hands=3,
        dealer_rotation="full_table",
        winner_resolution="chip_count_on_max_hands",
        rng=random.Random(123),
        ante_schedule=lambda _hand, _sb, bb: int(bb * 0.1),
        on_event=lambda event: (
            ante_events.append(event["ante"])
            if event["type"] == "hand_start"
            else None
        ),
        suppress_output=True,
    )
    passed &= _check(
        "scheduled antes follow blind escalation",
        ante_events == [2, 3, 4],
        f"antes={ante_events}",
    )
    schedule_final_chips = schedule_result["final_chips"]
    schedule_chip_leader = max(schedule_final_chips, key=schedule_final_chips.get)
    passed &= _check(
        "max-hands WTA winner resolved",
        len(set(schedule_final_chips.values())) == len(schedule_final_chips)
        and schedule_result["winner"] == schedule_chip_leader,
        f"result={schedule_result}",
    )

    result = run_tournament(
        [Seat("P1", 120), Seat("P2", 120), Seat("P3", 120)],
        {
            "P1": _wrapped(CheckCallBot()),
            "P2": _wrapped(CheckCallBot()),
            "P3": _wrapped(CheckCallBot()),
        },
        small_blind=5,
        big_blind=10,
        blind_increase_every=50,
        max_hands=20,
        dealer_rotation="full_table",
        winner_resolution="chip_count_on_max_hands",
        rng=random.Random(20260614),
        ante=3,
        suppress_output=True,
    )
    player_specs = [(pid, "check_call") for pid in ("P1", "P2", "P3")]
    aggregated = aggregate_results([result], player_specs)
    winner = result["winner"]
    passed &= _check(
        "first-place winner is counted as WTA win_rate",
        winner is not None
        and aggregated["summary"][winner]["wins"] == 1
        and aggregated["summary"][winner]["win_rate"] == 1.0
        and all(
            row["wins"] == 0
            for pid, row in aggregated["summary"].items()
            if pid != winner
        ),
        f"winner={winner} summary={aggregated['summary']}",
    )
    passed &= _check(
        "tournament chips conserve with antes",
        result["hand_count"] > 1 and sum(result["final_chips"].values()) == 360,
        f"hand_count={result['hand_count']} final_chips={result['final_chips']}",
    )

    print("=" * 60)
    print(
        "OVERALL: "
        f"{'ALL CHECKS PASSED [PASS]' if passed else 'SOME CHECKS FAILED [FAIL]'}"
    )
    return passed


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
