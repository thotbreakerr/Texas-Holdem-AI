"""Phase 1 gate: canonical tournament loop matches the legacy loops."""

from __future__ import annotations

import io
import random
import sys
from contextlib import redirect_stdout

sys.path.insert(0, ".")

from core.bot_api import Action
from core.engine import Seat, Table
from core.table_order import advance_dealer_seat_index, normalize_dealer_seat_index
from core.tournament import run_tournament
from bots import escalate_blinds
from run_tournament_stats import finalize_finish_order


class JamBot:
    """Deterministic legal bot that applies pressure and forces eliminations."""

    def act(self, view):
        for action_type in ("raise", "bet"):
            choices = [a for a in view.legal_actions if a["type"] == action_type]
            if choices:
                return Action(action_type, choices[0]["max"])
        for action_type in ("call", "check", "fold"):
            if any(a["type"] == action_type for a in view.legal_actions):
                return Action(action_type)
        return Action(view.legal_actions[0]["type"])


class CheckCallBot:
    """Deterministic legal bot that reaches max-hands tiebreaks."""

    def act(self, view):
        for action_type in ("check", "call", "fold"):
            if any(a["type"] == action_type for a in view.legal_actions):
                return Action(action_type)
        return Action(view.legal_actions[0]["type"])


def _make_bots(player_ids, bot_cls):
    return {pid: bot_cls() for pid in player_ids}


def _result_subset(result):
    return {
        "winner": result["winner"],
        "hand_count": result["hand_count"],
        "finish_order": result["finish_order"],
        "final_chips": result["final_chips"],
        "chip_swing": result["chip_swing"],
    }


def _legacy_full_table(player_ids, bot_cls, chips, base_sb, base_bb,
                       blind_increase_every, max_hands, seed):
    if seed is not None:
        random.seed(seed)

    bots = _make_bots(player_ids, bot_cls)
    seats = [Seat(player_id=pid, chips=chips) for pid in player_ids]
    table = Table(rng=random.Random(seed) if seed is not None else random.Random())
    dealer_index = 0
    hand_count = 0
    total_players = len(seats)
    finish_order: list[tuple[str, int, int, int]] = []

    with redirect_stdout(io.StringIO()):
        while True:
            active_seats = [s for s in seats if s.chips > 0]
            if len(active_seats) <= 1:
                break

            hand_count += 1
            sb, bb = escalate_blinds(
                hand_count, base_sb, base_bb, blind_increase_every
            )
            active_bots = {s.player_id: bots[s.player_id] for s in active_seats}
            dealer_index = normalize_dealer_seat_index(seats, dealer_index)
            if dealer_index is None:
                break

            table.play_hand(
                seats=seats,
                small_blind=sb,
                big_blind=bb,
                dealer_index=dealer_index,
                bot_for=active_bots,
                on_event=None,
                log_decisions=False,
            )
            next_dealer = advance_dealer_seat_index(seats, dealer_index)
            if next_dealer is not None:
                dealer_index = next_dealer

            for seat in seats:
                already_ranked = any(e[0] == seat.player_id for e in finish_order)
                if seat.chips <= 0 and not already_ranked:
                    pos = total_players - len(finish_order)
                    finish_order.append((seat.player_id, pos, hand_count, 0))

            if hand_count >= max_hands:
                break

    finalize_finish_order(seats, finish_order, total_players, hand_count)
    final_chips = {seat.player_id: seat.chips for seat in seats}
    winner = None
    for pid, pos, _, _ in finish_order:
        if pos == 1:
            winner = pid
            break
    if winner is None and final_chips:
        winner = max(final_chips, key=lambda pid: final_chips[pid])

    chip_swing = None
    if total_players == 2 and winner:
        loser = next(pid for pid in player_ids if pid != winner)
        chip_swing = final_chips[winner] - final_chips[loser]

    return {
        "winner": winner,
        "hand_count": hand_count,
        "finish_order": finish_order,
        "final_chips": final_chips,
        "chip_swing": chip_swing,
    }


def _new_full_table(player_ids, bot_cls, chips, base_sb, base_bb,
                    blind_increase_every, max_hands, seed):
    if seed is not None:
        random.seed(seed)

    seats = [Seat(player_id=pid, chips=chips) for pid in player_ids]
    return _result_subset(run_tournament(
        seats,
        _make_bots(player_ids, bot_cls),
        small_blind=base_sb,
        big_blind=base_bb,
        blind_increase_every=blind_increase_every,
        max_hands=max_hands,
        dealer_index=0,
        dealer_rotation="full_table",
        winner_resolution="chip_count_on_max_hands",
        rng=random.Random(seed) if seed is not None else random.Random(),
        suppress_output=True,
        log_decisions=False,
    ))


def _legacy_active_circle(player_ids, bot_cls, chips, base_sb, base_bb,
                          blind_increase_every, seed):
    bots = _make_bots(player_ids, bot_cls)
    seats = [Seat(player_id=pid, chips=chips) for pid in player_ids]
    active_seats = list(seats)
    table = Table(rng=random.Random(seed) if seed is not None else random.Random())
    dealer = 0
    hand_count = 0
    total_players = len(seats)
    finish_order: list[tuple[str, int, int, int]] = []

    with redirect_stdout(io.StringIO()):
        while len(active_seats) > 1:
            hand_count += 1
            sb, bb = escalate_blinds(
                hand_count, base_sb, base_bb, blind_increase_every
            )
            dealer_i = dealer % len(active_seats)
            active_bots = {s.player_id: bots[s.player_id] for s in active_seats}
            table.play_hand(active_seats, sb, bb, dealer_i, active_bots)

            eliminated = [seat for seat in active_seats if seat.chips <= 0]
            for seat in eliminated:
                pos = total_players - len(finish_order)
                finish_order.append((seat.player_id, pos, hand_count, 0))
                active_seats.remove(seat)

            dealer = (dealer + 1) % max(len(active_seats), 1)

    finalize_finish_order(seats, finish_order, total_players, hand_count)
    final_chips = {seat.player_id: seat.chips for seat in seats}
    winner = None
    for pid, pos, _, _ in finish_order:
        if pos == 1:
            winner = pid
            break
    chip_swing = None
    if total_players == 2 and winner:
        loser = next(pid for pid in player_ids if pid != winner)
        chip_swing = final_chips[winner] - final_chips[loser]
    return {
        "winner": winner,
        "hand_count": hand_count,
        "finish_order": finish_order,
        "final_chips": final_chips,
        "chip_swing": chip_swing,
    }


def _new_active_circle(player_ids, bot_cls, chips, base_sb, base_bb,
                       blind_increase_every, seed):
    seats = [Seat(player_id=pid, chips=chips) for pid in player_ids]
    return _result_subset(run_tournament(
        seats,
        _make_bots(player_ids, bot_cls),
        small_blind=base_sb,
        big_blind=base_bb,
        blind_increase_every=blind_increase_every,
        max_hands=None,
        dealer_index=0,
        dealer_rotation="active_circle",
        winner_resolution="finish_order",
        rng=random.Random(seed) if seed is not None else random.Random(),
        suppress_output=True,
    ))


def _check(label, actual, expected):
    ok = actual == expected
    print(f"[{label}] {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"  expected: {expected}")
        print(f"  actual:   {actual}")
    return ok


def run():
    passed = True
    seed_grid = [7, 101, 20260614]
    player_counts = [2, 3, 6]

    for count in player_counts:
        pids = [f"P{i}" for i in range(1, count + 1)]
        for seed in seed_grid:
            expected = _legacy_full_table(
                pids, JamBot, 120, 5, 10, 3, 20, seed
            )
            actual = _new_full_table(
                pids, JamBot, 120, 5, 10, 3, 20, seed
            )
            passed &= _check(
                f"full-table jam count={count} seed={seed}",
                actual,
                expected,
            )

    for seed in seed_grid:
        pids = ["A", "B"]
        expected = _legacy_full_table(
            pids, CheckCallBot, 500, 5, 10, 10_000, 1, seed
        )
        actual = _new_full_table(
            pids, CheckCallBot, 500, 5, 10, 10_000, 1, seed
        )
        passed &= _check(
            f"full-table max-hands seed={seed}",
            actual,
            expected,
        )

    for count in player_counts:
        pids = [f"P{i}" for i in range(1, count + 1)]
        for seed in seed_grid:
            expected = _legacy_active_circle(
                pids, JamBot, 120, 5, 10, 3, seed
            )
            actual = _new_active_circle(
                pids, JamBot, 120, 5, 10, 3, seed
            )
            passed &= _check(
                f"active-circle jam count={count} seed={seed}",
                actual,
                expected,
            )

    print("=" * 60)
    print(
        "OVERALL: "
        f"{'ALL CHECKS PASSED [PASS]' if passed else 'SOME CHECKS FAILED [FAIL]'}"
    )
    return passed


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
