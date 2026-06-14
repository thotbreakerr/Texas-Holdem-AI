"""Canonical tournament loop shared by eval, UI, and engine wrappers."""

from __future__ import annotations

import io
import random
from contextlib import nullcontext, redirect_stdout
from typing import Any, Callable, Literal

from bots import escalate_blinds
from core.engine import Seat, Table
from core.table_order import advance_dealer_seat_index, normalize_dealer_seat_index
from run_tournament_stats import finalize_finish_order


DealerRotation = Literal["full_table", "active_circle"]
WinnerResolution = Literal["finish_order", "chip_count_on_max_hands"]
TournamentEvent = dict[str, Any]
TournamentCallback = Callable[[TournamentEvent], None]
CancelCheck = Callable[[], bool]
PauseWait = Callable[[], None]


def _coerce_seats(seats: list[Seat | dict[str, Any]], chips: int | None) -> list[Seat]:
    """Return mutable Seat instances, optionally resetting their stack size."""
    result = [s if isinstance(s, Seat) else Seat(**s) for s in seats]
    if chips is not None:
        for seat in result:
            seat.chips = chips
    return result


def _active_seats(seats: list[Seat]) -> list[Seat]:
    return [s for s in seats if s.chips > 0 and not s.is_sitting_out]


def _emit(callback: TournamentCallback | None, event: TournamentEvent) -> None:
    if callback is not None:
        callback(event)


def _winner_from_finish_order(
    finish_order: list[tuple[str, int, int, int]],
) -> str | None:
    for pid, pos, _, _ in finish_order:
        if pos == 1:
            return pid
    return None


def run_tournament(
    seats: list[Seat | dict[str, Any]],
    bot_for: dict[str, Any],
    *,
    small_blind: int,
    big_blind: int,
    blind_increase_every: int = 50,
    max_hands: int | None = 10_000,
    chips: int | None = None,
    dealer_index: int = 0,
    dealer_rotation: DealerRotation = "full_table",
    winner_resolution: WinnerResolution = "chip_count_on_max_hands",
    rng: random.Random | None = None,
    table: Table | None = None,
    on_event: TournamentCallback | None = None,
    hand_delay: Callable[[], None] | None = None,
    should_cancel: CancelCheck | None = None,
    wait_if_paused: PauseWait | None = None,
    suppress_output: bool = False,
    log_decisions: bool = False,
) -> dict[str, Any]:
    """Run a last-bot-standing tournament and return eval-ready results.

    ``dealer_rotation="full_table"`` is the battle-tested ``run_eval`` loop:
    keep the original seat list, normalize the dealer over non-busted seats,
    stop at ``max_hands``, then resolve unfinished survivors by chip count.

    ``dealer_rotation="active_circle"`` preserves the legacy UI/manager loop:
    pass only active seats to the hand engine and advance the dealer in that
    shrinking active circle.
    """
    seats = _coerce_seats(seats, chips)
    if dealer_rotation not in ("full_table", "active_circle"):
        raise ValueError(f"unknown dealer_rotation: {dealer_rotation!r}")
    if winner_resolution not in ("finish_order", "chip_count_on_max_hands"):
        raise ValueError(f"unknown winner_resolution: {winner_resolution!r}")

    table = table or (Table(rng=rng) if rng is not None else Table())
    dealer = dealer_index
    hand_count = 0
    total_players = len(seats)
    finish_order: list[tuple[str, int, int, int]] = []
    chip_history: list[dict[str, int]] = [
        {"hand": 0, **{seat.player_id: seat.chips for seat in seats}}
    ]

    _emit(on_event, {
        "type": "start",
        "hand": 0,
        "seats": seats,
        "chip_history": chip_history,
    })

    output_cm = redirect_stdout(io.StringIO()) if suppress_output else nullcontext()
    with output_cm:
        while True:
            active = _active_seats(seats)
            if len(active) <= 1:
                break
            if max_hands is not None and hand_count >= max_hands:
                break
            if should_cancel is not None and should_cancel():
                _emit(on_event, {
                    "type": "cancelled",
                    "hand": hand_count,
                    "seats": seats,
                    "chip_history": chip_history,
                })
                return {
                    "winner": None,
                    "hand_count": hand_count,
                    "finish_order": finish_order,
                    "final_chips": {s.player_id: s.chips for s in seats},
                    "final_stacks": {s.player_id: s.chips for s in seats},
                    "chip_swing": None,
                    "chip_history": chip_history,
                    "cancelled": True,
                }
            if wait_if_paused is not None:
                wait_if_paused()

            hand_count += 1
            sb, bb = escalate_blinds(
                hand_count, small_blind, big_blind, blind_increase_every
            )
            _emit(on_event, {
                "type": "hand_start",
                "hand": hand_count,
                "small_blind": sb,
                "big_blind": bb,
                "seats": seats,
            })

            if dealer_rotation == "full_table":
                hand_seats = seats
                hand_bots = {s.player_id: bot_for[s.player_id] for s in active}
                normalized_dealer = normalize_dealer_seat_index(seats, dealer)
                if normalized_dealer is None:
                    break
                hand_dealer = normalized_dealer
                dealer = normalized_dealer
            else:
                hand_seats = active
                hand_bots = {s.player_id: bot_for[s.player_id] for s in hand_seats}
                hand_dealer = dealer % len(hand_seats)

            table.play_hand(
                seats=hand_seats,
                small_blind=sb,
                big_blind=bb,
                dealer_index=hand_dealer,
                bot_for=hand_bots,
                on_event=None,
                log_decisions=log_decisions,
            )

            if dealer_rotation == "full_table":
                next_dealer = advance_dealer_seat_index(seats, dealer)
                if next_dealer is not None:
                    dealer = next_dealer
            else:
                next_active_count = len(_active_seats(seats))
                dealer = (dealer + 1) % max(next_active_count, 1)

            snapshot = {"hand": hand_count}
            for seat in seats:
                snapshot[seat.player_id] = seat.chips
            chip_history.append(snapshot)

            eliminated_events = []
            for seat in seats:
                already_ranked = any(e[0] == seat.player_id for e in finish_order)
                if seat.chips <= 0 and not already_ranked:
                    pos = total_players - len(finish_order)
                    entry = (seat.player_id, pos, hand_count, 0)
                    finish_order.append(entry)
                    eliminated_events.append(entry)

            _emit(on_event, {
                "type": "hand_end",
                "hand": hand_count,
                "small_blind": sb,
                "big_blind": bb,
                "seats": seats,
                "chip_history": chip_history,
                "eliminations": eliminated_events,
            })
            for pid, pos, _, _ in eliminated_events:
                _emit(on_event, {
                    "type": "elimination",
                    "hand": hand_count,
                    "player_id": pid,
                    "position": pos,
                    "seats": seats,
                    "chip_history": chip_history,
                })

            if hand_delay is not None:
                hand_delay()

    finalize_finish_order(seats, finish_order, total_players, hand_count)
    final_chips = {seat.player_id: seat.chips for seat in seats}
    winner = _winner_from_finish_order(finish_order)
    if (
        winner is None
        and winner_resolution == "chip_count_on_max_hands"
        and final_chips
    ):
        winner = max(final_chips, key=lambda pid: final_chips[pid])

    chip_swing = None
    if total_players == 2 and winner:
        losers = [seat.player_id for seat in seats if seat.player_id != winner]
        if losers:
            chip_swing = final_chips[winner] - final_chips[losers[0]]

    result = {
        "winner": winner,
        "hand_count": hand_count,
        "hands_played": hand_count,
        "finish_order": finish_order,
        "final_chips": final_chips,
        "final_stacks": final_chips,
        "chip_swing": chip_swing,
        "chip_history": chip_history,
        "cancelled": False,
    }
    _emit(on_event, {
        "type": "finish",
        "hand": hand_count,
        "winner": winner,
        "result": result,
        "seats": seats,
        "chip_history": chip_history,
    })
    return result
