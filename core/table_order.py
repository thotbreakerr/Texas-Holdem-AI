"""Shared table-order helpers for engine drivers and CFR search."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional


def street_action_order(street: str, ring_order: Iterable[int]) -> List[int]:
    """Return the fresh-street action order for a dealer-first ring."""
    ring = list(ring_order)
    if not ring:
        return []
    n = len(ring)
    if street == "preflop":
        start = 3 if n > 2 else 0
    else:
        start = 1 if n > 1 else 0
    start %= n
    return ring[start:] + ring[:start]


def is_active_seat(seat: Any) -> bool:
    """True when a seat should be included in the next hand."""
    return (
        getattr(seat, "chips", 0) > 0
        and not getattr(seat, "is_sitting_out", False)
    )


def active_seat_indices(seats: Iterable[Any]) -> List[int]:
    """Indices of seats with chips that are not sitting out."""
    return [i for i, seat in enumerate(seats) if is_active_seat(seat)]


def next_active_seat_index(
    seats: List[Any],
    after_index: int,
) -> Optional[int]:
    """Return the next active seat clockwise after ``after_index``."""
    if not seats:
        return None
    n = len(seats)
    for offset in range(1, n + 1):
        idx = (after_index + offset) % n
        if is_active_seat(seats[idx]):
            return idx
    return None


def normalize_dealer_seat_index(
    seats: List[Any],
    dealer_seat_index: int,
) -> Optional[int]:
    """Return an active full-table dealer index nearest the requested button."""
    if not seats:
        return None
    idx = dealer_seat_index % len(seats)
    if is_active_seat(seats[idx]):
        return idx
    return next_active_seat_index(seats, idx - 1)


def advance_dealer_seat_index(
    seats: List[Any],
    dealer_seat_index: int,
) -> Optional[int]:
    """Advance the button using full-table indices, skipping busted seats."""
    if not seats:
        return None
    idx = dealer_seat_index % len(seats)
    return next_active_seat_index(seats, idx)
