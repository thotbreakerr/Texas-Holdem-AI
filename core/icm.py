"""
core/icm.py — Independent Chip Model transform
-----------------------------------------------
Malmuth-Harville ICM math extracted from bots/icm_bot.py.
Prize structure is a parameter — winner-take-all is the locked match
format but not baked in.

Both Path A and Path B import from here. This module is read-only
after Gate 1 closes.
"""
from __future__ import annotations

from typing import List


def equities(stacks: list, payouts: list) -> list:
    """Tournament equity for each player.

    Parameters
    ----------
    stacks : list of int
        Chip counts per seat. 0 = busted, included as 0 equity.
    payouts : list of float
        Prize fractions, must sum to 1.0.
        Winner-take-all = [1.0, 0.0, 0.0, ...].

    Returns
    -------
    list[float] of length len(stacks), summing to 1.0.
    """
    n = len(stacks)
    if n == 0:
        return []

    alive = [i for i in range(n) if stacks[i] > 0]
    n_alive = len(alive)

    # All busted
    if n_alive == 0:
        return [0.0] * n

    # Single survivor
    if n_alive == 1:
        result = [0.0] * n
        result[alive[0]] = 1.0
        return result

    chip_total = sum(stacks[i] for i in alive)
    if chip_total == 0:
        # Edge case: all-zero but alive (shouldn't happen, defensive)
        eq = 1.0 / n_alive
        result = [0.0] * n
        for i in alive:
            result[i] = eq
        return result

    # Ensure payouts list is at least n_alive long (pad with zeros)
    padded_payouts = list(payouts) + [0.0] * max(0, n_alive - len(payouts))

    # Index of the last payout slot that actually pays out anything.  Once the
    # recursion advances past it there is no equity left to distribute, so we
    # can stop — this prunes the full O(N!) permutation tree (e.g. winner-take-
    # all collapses to O(N) since only slot 0 is nonzero).
    last_paid_idx = -1
    for i, p in enumerate(padded_payouts):
        if abs(p) > 1e-12:
            last_paid_idx = i

    equity_result = [0.0] * n

    def _recurse(remaining: list, remaining_total: int, payout_idx: int,
                 path_prob: float):
        """Malmuth-Harville recursion.

        For each remaining player, their probability of finishing in
        position `payout_idx` is proportional to their stack fraction.
        Multiply by that position's payout and recurse.
        """
        if (payout_idx >= len(padded_payouts) or not remaining
                or payout_idx > last_paid_idx):
            return
        if len(remaining) == 1:
            # Only one player can fill this finishing position — they take the
            # payout for THIS slot only.  (Trailing slots beyond the number of
            # alive players belong to already-busted seats whose locked prizes
            # are not redistributed; summing them here misallocated those chips
            # to the last survivor and made the result stack-insensitive.)
            seat = remaining[0]
            equity_result[seat] += path_prob * padded_payouts[payout_idx]
            return

        for seat in remaining:
            prob = (stacks[seat] / remaining_total
                    if remaining_total > 0
                    else 1.0 / len(remaining))
            joint_prob = path_prob * prob
            equity_result[seat] += joint_prob * padded_payouts[payout_idx]

            new_remaining = [s for s in remaining if s != seat]
            new_total = remaining_total - stacks[seat]
            _recurse(new_remaining, new_total, payout_idx + 1, joint_prob)

    # Exact recursion for up to 8 players (O(N!) complexity)
    if n_alive <= 8:
        _recurse(alive, chip_total, 0, 1.0)
    else:
        if any(abs(p) > 1e-12 for p in padded_payouts[1:]):
            raise ValueError(
                "Exact ICM not implemented for >8 players with non-WTA payouts; "
                "use core.icm with <=8 players or supply WTA payouts"
            )
        # Approximation for large fields: equity ≈ stack fraction
        for i in alive:
            equity_result[i] = stacks[i] / chip_total

    return equity_result


def equity_delta(stacks_before: list, stacks_after: list,
                 payouts: list, hero_seat: int) -> float:
    """Hero's tournament-equity delta from stacks_before to stacks_after.

    Parameters
    ----------
    stacks_before : list of int
    stacks_after : list of int
    payouts : list of float
    hero_seat : int (index into stacks lists)

    Returns
    -------
    float: eq_after[hero_seat] - eq_before[hero_seat]
    """
    eq_before = equities(stacks_before, payouts)
    eq_after = equities(stacks_after, payouts)
    return eq_after[hero_seat] - eq_before[hero_seat]
