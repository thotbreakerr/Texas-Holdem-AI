"""
Regression for core.icm trailing-payout misallocation + WTA pruning.

Bug (pre-fix): when the payout list was longer than the number of alive players
(busted seats with locked prizes), the last survivor absorbed the SUM of all
trailing payout slots.  That made the output stack-insensitive
(equities([0,600,400],...) == equities([0,800,200],...)) and corrupted the
AIVAT tournament delta.  Separately, winner-take-all walked the full O(N!)
permutation tree because zero trailing payouts were never pruned.
"""
import sys

sys.path.insert(0, "/Users/jaroslavaupart/Desktop/Projects/Texas-Holdem-AI")

from core import icm


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def run():
    PASS = True

    # CHECK 1: payouts longer than alive count — last survivor must NOT absorb
    # the trailing (busted-seat) slot; result must be stack-sensitive.
    r1 = icm.equities([0, 600, 400], [0.5, 0.3, 0.2])
    expected = [0.0, 0.42, 0.38]
    if all(approx(a, b) for a, b in zip(r1, expected)):
        print(f"[CHECK 1] PASS — {r1} == {expected}")
    else:
        PASS = False
        print(f"[CHECK 1] FAIL — got {r1}, expected {expected}")

    # CHECK 2: output is stack-sensitive (the bug made these identical).
    r2 = icm.equities([0, 800, 200], [0.5, 0.3, 0.2])
    if not all(approx(a, b) for a, b in zip(r1, r2)):
        print(f"[CHECK 2] PASS — stack-sensitive ({r1} != {r2})")
    else:
        PASS = False
        print(f"[CHECK 2] FAIL — stack-insensitive: {r1} == {r2}")

    # CHECK 3: normal full payout case (len == alive) is unchanged and sums to 1.
    r3 = icm.equities([500, 300, 200], [0.5, 0.3, 0.2])
    if approx(sum(r3), 1.0):
        print(f"[CHECK 3] PASS — full payout sums to 1.0 ({r3})")
    else:
        PASS = False
        print(f"[CHECK 3] FAIL — full payout does not sum to 1: {r3} (sum={sum(r3)})")

    # CHECK 4: winner-take-all == stack fractions, and pruning gives the right
    # answer for a larger field.
    stacks = [100, 200, 300, 400]
    wta = icm.equities(stacks, [1.0, 0, 0, 0])
    total = sum(stacks)
    expected_wta = [s / total for s in stacks]
    if all(approx(a, b) for a, b in zip(wta, expected_wta)):
        print(f"[CHECK 4] PASS — WTA == stack fractions ({wta})")
    else:
        PASS = False
        print(f"[CHECK 4] FAIL — WTA {wta} != {expected_wta}")

    # CHECK 5: 8-player WTA computes quickly (pruning) and is correct.
    big = [100 * (i + 1) for i in range(8)]
    wta8 = icm.equities(big, [1.0] + [0.0] * 7)
    if approx(wta8[0], big[0] / sum(big)) and approx(sum(wta8), 1.0):
        print(f"[CHECK 5] PASS — 8-player WTA correct (seat0={wta8[0]:.4f})")
    else:
        PASS = False
        print(f"[CHECK 5] FAIL — 8-player WTA wrong: {wta8}")

    print("=" * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED [PASS]' if PASS else 'SOME CHECKS FAILED [FAIL]'}")
    return PASS


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
