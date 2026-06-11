"""
sanity_deep_cfr_curriculum.py — training-state curriculum gate (Fix 5 / I6)

Pre-fix, build_initial_state was the ONLY source of Deep CFR training states:
always 6 seats, 1000 chips each, 5/10 blinds — while eval plays 500 chips and
tournaments are decided heads-up, short-stacked, at escalated blinds.  The
curriculum randomizes player count (2-6) and stack depth (10-200 BB
log-uniform, 50% shared / 50% per-seat) at fixed 5/10 blinds, which is only
sound if the network features are big-blind-relative.  Hence the checks:

  1. Sampled states are structurally valid for n=2..6: engine blind seats
     (heads-up: button posts SB), engine action order, correct pot, hero
     rotation, depth bounds, both shared- and per-seat-depth modes.
  2. Depth range honored: chips span [10 BB, 200 BB] and actually spread.
  3. Action order parity vs the REAL engine for every n in 2..6 (scripted
     all-call/check hands compared event-by-event).
  4. to_network_input / opp_mask handle n < 6 opponents (mask sums to n-1,
     padded rows stay zero) — the mask supported this but training never
     exercised it.
  5. Blind posting with stacks below the blind (all-in from the blinds) keeps
     chip accounting sound and skips unactionable seats like the engine does.
  6. Scale invariance: multiplying ALL chips AND blinds by 10 leaves every
     network input tensor bit-identical (tree encoder and PlayerView encoder),
     with a negative control proving chips DO affect features at fixed blinds.
"""
from __future__ import annotations

import os
import random
import sys

# Repo root = this file's directory (gates live at the repo root), so the
# gate runs in any clone — never hard-code an absolute machine path here.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from core.action_history import ActionEvent, extract_history
from core.table_order import street_action_order
from core.bot_api import PlayerView
from bots.deep_cfr_bot import _DeepCFRGameState, build_network_input
from training.train_deep_cfr import (
    BIG_BLIND, SMALL_BLIND,
    CURRICULUM_PLAYER_COUNTS,
    CURRICULUM_MIN_DEPTH_BB, CURRICULUM_MAX_DEPTH_BB,
    build_initial_state,
    sample_curriculum_state,
)
# Engine-vs-tree drivers shared with the history-parity gate.
from sanity_deep_cfr_history_parity import (
    IDX, as_tuples, drive_tree, play_engine_hand,
)

MIN_CHIPS = CURRICULUM_MIN_DEPTH_BB * BIG_BLIND   # 100
MAX_CHIPS = CURRICULUM_MAX_DEPTH_BB * BIG_BLIND   # 2000


def check_sampler_validity() -> bool:
    ok = True
    rng = random.Random(20260611)
    random.seed(9000)  # build_initial_state shuffles the module-RNG deck

    n_samples = 600
    seen_counts = set()
    seen_shared = 0
    seen_asymmetric = 0
    depth_lo = float("inf")
    depth_hi = 0.0

    for t in range(1, n_samples + 1):
        state, hero_seat = sample_curriculum_state(t, rng=rng)
        n = len(state.stacks)
        seen_counts.add(n)

        # Hero rotation rule unchanged: (iteration - 1) % n_seats.
        if hero_seat != (t - 1) % n:
            ok = False
            print(f"  [FAIL] — t={t}: hero_seat {hero_seat} != {(t - 1) % n}")
            break

        # Blind seats follow the engine; pot equals the posted blinds.
        sb_seat, bb_seat = (0, 1) if n == 2 else (1, 2)
        if state.committed_per_seat[sb_seat] != SMALL_BLIND:
            ok = False
            print(f"  [FAIL] — t={t}: SB seat {sb_seat} posted "
                  f"{state.committed_per_seat[sb_seat]}")
            break
        if state.committed_per_seat[bb_seat] != BIG_BLIND:
            ok = False
            print(f"  [FAIL] — t={t}: BB seat {bb_seat} posted "
                  f"{state.committed_per_seat[bb_seat]}")
            break
        if state.pot != SMALL_BLIND + BIG_BLIND:
            ok = False
            print(f"  [FAIL] — t={t}: pot {state.pot}")
            break

        # Action order: the shared engine helper, minus busted seats.
        expected_order = [s for s in street_action_order("preflop", list(range(n)))
                          if state.stacks[s] > 0]
        if state.seat_order != expected_order:
            ok = False
            print(f"  [FAIL] — t={t}: seat_order {state.seat_order} != "
                  f"{expected_order} (n={n})")
            break

        # Depth bounds: pre-blind stack = stack + committed, in chips.
        pre_blind = [state.stacks[i] + state.committed_per_seat[i]
                     for i in range(n)]
        # int(round()) of a 10-200 BB depth in chips.
        if min(pre_blind) < MIN_CHIPS or max(pre_blind) > MAX_CHIPS:
            ok = False
            print(f"  [FAIL] — t={t}: stacks {pre_blind} outside "
                  f"[{MIN_CHIPS}, {MAX_CHIPS}]")
            break
        depth_lo = min(depth_lo, min(pre_blind))
        depth_hi = max(depth_hi, max(pre_blind))
        if len(set(pre_blind)) == 1:
            seen_shared += 1
        else:
            seen_asymmetric += 1

        # Card accounting: 2 unique hole cards per seat, deck holds the rest.
        all_hole = [c for cards in state.hole_cards.values() for c in cards]
        if len(set(all_hole)) != 2 * n or len(state.deck_remaining) != 52 - 2 * n:
            ok = False
            print(f"  [FAIL] — t={t}: card accounting broken")
            break

    if ok:
        print(f"  [PASS] — {n_samples} sampled states structurally valid "
              f"(blinds, order, pot, rotation, cards)")
    if seen_counts == set(CURRICULUM_PLAYER_COUNTS):
        print(f"  [PASS] — all player counts sampled: {sorted(seen_counts)}")
    else:
        ok = False
        print(f"  [FAIL] — player counts seen {sorted(seen_counts)}, "
              f"expected {list(CURRICULUM_PLAYER_COUNTS)}")
    # 50/50 shared vs per-seat: both modes must actually occur (loose bounds).
    if seen_shared >= n_samples // 5 and seen_asymmetric >= n_samples // 5:
        print(f"  [PASS] — both depth modes sampled "
              f"(shared={seen_shared}, per-seat={seen_asymmetric})")
    else:
        ok = False
        print(f"  [FAIL] — depth-mode split degenerate "
              f"(shared={seen_shared}, per-seat={seen_asymmetric})")
    print(f"  observed depth range: [{depth_lo:.0f}, {depth_hi:.0f}] chips")
    # Log-uniform over [100, 2000] must reach both tails (loose bounds).
    if depth_lo <= 300 and depth_hi >= 1000:
        print("  [PASS] — depth range spreads into both tails")
    else:
        ok = False
        print("  [FAIL] — sampled depths did not cover the range")
    return ok


def check_engine_order_parity() -> bool:
    """Scripted all-call/check hands: tree (production builder) vs engine."""
    ok = True
    for n in (2, 3, 4, 5, 6):
        stacks = [500] * n
        # Every seat plays check_call once per street (call preflop, check
        # after) — exactly 4 actions each in an aggression-free hand.
        scripts = {i: [IDX["check_call"]] * 4 for i in range(n)}

        engine_view = play_engine_hand(stacks, {k: list(v) for k, v in scripts.items()})
        engine_events = as_tuples(extract_history(engine_view))

        random.seed(4242)  # deck order inside build_initial_state — irrelevant
        state = build_initial_state(n_seats=n, hero_seat=0, stacks=stacks)
        tree_events = as_tuples(drive_tree(state, scripts))

        # The final view excludes the very last action; compare the prefix.
        n_compare = len(engine_events)
        if n_compare >= 4 * n - 1 and tree_events[:n_compare] == engine_events:
            print(f"  [PASS] — n={n}: {n_compare} events match the engine "
                  f"(street, seat, action, amount, pot_before)")
        else:
            ok = False
            print(f"  [FAIL] — n={n}: tree/engine action sequences diverge")
            for i in range(max(len(tree_events), n_compare)):
                t = tree_events[i] if i < len(tree_events) else "<missing>"
                e = engine_events[i] if i < n_compare else "<missing>"
                marker = "  " if t == e else "✗ "
                print(f"      {marker}tree={t}  engine={e}")
    return ok


def check_opp_mask() -> bool:
    """to_network_input must mask exactly n-1 opponents for n=2..6."""
    ok = True
    for n in (2, 3, 4, 5, 6):
        random.seed(777)
        state = build_initial_state(n_seats=n, hero_seat=0,
                                    stacks=[400] * n)
        batch = state.to_network_input(0)
        mask = batch["opp_mask"]
        feats = batch["opp_features"]
        n_masked = int(mask.sum().item())
        padded_rows_zero = bool(
            (feats[0][mask[0] == 0].abs().sum() == 0).item()
        ) if n_masked < feats.shape[1] else True
        if n_masked == n - 1 and padded_rows_zero:
            print(f"  [PASS] — n={n}: opp_mask sums to {n - 1}, padded rows zero")
        else:
            ok = False
            print(f"  [FAIL] — n={n}: opp_mask sum {n_masked} "
                  f"(want {n - 1}), padded_zero={padded_rows_zero}")
    return ok


def check_short_blind_posting() -> bool:
    """Stacks below the blind: min() posting, busted seats skipped, chips conserved."""
    ok = True
    random.seed(31337)
    state = build_initial_state(n_seats=3, hero_seat=0, stacks=[100, 3, 7])
    # SB had 3 chips (< 5), BB had 7 (< 10): both all-in from the blinds.
    if state.committed_per_seat == [0, 3, 7] and state.pot == 10:
        print("  [PASS] — short blinds post min(stack, blind); pot = 10")
    else:
        ok = False
        print(f"  [FAIL] — committed {state.committed_per_seat}, pot {state.pot}")
    if state.stacks == [100, 0, 0] and min(state.stacks) >= 0:
        print("  [PASS] — no negative stacks after posting")
    else:
        ok = False
        print(f"  [FAIL] — stacks {state.stacks}")
    if state.seat_order == [0]:
        print("  [PASS] — all-in blind seats excluded from the action order "
              "(engine skips them silently)")
    else:
        ok = False
        print(f"  [FAIL] — seat_order {state.seat_order}, expected [0]")

    # Drive the hand to the river: the lone actionable seat checks/calls down.
    try:
        events = drive_tree(state, {0: [IDX["check_call"]] * 4})
        # Seat 0 owes the 7-chip BB all-in: a call of min(stack, to_call)=7.
        flat = [(e.street, e.seat, e.action, e.amount) for e in events]
        if ("preflop", 0, "call", 7) in flat:
            print("  [PASS] — facing the short BB all-in, seat 0 calls 7")
        else:
            ok = False
            print(f"  [FAIL] — expected a 7-chip call; events={flat}")
    except Exception as e:  # noqa: BLE001 — any crash is the failure
        ok = False
        print(f"  [FAIL] — driving the short-blind hand raised "
              f"{type(e).__name__}: {e}")
    return ok


def _scale_state(state: _DeepCFRGameState, k: int) -> _DeepCFRGameState:
    """Multiply every chip quantity AND the blind level by k (same game)."""
    return _DeepCFRGameState(
        pot=state.pot * k,
        stacks=[s * k for s in state.stacks],
        committed_per_seat=[c * k for c in state.committed_per_seat],
        total_committed_per_seat=[c * k for c in state.total_committed_per_seat],
        alive=list(state.alive),
        street=state.street,
        board=list(state.board),
        hole_cards=dict(state.hole_cards),
        seat_order=list(state.seat_order),
        action_idx=state.action_idx,
        history_events=[
            ActionEvent(seat=e.seat, street=e.street, action=e.action,
                        amount=e.amount * k, pot_before=e.pot_before * k)
            for e in state.history_events
        ],
        deck_remaining=list(state.deck_remaining),
        big_blind=state.big_blind * k,
        ring_order=list(state.ring_order),
        street_actions=state.street_actions,
        last_raise_size=state.last_raise_size * k,
        raise_blocked=set(state.raise_blocked),
        acted=set(state.acted),
    )


def _batches_equal(a: dict, b: dict) -> tuple[bool, str]:
    for key in a:
        if not torch.equal(a[key], b[key]):
            diff = (a[key].float() - b[key].float()).abs().max().item()
            return False, f"{key} differs (max abs diff {diff})"
    return True, ""


def check_scale_invariance() -> bool:
    ok = True
    # Build a 3-handed state with asymmetric depths and some betting history
    # (a raise and a call) so the history channels are exercised too.
    random.seed(123)
    state = build_initial_state(n_seats=3, hero_seat=0,
                                stacks=[300, 500, 800])
    state = state.apply_action(0, IDX["bet_50"])      # BTN raises to 3 BB
    state = state.apply_action(1, IDX["check_call"])  # SB calls

    hero = 2  # BB to act, facing the raise
    base = state.to_network_input(hero)
    scaled = _scale_state(state, 10).to_network_input(hero)
    equal, why = _batches_equal(base, scaled)
    if equal:
        print("  [PASS] — tree encoder: x10 chips AND blinds → identical tensors")
    else:
        ok = False
        print(f"  [FAIL] — tree encoder not blind-invariant: {why}")

    # Negative control: x10 chips at UNCHANGED blinds must differ (a 30-BB
    # spot and a 300-BB spot are different states — chips must matter).
    control = _scale_state(state, 10)
    control.big_blind = state.big_blind            # undo the blind scaling
    control = control.to_network_input(hero)
    equal, _ = _batches_equal(base, control)
    if not equal:
        print("  [PASS] — control: x10 chips at FIXED blinds changes features")
    else:
        ok = False
        print("  [FAIL] — features ignore chip scale entirely (vacuous invariance)")

    # PlayerView (inference) encoder: same invariance through build_network_input,
    # with the big blind INFERRED from the view's history.
    def make_view(k: int) -> PlayerView:
        return PlayerView(
            me="hero", street="preflop", position="BB",
            hole_cards=[("A", "h"), ("K", "h")], board=[],
            pot=75 * k, to_call=20 * k,
            min_raise=50 * k, max_raise=470 * k,
            legal_actions=[{"type": "fold"}, {"type": "call"},
                           {"type": "raise", "min": 50 * k, "max": 480 * k}],
            stacks={"hero": 470 * k, "opp0": 270 * k, "opp1": 770 * k},
            opponents=["opp0", "opp1"],
            history=[
                {"street": "preflop", "pid": "opp0", "type": "raise",
                 "amount": 30 * k, "to_call_before": 10 * k,
                 "pot_before": 15 * k},
                {"street": "preflop", "pid": "opp1", "type": "call",
                 "amount": 30 * k, "to_call_before": 30 * k,
                 "pot_before": 45 * k},
            ],
        )

    base_v = build_network_input(make_view(1))
    scaled_v = build_network_input(make_view(10))
    equal, why = _batches_equal(base_v, scaled_v)
    if equal:
        print("  [PASS] — PlayerView encoder: x10 chips AND blinds → "
              "identical tensors (blind inferred from history)")
    else:
        ok = False
        print(f"  [FAIL] — PlayerView encoder not blind-invariant: {why}")
    return ok


def run() -> bool:
    PASS = True
    sections = [
        ("Check 1: curriculum sampler validity (n, blinds, order, depth, rotation)",
         check_sampler_validity),
        ("Check 2: action-order parity vs the real engine, n=2..6",
         check_engine_order_parity),
        ("Check 3: to_network_input / opp_mask with n<6 opponents",
         check_opp_mask),
        ("Check 4: blind posting with stacks below the blind",
         check_short_blind_posting),
        ("Check 5: scale invariance (x10 chips and blinds → same tensors)",
         check_scale_invariance),
    ]
    for title, fn in sections:
        print("=" * 60)
        print(title)
        print("=" * 60)
        try:
            PASS &= fn()
        except Exception as e:  # noqa: BLE001 — a crash is a failure, not an abort
            import traceback
            traceback.print_exc()
            print(f"  [FAIL] — {type(e).__name__}: {e}")
            PASS = False
        print()

    print("=" * 60)
    if PASS:
        print("ALL CHECKS PASSED [PASS]")
    else:
        print("SOME CHECKS FAILED [FAIL]")
    print("=" * 60)
    return PASS


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
