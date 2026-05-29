"""
sanity_deep_cfr.py — Gate 2B sanity checks for Deep CFR Plus
-------------------------------------------------------------
Mirrors sanity_aivat.py structure. Sections:
  1. Parameter count check
  2. Forward-pass shape check
  3. Overfit-tiny-batch on each head
  4. Real-time search OOM check at production depth
  5. Known-equity ballpark check
  6. Sizing head output range and distribution
  7. Regret head distribution
  8. Smoke tournament with search exercised
  9. Recursive CFR consumes state and descends
 10. Opponent features update across hands
"""
import argparse
import io
import math
import random
import sys
import time as _time
from contextlib import redirect_stdout

sys.path.insert(0, ".")

import torch
import torch.nn as nn

from bots.deep_cfr_bot import (
    DeepCFRBot, DeepCFRConfig, DeepCFRNetwork,
    build_network_input, _build_random_synthetic_input,
    _DeepCFRGameState, _legal_abstract_actions,
    ABSTRACT_ACTIONS, NUM_ACTIONS, _SCALAR_DIM, HIST_FEATURE_DIM, _HISTORY_MAX_LEN,
    _MAX_OPPONENTS, _OPP_FEAT_DIM,
)
from core.bot_api import PlayerView, Action
from core.engine import Table, Seat, InProcessBot, RandomBot

PASS = True


def _make_view(hole, board, pot=100, to_call=30, street="flop",
               position="BTN", n_opp=1, stacks=None):
    """Build a minimal synthetic PlayerView for testing."""
    hero = "P_hero"
    if stacks is None:
        stacks = {hero: 500}
        for i in range(n_opp):
            stacks[f"P_opp{i}"] = 500
    opponents = [p for p in stacks if p != hero]
    min_raise = max(to_call * 2, 20)
    max_raise = stacks[hero]

    legal = [{"type": "fold"}]
    if to_call == 0:
        legal = [{"type": "check"}]
    else:
        legal.append({"type": "call"})
    if stacks[hero] > to_call:
        legal.append({"type": "raise", "min": min_raise, "max": max_raise})

    return PlayerView(
        me=hero, street=street, position=position,
        hole_cards=list(hole), board=list(board),
        pot=pot, to_call=to_call,
        min_raise=min_raise, max_raise=max_raise,
        legal_actions=legal,
        stacks=stacks,
        opponents=opponents,
        history=[],
    )


def _concat_batches(batches):
    """Concatenate single-row network-input batches along the batch axis."""
    return {k: torch.cat([b[k] for b in batches], dim=0) for k in batches[0]}


def run_checks(variant: str):
    global PASS
    config = DeepCFRConfig.small() if variant == "small" else DeepCFRConfig.large()
    print(f"\n{'='*60}")
    print(f"  VARIANT: {variant.upper()}")
    print(f"{'='*60}\n")

    # ═══════════════════════════════════════════════════════════════════════
    #  SECTION 1 — Parameter count check
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("Section 1: Parameter count check")
    print("=" * 60)

    with redirect_stdout(io.StringIO()):
        net = DeepCFRNetwork(config)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"  {variant} params: {total_params:,}")

    if 100_000 <= total_params <= 50_000_000:
        print("  [PASS] — within sane bounds [100K, 50M]")
    else:
        print(f"  [FAIL] — {total_params:,} outside [100K, 50M]")
        PASS = False

    # Cross-variant check
    other_config = DeepCFRConfig.large() if variant == "small" else DeepCFRConfig.small()
    with redirect_stdout(io.StringIO()):
        other_net = DeepCFRNetwork(other_config)
    other_params = sum(p.numel() for p in other_net.parameters())
    print(f"  other variant params: {other_params:,}")

    if variant == "small":
        if other_params >= 2 * total_params:
            print("  [PASS] — large >= 2x small")
        else:
            print(f"  [FAIL] — large ({other_params:,}) < 2x small ({total_params:,})")
            PASS = False
    else:
        if total_params >= 2 * other_params:
            print("  [PASS] — large >= 2x small")
        else:
            print(f"  [FAIL] — large ({total_params:,}) < 2x small ({other_params:,})")
            PASS = False

    del other_net
    print()

    # ═══════════════════════════════════════════════════════════════════════
    #  SECTION 2 — Forward-pass shape check
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("Section 2: Forward-pass shape check")
    print("=" * 60)

    view = _make_view(
        hole=[("A", "h"), ("K", "h")],
        board=[("K", "s"), ("7", "d"), ("2", "c")],
        pot=100, to_call=30, street="flop",
    )
    batch = build_network_input(view)
    with torch.no_grad():
        out = net(batch)

    checks = [
        ("regret", out["regret"].shape, (1, NUM_ACTIONS)),
        ("state", out["state"].shape, (1, config.state_dim)),
    ]
    for name, got, expected in checks:
        if got == expected:
            print(f"  {name}: shape {got} [PASS]")
        else:
            print(f"  {name}: shape {got}, expected {expected} [FAIL]")
            PASS = False

    # value and sizing can be (1,) or (1,1)
    for name in ("value", "sizing"):
        t = out[name]
        if t.shape in ((1,), (1, 1), torch.Size([1]), torch.Size([1, 1])):
            print(f"  {name}: shape {tuple(t.shape)} [PASS]")
        else:
            print(f"  {name}: shape {tuple(t.shape)} [FAIL]")
            PASS = False

    # NaN/Inf check
    any_bad = False
    for name in ("regret", "value", "sizing", "state"):
        t = out[name]
        if torch.isnan(t).any() or torch.isinf(t).any():
            print(f"  {name}: contains NaN/Inf [FAIL]")
            PASS = False
            any_bad = True
    if not any_bad:
        print("  No NaN/Inf in any output [PASS]")
    print()

    # ═══════════════════════════════════════════════════════════════════════
    #  SECTION 3 — Overfit-tiny-batch on each head
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("Section 3: Overfit-tiny-batch on each head")
    print("=" * 60)

    # Rebuild fresh network for overfit test
    with redirect_stdout(io.StringIO()):
        overfit_net = DeepCFRNetwork(config)
    overfit_net.train()

    # Generate 32 real-shaped samples so the value head learns an actual
    # card signal. Pure random tensors can overfit loss without learning that
    # AA should score above 72o in Section 5.
    strong_hands = [
        [("A", "h"), ("A", "s")],
        [("A", "h"), ("K", "h")],
        [("K", "c"), ("K", "d")],
        [("Q", "s"), ("Q", "h")],
    ]
    weak_hands = [
        [("7", "h"), ("2", "s")],
        [("8", "c"), ("3", "d")],
        [("9", "s"), ("4", "c")],
        [("3", "h"), ("2", "d")],
    ]
    train_views = []
    value_target_vals = []
    for _ in range(4):
        for hand in strong_hands:
            train_views.append(_make_view(hand, [], pot=30, to_call=20, street="preflop"))
            value_target_vals.append(0.6)
        for hand in weak_hands:
            train_views.append(_make_view(hand, [], pot=30, to_call=20, street="preflop"))
            value_target_vals.append(-0.4)
    value_batch = _concat_batches([build_network_input(v) for v in train_views])
    random_batch = _build_random_synthetic_input(config, batch_size=32)

    random.seed(42)
    torch.manual_seed(42)

    # Value head targets
    value_targets = torch.tensor(value_target_vals, dtype=torch.float32)

    # Regret head targets
    regret_targets = torch.randn(32, NUM_ACTIONS) * 0.5

    # Sizing head targets
    sizing_targets = torch.tensor([
        0.75, 0.33, 0.50, 1.0, 0.67, 0.25, 0.80, 0.45,
        0.75, 0.33, 0.50, 1.0, 0.67, 0.25, 0.80, 0.45,
        0.75, 0.33, 0.50, 1.0, 0.67, 0.25, 0.80, 0.45,
        0.75, 0.33, 0.50, 1.0, 0.67, 0.25, 0.80, 0.45,
    ], dtype=torch.float32)

    heads = [
        ("value", overfit_net.value_head, overfit_net.regret_head, overfit_net.sizing_head,
         value_batch, value_targets, lambda o, t: nn.functional.mse_loss(o["value"], t)),
        ("regret", overfit_net.regret_head, overfit_net.value_head, overfit_net.sizing_head,
         random_batch, regret_targets, lambda o, t: nn.functional.mse_loss(o["regret"], t)),
        ("sizing", overfit_net.sizing_head, overfit_net.value_head, overfit_net.regret_head,
         random_batch, sizing_targets, lambda o, t: nn.functional.mse_loss(o["sizing"], t)),
    ]

    for head_name, train_head, freeze1, freeze2, batch, targets, loss_fn in heads:
        # Freeze other heads
        for p in freeze1.parameters():
            p.requires_grad = False
        for p in freeze2.parameters():
            p.requires_grad = False
        # Unfreeze this head + encoder
        for p in train_head.parameters():
            p.requires_grad = True
        for p in overfit_net.encoder.parameters():
            p.requires_grad = True

        trainable = [p for p in overfit_net.parameters() if p.requires_grad]
        opt = torch.optim.Adam(trainable, lr=1e-3)

        final_loss = float("inf")
        for step in range(200):
            opt.zero_grad()
            out = overfit_net(batch)
            loss = loss_fn(out, targets)
            loss.backward()
            opt.step()
            final_loss = loss.item()

        threshold = 0.05
        if final_loss < threshold:
            print(f"  {head_name} head: final loss={final_loss:.6f} < {threshold} [PASS]")
        else:
            print(f"  {head_name} head: final loss={final_loss:.6f} >= {threshold} [FAIL]")
            PASS = False

        # Restore grads
        for p in freeze1.parameters():
            p.requires_grad = True
        for p in freeze2.parameters():
            p.requires_grad = True

    print()

    # ═══════════════════════════════════════════════════════════════════════
    #  SECTION 4 — Real-time search expansion check at production depth
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("Section 4: Real-time search expansion check (depth=4)")
    print("=" * 60)

    with redirect_stdout(io.StringIO()):
        bot = DeepCFRBot(config=config, inference_mode=True)
    bot._weights_loaded = True  # pretend weights loaded to trigger search

    view_6h = _make_view(
        hole=[("A", "s"), ("K", "d")],
        board=[("Q", "h"), ("J", "c"), ("T", "s")],
        pot=200, to_call=50, street="flop",
        position="CO", n_opp=5,
    )

    legal_mask = _legal_abstract_actions(view_6h.legal_actions, view_6h.pot)
    prior = [1.0 / NUM_ACTIONS] * NUM_ACTIONS

    search_counts = {}
    search_times = {}
    try:
        for depth in (1, 2, 3, 4):
            bot._search_leaf_calls = 0
            t0 = _time.monotonic()
            result = bot._subgame_search(view_6h, prior, legal_mask, depth=depth)
            elapsed = _time.monotonic() - t0
            search_counts[depth] = bot._search_leaf_calls
            search_times[depth] = elapsed
            dist_sum = sum(result[a] for a in legal_mask)
            print(
                f"  depth={depth}: leaves={search_counts[depth]}, "
                f"time={elapsed:.2f}s, legal-prob-sum={dist_sum:.3f}"
            )
            if not all(math.isfinite(result[a]) for a in legal_mask):
                print(f"  [FAIL] — non-finite strategy at depth={depth}")
                PASS = False

        if (search_counts[1] < search_counts[2] <
                search_counts[3] < search_counts[4] and search_counts[4] > 10):
            print("  [PASS] — leaf-call count grows with depth")
        else:
            print(f"  [FAIL] — leaf counts did not grow properly: {search_counts}")
            PASS = False

        elapsed4 = search_times[4]
        if elapsed4 > 15.0:
            print(f"  [FAIL] — depth=4 exceeded 15s budget")
            PASS = False
        elif elapsed4 > 5.0:
            print(f"  [DIAGNOSTIC] depth=4 exceeded 5s soft budget")
        else:
            print("  [PASS] — depth=4 within time budget")

    except Exception as e:
        print(f"  [FAIL] — exception: {e}")
        PASS = False

    print("  (memory tracking skipped; leaf counts verify expansion)")

    del bot
    print()

    # ═══════════════════════════════════════════════════════════════════════
    #  SECTION 5 — Known-equity ballpark check
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("Section 5: Known-equity ballpark check")
    print("=" * 60)

    # Untrained check
    with redirect_stdout(io.StringIO()):
        fresh_net = DeepCFRNetwork(config)
    fresh_net.eval()

    view_aa = _make_view(
        hole=[("A", "h"), ("A", "s")],
        board=[], pot=30, to_call=20, street="preflop",
    )
    batch_aa = build_network_input(view_aa)
    with torch.no_grad():
        out_aa = fresh_net(batch_aa)

    val_aa = out_aa["value"].item()
    print(f"  Untrained AA preflop value: {val_aa:.4f}")

    if -1.0 <= val_aa <= 1.0:
        print("  [PASS] — in [-1, 1]")
    elif not math.isnan(val_aa) and math.isfinite(val_aa):
        print(f"  [PASS] — finite (outside [-1,1] but no NaN/Inf)")
    else:
        print("  [FAIL] — NaN or Inf")
        PASS = False

    any_nan = any(torch.isnan(out_aa[k]).any() or torch.isinf(out_aa[k]).any()
                  for k in ("regret", "value", "sizing", "state"))
    if not any_nan:
        print("  No NaN/Inf [PASS]")
    else:
        print("  [FAIL] — NaN/Inf detected")
        PASS = False

    # Partially trained: use overfit network from section 3
    overfit_net.eval()
    view_72 = _make_view(
        hole=[("7", "h"), ("2", "s")],
        board=[], pot=30, to_call=20, street="preflop",
    )
    batch_72 = build_network_input(view_72)
    batch_aa2 = build_network_input(view_aa)
    with torch.no_grad():
        val_aa_trained = overfit_net(batch_aa2)["value"].item()
        val_72_trained = overfit_net(batch_72)["value"].item()

    print(f"  Trained AA value: {val_aa_trained:.4f}, 72o value: {val_72_trained:.4f}")
    if val_aa_trained > val_72_trained:
        print("  [PASS] — AA > 72o directionally correct")
    else:
        print(f"  [FAIL] — direction wrong (AA={val_aa_trained:.4f}, "
              f"72o={val_72_trained:.4f})")
        PASS = False
    print()

    del fresh_net

    # ═══════════════════════════════════════════════════════════════════════
    #  SECTION 6 — Sizing head output range and distribution
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("Section 6: Sizing head output range and distribution")
    print("=" * 60)

    with redirect_stdout(io.StringIO()):
        range_net = DeepCFRNetwork(config)
    range_net.eval()

    syn_200 = _build_random_synthetic_input(config, batch_size=200)
    with torch.no_grad():
        out_200 = range_net(syn_200)
    sizing_vals = out_200["sizing"]
    if sizing_vals.dim() > 1:
        sizing_vals = sizing_vals.squeeze(-1)

    all_in_range = (sizing_vals >= 0).all() and (sizing_vals <= 2.0).all()
    std = sizing_vals.std().item()

    print(f"  Sizing range: [{sizing_vals.min().item():.4f}, {sizing_vals.max().item():.4f}]")
    print(f"  Sizing std: {std:.4f}")

    if all_in_range:
        print("  [PASS] — all in [0, 2]")
    else:
        print("  [FAIL] — values outside [0, 2]")
        PASS = False

    if std > 0.05:
        print("  [PASS] — non-degenerate distribution")
    else:
        print(f"  [FAIL] — std {std:.4f} <= 0.05")
        PASS = False

    del range_net
    print()

    # ═══════════════════════════════════════════════════════════════════════
    #  SECTION 7 — Regret head distribution
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("Section 7: Regret head distribution")
    print("=" * 60)

    with redirect_stdout(io.StringIO()):
        reg_net = DeepCFRNetwork(config)
    reg_net.eval()

    syn_r = _build_random_synthetic_input(config, batch_size=200)
    with torch.no_grad():
        out_r = reg_net(syn_r)
    regrets = out_r["regret"]

    has_nan = torch.isnan(regrets).any() or torch.isinf(regrets).any()
    if not has_nan:
        print("  No NaN/Inf [PASS]")
    else:
        print("  [FAIL] — NaN/Inf in regret outputs")
        PASS = False

    nonzero_count = (regrets.abs() > 1e-8).any(dim=1).sum().item()
    nonzero_pct = nonzero_count / 200.0
    print(f"  States with nonzero regrets: {nonzero_count}/200 ({nonzero_pct*100:.0f}%)")
    if nonzero_pct >= 0.80:
        print("  [PASS] — >= 80% have nonzero")
    else:
        print(f"  [FAIL] — only {nonzero_pct*100:.0f}%")
        PASS = False

    per_state_var = regrets.var(dim=1)
    low_var = (per_state_var < 0.001).sum().item()
    print(f"  States with var < 0.001: {low_var}/200")
    if low_var < 200 * 0.5:
        print("  [PASS] — not all constant")
    else:
        print("  [FAIL] — too many constant-output states")
        PASS = False

    del reg_net
    print()

    # ═══════════════════════════════════════════════════════════════════════
    #  SECTION 8 — Smoke tournament
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("Section 8: Smoke tournament exercises search path")
    print("=" * 60)

    try:
        tournaments_ok = 0
        total_search_calls = 0

        for t_idx in range(3):
            with redirect_stdout(io.StringIO()):
                deep_bot = DeepCFRBot(config=config, inference_mode=True, search_depth=2)
            deep_bot._weights_loaded = True
            seats = [
                Seat("P1", 300), Seat("P2", 300), Seat("P3", 300),
                Seat("P4", 300), Seat("P5", 300), Seat("P6", 300),
            ]
            bot_for = {"P1": deep_bot}
            for seat in seats[1:]:
                bot_for[seat.player_id] = InProcessBot(RandomBot())

            table = Table(rng=random.Random(12345 + t_idx))
            dealer_index = 0
            hand_count = 0
            try:
                with redirect_stdout(io.StringIO()):
                    while hand_count < 25 and sum(s.chips > 0 for s in seats) > 1:
                        active = [s for s in seats if s.chips > 0]
                        table.play_hand(
                            seats=active,
                            small_blind=5,
                            big_blind=10,
                            dealer_index=dealer_index % len(active),
                            bot_for={s.player_id: bot_for[s.player_id] for s in active},
                            on_event=None,
                        )
                        dealer_index += 1
                        hand_count += 1
                total_search_calls += deep_bot._subgame_search_calls
                if hand_count > 0:
                    tournaments_ok += 1
            except Exception as e:
                print(f"  Tournament {t_idx+1} failed: {e}")

        print(f"  Completed: {tournaments_ok}/3")
        print(f"  DeepCFR search calls: {total_search_calls}")
        if tournaments_ok == 3:
            print("  [PASS] — all 3 tournaments completed")
        else:
            print(f"  [FAIL] — only {tournaments_ok}/3 completed")
            PASS = False
        if total_search_calls > 0:
            print("  [PASS] — smoke exercised _subgame_search")
        else:
            print("  [FAIL] — smoke never called _subgame_search")
            PASS = False
    except Exception as e:
        print(f"  [FAIL] — could not run tournaments: {e}")
        PASS = False

    print()

    # ═══════════════════════════════════════════════════════════════════════
    #  SECTION 9 — _cfr_recurse exercises state and recurses
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("Section 9: _cfr_recurse exercises state and recurses")
    print("=" * 60)

    def _make_deep_state(hero_hole):
        board = [("K", "s"), ("7", "d"), ("2", "c")]
        opp_hole = [("7", "h"), ("2", "s")]
        used = set(hero_hole + opp_hole + board)
        deck = [(r, s) for r in "23456789TJQKA" for s in "cdhs" if (r, s) not in used]
        return _DeepCFRGameState(
            pot=100,
            stacks=[200, 200],
            committed_per_seat=[50, 50],
            alive=[True, True],
            street="flop",
            board=board,
            hole_cards={0: tuple(hero_hole), 1: tuple(opp_hole)},
            seat_order=[0, 1],
            action_idx=0,
            history_events=[],
            deck_remaining=deck,
            big_blind=10,
        )

    with redirect_stdout(io.StringIO()):
        recurse_bot = DeepCFRBot(config=config, inference_mode=False)
    state_ak = _make_deep_state([("A", "h"), ("K", "h")])
    state_72 = _make_deep_state([("7", "c"), ("2", "h")])

    try:
        random.seed(777)
        recurse_bot._recursion_calls = 0
        val_ak = recurse_bot._cfr_recurse(state_ak, hero_seat=0, depth=2)
        calls_d2 = recurse_bot._recursion_calls

        random.seed(777)
        val_72 = recurse_bot._cfr_recurse(state_72, hero_seat=0, depth=2)

        recurse_bot._recursion_calls = 0
        random.seed(888)
        val_d3 = recurse_bot._cfr_recurse(state_ak, hero_seat=0, depth=3)
        calls_d3 = recurse_bot._recursion_calls

        print(f"  AK value={val_ak:.6f}, 72o value={val_72:.6f}")
        print(f"  recursion calls: depth=2 -> {calls_d2}, depth=3 -> {calls_d3}")
        if abs(val_ak - val_72) > 1e-4:
            print("  [PASS] — recursion output depends on game state")
        else:
            print("  [FAIL] — recursion output did not change with state")
            PASS = False
        if calls_d2 >= 2 and calls_d3 >= 3:
            print("  [PASS] — recursion descends with depth")
        else:
            print("  [FAIL] — recursion call counts too low")
            PASS = False
        if all(math.isfinite(v) for v in (val_ak, val_72, val_d3)):
            print("  [PASS] — recursion outputs are finite")
        else:
            print("  [FAIL] — recursion produced NaN/Inf")
            PASS = False
    except Exception as e:
        print(f"  [FAIL] — _cfr_recurse raised {type(e).__name__}: {e}")
        PASS = False

    print()

    # ═══════════════════════════════════════════════════════════════════════
    #  SECTION 10 — Opponent features update across hands
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("Section 10: Opponent features update across hands")
    print("=" * 60)

    with redirect_stdout(io.StringIO()):
        stat_bot = DeepCFRBot(config=config, inference_mode=True)
    try:
        opening_view = _make_view(
            hole=[("A", "d"), ("Q", "d")],
            board=[], pot=15, to_call=10, street="preflop",
            n_opp=1, stacks={"P_hero": 500, "P_opp0": 500},
        )
        stat_bot.act(opening_view)
        initial_tensor = stat_bot._opp_stats.to_tensor(1)
        if float(initial_tensor.abs().sum()) == 0.0:
            print("  Fresh opponent tensor: all zeros [PASS]")
        else:
            print(f"  [FAIL] — fresh tensor unexpectedly nonzero: {initial_tensor}")
            PASS = False

        for hand_idx in range(8):
            hist = [{
                "street": "preflop",
                "pid": "P_opp0",
                "type": "raise",
                "amount": 30,
                "pot_before": 15,
            }]
            active_view = _make_view(
                hole=[("A", "d"), ("Q", "d")],
                board=[], pot=45, to_call=20, street="preflop",
                n_opp=1, stacks={"P_hero": 500, "P_opp0": 470},
            )
            active_view.history = hist
            stat_bot.act(active_view)

            boundary_view = _make_view(
                hole=[("K", "c"), ("Q", "c")],
                board=[], pot=15, to_call=10, street="preflop",
                n_opp=1, stacks={"P_hero": 500, "P_opp0": 500},
            )
            boundary_view.history = []
            stat_bot.act(boundary_view)

        updated_tensor = stat_bot._opp_stats.to_tensor(1)
        bucket = stat_bot._opp_stats.bucket(1)
        print(f"  Updated opponent tensor: {updated_tensor.tolist()}")
        print(f"  Seat-1 bucket: {bucket}")
        if float(updated_tensor.abs().sum()) > 0.0:
            print("  [PASS] — opponent tensor is live")
        else:
            print("  [FAIL] — opponent tensor stayed zero")
            PASS = False
        if bucket == "LA":
            print("  [PASS] — repeated raises bucket as LA")
        else:
            print("  [FAIL] — repeated raises did not bucket as LA")
            PASS = False

        feature_view = _make_view(
            hole=[("A", "s"), ("J", "s")],
            board=[], pot=45, to_call=20, street="preflop",
            n_opp=1, stacks={"P_hero": 500, "P_opp0": 470},
        )
        feature_batch = build_network_input(feature_view, opp_tracker=stat_bot._opp_stats)
        opp_feature_sum = float(feature_batch["opp_features"].abs().sum())
        if opp_feature_sum > 0.0:
            print("  [PASS] — encoder input receives opponent features")
        else:
            print("  [FAIL] — encoder opp_features path stayed zero")
            PASS = False
    except Exception as e:
        print(f"  [FAIL] — opponent feature test raised {type(e).__name__}: {e}")
        PASS = False

    print()

    # ═══════════════════════════════════════════════════════════════════════
    #  SECTION 11 — Search must not amplify 72o into all-in
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("Section 11: 72o search all-in collapse regression")
    print("=" * 60)

    try:
        with redirect_stdout(io.StringIO()):
            search_bot = DeepCFRBot(config=config, inference_mode=True, search_depth=4)
        search_bot._weights_loaded = True
        search_bot.training_iteration = 200_000  # all-in warmup no longer masks.

        trash_view = _make_view(
            hole=[("7", "h"), ("2", "c")],
            board=[], pot=15, to_call=10, street="preflop",
            position="UTG", n_opp=5,
            stacks={
                "P_hero": 500,
                "P_opp0": 500,
                "P_opp1": 500,
                "P_opp2": 500,
                "P_opp3": 500,
                "P_opp4": 500,
            },
        )
        legal_mask = _legal_abstract_actions(
            trash_view.legal_actions,
            trash_view.pot,
            street=trash_view.street,
            big_blind=10,
        )
        raw_policy = [0.0] * NUM_ACTIONS
        for a in legal_mask:
            raw_policy[a] = 1.0 / len(legal_mask)

        original_search_subtree = search_bot._search_subtree

        def biased_value_head(state, hero_seat, depth):
            """Reproduce the observed bad value ordering: all-in >> other actions."""
            if not state.history_events:
                return 0.0
            last = state.history_events[-1]
            if last.action == "all_in":
                return 115.0
            if last.action in ("bet", "raise"):
                return 88.0
            if last.action == "call":
                return 74.0
            if last.action == "fold":
                return 64.0
            return 70.0

        search_bot._search_subtree = biased_value_head
        refined = search_bot._subgame_search(
            trash_view, raw_policy, legal_mask, depth=4)
        search_bot._search_subtree = original_search_subtree

        all_in_idx = ABSTRACT_ACTIONS.index("all_in")
        raw_all_in = raw_policy[all_in_idx] if all_in_idx in legal_mask else 0.0
        refined_all_in = refined[all_in_idx] if all_in_idx in legal_mask else 0.0
        print(f"  legal={[(a, ABSTRACT_ACTIONS[a]) for a in legal_mask]}")
        print(f"  raw all_in={raw_all_in:.3f}")
        print(f"  search-refined all_in={refined_all_in:.3f}")
        if refined_all_in <= 0.50:
            print("  [PASS] — search did not amplify 72o to >50% all-in")
        else:
            print("  [FAIL] — search-refined policy put >50% on all-in")
            PASS = False
    except Exception as e:
        print(f"  [FAIL] — 72o search regression raised {type(e).__name__}: {e}")
        PASS = False

    print()

    # ═══════════════════════════════════════════════════════════════════════
    #  SECTION 12 — Hero action cost subtraction (P1)
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("Section 12: Hero action cost subtraction (P1)")
    print("=" * 60)

    try:
        with redirect_stdout(io.StringIO()):
            cost_bot = DeepCFRBot(config=config, inference_mode=True)
        cost_bot._weights_loaded = True
        cost_bot.training_iteration = 200_000  # disable all-in warmup mask

        # Heads-up river state: opp bet 50, hero (seat 0) is to act.
        used = {("A", "s"), ("K", "s"), ("Q", "s"), ("J", "s"), ("T", "s"),
                ("A", "h"), ("K", "h"), ("7", "c"), ("2", "c")}
        deck = [(r, s) for r in "23456789TJQKA"
                for s in "cdhs" if (r, s) not in used]
        cost_state = _DeepCFRGameState(
            pot=100,
            stacks=[200, 150],
            committed_per_seat=[0, 50],  # opp already bet 50 on the river
            alive=[True, True],
            street="river",
            board=[("A", "s"), ("K", "s"), ("Q", "s"), ("J", "s"), ("T", "s")],
            hole_cards={0: (("A", "h"), ("K", "h")),
                        1: (("7", "c"), ("2", "c"))},
            seat_order=[0, 1],
            action_idx=0,
            history_events=[],
            deck_remaining=deck,
            big_blind=10,
        )

        # Step 1: apply_action delta unit check — check_call (idx 1) cost == to_call.
        to_call = cost_state.to_call_for(0)
        next_call = cost_state.apply_action(0, 1)
        call_delta = next_call.committed_per_seat[0] - cost_state.committed_per_seat[0]
        if call_delta == to_call == 50:
            print(f"  [PASS] — apply_action(check_call) delta={call_delta} matches to_call=50")
        else:
            print(f"  [FAIL] — delta={call_delta}, to_call={to_call} (expected both 50)")
            PASS = False

        # Step 2: depth-1 _search_subtree must return max_a(V_a - cost_a). The leaf
        # at depth=0 returns the value head's deterministic output, so we can
        # reconstruct the expected max by computing each action's net value manually.
        hero_seat = 0
        legal_mask = cost_state.legal_abstract_actions()
        per_action = {}
        for a in legal_mask:
            before_commit = cost_state.committed_per_seat[hero_seat]
            next_state = cost_state.apply_action(hero_seat, a)
            cost = max(0, next_state.committed_per_seat[hero_seat] - before_commit)
            leaf_v = cost_bot._search_subtree(next_state, hero_seat, depth=0)
            per_action[a] = (leaf_v, cost, leaf_v - cost)
        expected_max = max(net for _, _, net in per_action.values())

        actual_max = cost_bot._search_subtree(cost_state, hero_seat, depth=1)

        print("  per-action (V_a, cost, V_a-cost):")
        for a, (v, c, net) in per_action.items():
            print(f"    {ABSTRACT_ACTIONS[a]:>10s}: V={v:+.4f} cost={c} net={net:+.4f}")
        print(f"  expected max_a(V_a - cost_a) = {expected_max:+.6f}")
        print(f"  actual _search_subtree(depth=1) = {actual_max:+.6f}")

        if abs(actual_max - expected_max) < 1e-4:
            print("  [PASS] — depth-1 hero branch returns max_a(V_a - cost_a)")
        else:
            print(f"  [FAIL] — diff = {abs(actual_max - expected_max):.6f}")
            PASS = False

        # Step 3: pre-fix this would equal max_a(V_a) (no subtraction). When
        # the gross-best action has positive cost, the post-fix value must be
        # strictly less than the gross max. If the gross-best action is free,
        # Step 2 above is the meaningful regression check.
        gross_max = max(v for v, _, _ in per_action.values())
        gross_best_costs = [
            c for v, c, _ in per_action.values()
            if abs(v - gross_max) < 1e-6
        ]
        gross_best_is_costly = any(c > 0 for c in gross_best_costs)
        if gross_best_is_costly and actual_max < gross_max - 1e-6:
            print(f"  [PASS] — net max ({actual_max:+.4f}) < gross max ({gross_max:+.4f}) — cost subtraction observable")
        elif gross_best_is_costly:
            print(f"  [FAIL] — net max == gross max ({actual_max:+.4f}); subtraction not taking effect")
            PASS = False
        else:
            print("  [PASS] — gross-best action is free; Step 2 confirms cost subtraction")

    except Exception as e:
        print(f"  [FAIL] — Section 12 raised {type(e).__name__}: {e}")
        PASS = False

    print()

    # ═══════════════════════════════════════════════════════════════════════
    #  SECTION 13 — Per-instance RNG independence (review finding fix)
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("Section 13: Per-instance RNG independence")
    print("=" * 60)

    try:
        with redirect_stdout(io.StringIO()):
            rng_a = DeepCFRBot(config=config, inference_mode=False)
            rng_b = DeepCFRBot(config=config, inference_mode=False)
        sa = rng_a._rng.random()
        sb = rng_b._rng.random()
        if sa != sb:
            print(f"  [PASS] — two DeepCFRBots have independent RNGs "
                  f"({sa:.6f} vs {sb:.6f})")
        else:
            print(f"  [FAIL] — bots collided on first sample (both {sa})")
            PASS = False

        # Bot-seeding determinism: same global seed → same bot RNG.
        random.seed(987)
        with redirect_stdout(io.StringIO()):
            sd_a = DeepCFRBot(config=config, inference_mode=False)
        x = sd_a._rng.random()
        random.seed(987)
        with redirect_stdout(io.StringIO()):
            sd_b = DeepCFRBot(config=config, inference_mode=False)
        y = sd_b._rng.random()
        if x == y:
            print(f"  [PASS] — same global seed → same bot RNG sample ({x:.6f})")
        else:
            print(f"  [FAIL] — seed cascade broken: {x:.6f} vs {y:.6f}")
            PASS = False
    except Exception as e:
        print(f"  [FAIL] — Section 13 raised {type(e).__name__}: {e}")
        PASS = False

    print()


def main():
    global PASS

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["small", "large", "both"],
                        default="both")
    args = parser.parse_args()

    variants = ["small", "large"] if args.variant == "both" else [args.variant]

    for v in variants:
        run_checks(v)

    print("=" * 60)
    if PASS:
        print("ALL CHECKS PASSED [PASS]")
    else:
        print("SOME CHECKS FAILED [FAIL]")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
