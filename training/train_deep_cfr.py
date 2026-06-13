"""
training/train_deep_cfr.py — Deep CFR Plus training loop
---------------------------------------------------------
Wraps _cfr_recurse with target collection, periodic gradient steps,
and atomic checkpoint save/load.

Usage:
    python training/train_deep_cfr.py --variant small --iterations 50 \
        --update-interval 10 --checkpoint-interval 25 --batch-size 32 \
        --aivat-sims 50 --save-path /tmp/deep_cfr_smoke.pt
    python training/train_deep_cfr.py --variant large --iterations 1000000 \\
        --save-path models/deep_cfr_v1.pt
"""
from __future__ import annotations

import argparse
import math
import os
import random
import signal
import sys
import time as _time

# Add project root so imports work from any CWD.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn

from bots.deep_cfr_bot import (
    DeepCFRBot, DeepCFRConfig, DeepCFRNetwork,
    ReservoirBuffer, _DeepCFRGameState, _FULL_DECK,
)
from core.action_history import ActionEvent
from core.bot_api import PlayerView
from core.table_order import street_action_order

# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

N_SEATS = 6
START_CHIPS = 1000
BIG_BLIND = 10
SMALL_BLIND = 5

# Training-state curriculum (Fix 5 / I6).  Tournaments are decided heads-up,
# short-stacked, at escalated blinds — states a fixed "6 seats x 1000 chips"
# trainer never samples.  Each training state now randomizes the player count
# and the stack depths; blinds stay at 5/10 because the network features are
# big-blind-relative (see core.action_history.REF_BIG_BLIND), which makes the
# blind level a pure scale factor — varying chip depth at fixed blinds covers
# escalated blinds at fixed chips too.
CURRICULUM_PLAYER_COUNTS = (2, 3, 4, 5, 6)
CURRICULUM_MIN_DEPTH_BB = 10     # shortest stack: 10 big blinds
CURRICULUM_MAX_DEPTH_BB = 200    # deepest stack: 200 big blinds
CURRICULUM_SHARED_DEPTH_PROB = 0.5  # 50% one depth for all seats, 50% per-seat
ALL_IN_WARMUP_ITERATIONS = 100_000
ALL_IN_DEPLOY_ITERATION = 150_000
ALL_IN_FULL_RELEASE_ITERATION = 350_000
CANARY_PASS_SEARCH_MAX = 0.15
CANARY_PASS_RAW_MAX = 0.30
CANARY_FAIL_SEARCH_MIN = 0.35
CANARY_FAIL_RAW_MIN = 0.60
# Additional health-metric canary bounds, layered ON TOP of the raw/search
# all-in canary above (mirrors probe_deep_cfr.py --fail-on-unhealthy).  A metric
# WARNs at its *_WARN level (side checkpoint only) and FAILs/aborts at its *_FAIL
# level; the overall checkpoint status is the worst of the all-in canary and
# these.  PFR / strong-all-in / strong-continue are fractions in [0,1]; avg-raise
# is in x-pot.
#
# All of these except strong-continue are "high is bad" (trip when value >= the
# level).  ``strong_continue`` is the lone "low is bad" gate (the fold-collapse
# signature): it WARNs when continue < WARN and FAILs when continue < FAIL.  Note
# the WARN level is the HIGHER number for this metric — anything below 80% is at
# least a WARN, anything below 60% is a FAIL.
CANARY_PFR_WARN = 0.40
CANARY_PFR_FAIL = 0.55
CANARY_AVG_RAISE_WARN = 10.0
CANARY_AVG_RAISE_FAIL = 25.0
CANARY_STRONG_ALL_IN_WARN = 0.25
CANARY_STRONG_ALL_IN_FAIL = 0.45
CANARY_STRONG_CONTINUE_WARN = 0.80  # continue < this -> at least WARN
CANARY_STRONG_CONTINUE_FAIL = 0.60  # continue < this -> FAIL (fold collapse)
# Finiteness guard (B2/I5): a non-finite total loss skips the optimizer step.
# Skipping forever is its own silent failure mode (the run "trains" while no
# parameter ever moves), so this many CONSECUTIVE skips abort the run with a
# nonzero exit.  The consecutive counter resets on any successful finite step.
NONFINITE_SKIPS_ABORT_THRESHOLD = 50


def epsilon_for_iteration(t: int, total: int) -> float:
    """Anneal training-time opponent exploration from 0.30 to 0.05."""
    progress = min(1.0, t / max(total, 1))
    return 0.30 - 0.25 * progress


def allow_all_in_for_iteration(t: int, warmup_iterations: int) -> bool:
    """Return whether iteration ``t`` may expose all-in to self-play."""
    return t >= max(0, int(warmup_iterations))


def all_in_policy_probability_for_iteration(
    t: int,
    warmup_iterations: int,
    full_release_iteration: int,
) -> float:
    """Staged all-in self-play exposure probability for iteration ``t``."""
    warmup = max(0, int(warmup_iterations))
    full = max(warmup + 1, int(full_release_iteration))
    if t < warmup:
        return 0.0
    if t >= full:
        return 1.0
    return max(0.0, min(1.0, (t - warmup) / (full - warmup)))


def all_in_phase_for_iteration(
    t: int,
    warmup_iterations: int,
    full_release_iteration: int,
) -> str:
    """Human-readable all-in curriculum phase."""
    if t < max(0, int(warmup_iterations)):
        return "shadow"
    if all_in_policy_probability_for_iteration(
        t,
        warmup_iterations,
        full_release_iteration,
    ) < 1.0:
        return "staged"
    return "full"


# ═══════════════════════════════════════════════════════════════════════════════
#  Device selection
# ═══════════════════════════════════════════════════════════════════════════════

def pick_device(arg: str) -> torch.device:
    if arg == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════════════
#  Build initial game state
# ═══════════════════════════════════════════════════════════════════════════════

def build_initial_state(n_seats: int = N_SEATS, hero_seat: int = 0,
                        stacks: list[int] | None = None) -> _DeepCFRGameState:
    """Construct a fresh preflop _DeepCFRGameState with random hole cards.

    Seat 0 is the button and ``ring_order`` runs clockwise from it, exactly
    like the engine's dealer-first ring.  ``stacks`` optionally sets per-seat
    chip counts (curriculum); the default is the classic equal START_CHIPS.

    Blind seats mirror core.engine.Table.play_hand: heads-up the BUTTON posts
    the small blind (and acts first preflop; the BB acts first postflop);
    with 3+ players seat 1 posts the SB and seat 2 the BB.
    """
    deck = list(_FULL_DECK)
    random.shuffle(deck)

    hole_cards = {}
    for seat in range(n_seats):
        hole_cards[seat] = (deck.pop(), deck.pop())

    if stacks is None:
        stacks = [START_CHIPS] * n_seats
    else:
        if len(stacks) != n_seats:
            raise ValueError(
                f"stacks has {len(stacks)} entries for n_seats={n_seats}")
        stacks = [int(s) for s in stacks]
    committed = [0] * n_seats

    # Post blinds (engine convention; min() handles stacks below the blind)
    if n_seats == 2:
        sb_seat, bb_seat = 0, 1   # heads-up: the button is the small blind
    else:
        sb_seat, bb_seat = 1, 2
    sb_amt = min(SMALL_BLIND, stacks[sb_seat])
    bb_amt = min(BIG_BLIND, stacks[bb_seat])
    stacks[sb_seat] -= sb_amt
    committed[sb_seat] = sb_amt
    stacks[bb_seat] -= bb_amt
    committed[bb_seat] = bb_amt
    pot = sb_amt + bb_amt

    # Preflop action order from the shared engine helper: UTG first for 3+
    # players (seat 3 clockwise from the button), button/SB first heads-up.
    # A seat already all-in from posting its blind cannot act — the engine
    # skips such seats without a history entry, so they are excluded here.
    ring = list(range(n_seats))
    seat_order = [s for s in street_action_order("preflop", ring)
                  if stacks[s] > 0]

    return _DeepCFRGameState(
        pot=pot,
        stacks=stacks,
        committed_per_seat=committed,
        alive=[True] * n_seats,
        street="preflop",
        board=[],
        hole_cards=hole_cards,
        seat_order=seat_order,
        action_idx=0,
        history_events=[],
        deck_remaining=deck,
        big_blind=BIG_BLIND,
        ring_order=ring,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Training-state curriculum (Fix 5 / I6)
# ═══════════════════════════════════════════════════════════════════════════════

def sample_stack_depth_chips(rng=random) -> int:
    """One stack depth in CHIPS, log-uniform between 10 BB and 200 BB.

    Log-uniform (rather than uniform) gives short-stack regimes — where
    tournaments are actually decided — as much sample mass as deep ones.
    """
    log_lo = math.log(CURRICULUM_MIN_DEPTH_BB)
    log_hi = math.log(CURRICULUM_MAX_DEPTH_BB)
    depth_bb = math.exp(rng.uniform(log_lo, log_hi))
    return max(1, int(round(depth_bb * BIG_BLIND)))


def sample_curriculum_stacks(n_seats: int, rng=random) -> list:
    """Per-seat chip stacks: 50% one shared depth, 50% independent depths."""
    if rng.random() < CURRICULUM_SHARED_DEPTH_PROB:
        depth = sample_stack_depth_chips(rng)
        return [depth] * n_seats
    return [sample_stack_depth_chips(rng) for _ in range(n_seats)]


def sample_curriculum_state(iteration: int, rng=random):
    """Random training state matching the eval distribution (I6).

    Returns ``(state, hero_seat)``.  Player count is uniform over 2..6,
    stacks are 10-200 BB log-uniform (shared or per-seat), blinds stay 5/10
    (the features are BB-relative, so blind level is a pure scale factor),
    and the hero seat keeps rotating with the iteration index.
    """
    n_seats = rng.choice(CURRICULUM_PLAYER_COUNTS)
    hero_seat = (iteration - 1) % n_seats   # same rotation rule as before
    stacks = sample_curriculum_stacks(n_seats, rng)
    state = build_initial_state(n_seats=n_seats, hero_seat=hero_seat,
                                stacks=stacks)
    return state, hero_seat


# ═══════════════════════════════════════════════════════════════════════════════
#  Training step
# ═══════════════════════════════════════════════════════════════════════════════

def _stack_input_dicts(samples_inputs: list, device: torch.device) -> dict:
    """Stack a list of single-row input dicts into one batched dict."""
    keys = samples_inputs[0].keys()
    return {
        k: torch.cat([d[k] for d in samples_inputs], dim=0).to(device)
        for k in keys
    }


def train_step(
    network: DeepCFRNetwork,
    optimizer: torch.optim.Optimizer,
    regret_buf: ReservoirBuffer,
    value_buf: ReservoirBuffer,
    sizing_buf: ReservoirBuffer,
    batch_size: int,
    device: torch.device,
) -> tuple:
    """One gradient step on ready heads.

    Returns (r_loss, v_loss, s_loss, info), where info records which buffers
    contributed to the optimizer step.
    """
    network.train()

    r_loss_val = 0.0
    v_loss_val = 0.0
    s_loss_val = 0.0
    heads_trained = {"regret": False, "value": False, "sizing": False}

    # ── Regret loss ──
    if len(regret_buf) >= batch_size:
        samples = regret_buf.sample(batch_size)
        inputs = [s[0] for s in samples]
        targets = torch.stack([s[1] for s in samples]).to(device)
        masks = torch.stack([s[2] for s in samples]).to(device)
        weights = torch.tensor([s[3] for s in samples], dtype=torch.float32, device=device)
        # Normalize weights
        weights = weights / (weights.sum() + 1e-8)

        batch = _stack_input_dicts(inputs, device)
        out = network(batch)
        preds = out["regret"] * masks
        targets = targets * masks
        # SmoothL1 only on legal actions, averaged per sample, then CFR+
        # weighted across samples. This keeps large initial regret targets
        # from dominating the optimizer with quadratic gradients.
        elementwise = nn.functional.smooth_l1_loss(
            preds, targets, reduction="none")
        elementwise = elementwise * masks
        n_legal = masks.sum(dim=1).clamp(min=1.0)
        per_sample = elementwise.sum(dim=1) / n_legal
        r_loss = (per_sample * weights).sum()
        r_loss_val = r_loss.item()
        heads_trained["regret"] = True
    else:
        r_loss = torch.tensor(0.0, device=device)

    # ── Value loss ──
    if len(value_buf) >= batch_size:
        samples = value_buf.sample(batch_size)
        inputs = [s[0] for s in samples]
        targets = torch.tensor([s[1] for s in samples], dtype=torch.float32, device=device)

        batch = _stack_input_dicts(inputs, device)
        out = network(batch)
        pred = out["value"]
        v_loss = nn.functional.smooth_l1_loss(pred, targets)
        v_loss_val = v_loss.item()
        heads_trained["value"] = True
    else:
        v_loss = torch.tensor(0.0, device=device)

    # ── Sizing loss ──
    if len(sizing_buf) >= min(batch_size, 16):
        n = min(batch_size, len(sizing_buf))
        samples = sizing_buf.sample(n)
        inputs = [s[0] for s in samples]
        targets = torch.tensor([s[1] for s in samples], dtype=torch.float32, device=device)

        batch = _stack_input_dicts(inputs, device)
        out = network(batch)
        pred = out["sizing"]
        s_loss = nn.functional.mse_loss(pred, targets)
        s_loss_val = s_loss.item()
        heads_trained["sizing"] = True
    else:
        s_loss = torch.tensor(0.0, device=device)

    # ── Combined backward ──
    total_loss = r_loss + v_loss + s_loss
    optimizer.zero_grad()
    did_step = bool(total_loss.requires_grad)

    # Finiteness guard (B2/I5): clip_grad_norm_ does NOT protect against NaN —
    # one NaN gradient makes the total norm NaN, the clip coefficient NaN, and
    # then optimizer.step() writes NaN into EVERY parameter, silently
    # corrupting all later checkpoints.  On a non-finite loss we skip the
    # optimizer step entirely (parameters stay untouched) and report the skip
    # so the training loop can count and, past a threshold, abort.
    nonfinite_skip = False
    if did_step and not bool(torch.isfinite(total_loss).item()):
        nonfinite_skip = True
        did_step = False

    if did_step:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=10.0)
        optimizer.step()

    return r_loss_val, v_loss_val, s_loss_val, {
        "heads_trained": heads_trained,
        "did_step": did_step,
        "nonfinite_skip": nonfinite_skip,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Checkpoint save/load (atomic via .tmp + os.replace)
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(path: str, iteration: int, bot: DeepCFRBot,
                    optimizer: torch.optim.Optimizer, losses: dict,
                    *, nonfinite_skips: int = 0,
                    shadow_only: bool | None = None):
    """Atomic save of training state.

    ``nonfinite_skips`` is the cumulative count of optimizer steps skipped by
    the finiteness guard (B2/I5) — persisted so an operator inspecting a
    checkpoint can see whether the run hit numerical trouble.

    ``shadow_only`` is stamped (as ``True``) only on FINAL checkpoints whose
    ``iteration`` is below the all-in deploy gate — inference will permanently
    mask all-in for such a model (B4/I9).  ``None`` omits the key (mid-run
    checkpoints, where being below the gate is normal and transient).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    payload = {
        "iteration": iteration,
        "config": bot.config,
        "network_state_dict": bot.network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "all_in_warmup_iterations": bot.all_in_warmup_iterations,
        "all_in_deploy_iteration": bot.all_in_deploy_iteration,
        "all_in_full_release_iteration": bot.all_in_full_release_iteration,
        "losses": {k: v[-100:] for k, v in losses.items()},  # last 100 only
        "nonfinite_skips": int(nonfinite_skips),
    }
    if shadow_only is not None:
        payload["shadow_only"] = bool(shadow_only)
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def safe_checkpoint_path(path: str) -> str:
    """Return the stable safe-checkpoint path beside ``path``."""
    root, ext = os.path.splitext(path)
    return f"{root}.safe{ext or '.pt'}"


def warn_checkpoint_path(path: str, iteration: int) -> str:
    """Return the side checkpoint path for a canary warning."""
    root, ext = os.path.splitext(path)
    return f"{root}.warn_{iteration}{ext or '.pt'}"


def classify_canary(raw_all_in: float, search_all_in: float) -> str:
    """Classify checkpoint health from all-in canary frequencies."""
    if search_all_in > CANARY_FAIL_SEARCH_MIN or raw_all_in > CANARY_FAIL_RAW_MIN:
        return "FAIL"
    if search_all_in <= CANARY_PASS_SEARCH_MAX and raw_all_in <= CANARY_PASS_RAW_MAX:
        return "PASS"
    return "WARN"


_CANARY_STATUS_RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}


def _worst_canary_status(a: str, b: str) -> str:
    """Return the more severe of two PASS/WARN/FAIL statuses."""
    return a if _CANARY_STATUS_RANK.get(a, 0) >= _CANARY_STATUS_RANK.get(b, 0) else b


def classify_extra_canary_metrics(metrics: dict) -> tuple[str, list[str], list[str]]:
    """Classify the PFR / avg-raise / strong-all-in / strong-continue metrics.

    Returns ``(status, failed, warned)`` where ``status`` is PASS/WARN/FAIL and
    ``failed`` / ``warned`` are human-readable label strings for the
    ``[CANARY]`` / ``[WARN]`` log lines and the abort message.

    Each spec carries a ``direction``: ``"high"`` metrics trip when value >= the
    threshold (the all-in-collapse signature); the lone ``"low"`` metric,
    ``strong_continue``, trips when value < the threshold (the fold-collapse
    signature — AA/KK/AKs folded).  The failed/warned reason strings name the
    specific metric (e.g. ``strong_continue``) so an abort can be attributed to
    fold collapse vs. all-in collapse.

    Missing keys default to a HEALTHY value so a probe that returns only the
    legacy two-key ``{raw_all_in, search_all_in}`` dict (e.g. the monkeypatched
    probes in sanity_train_deep_cfr / sanity_train_deep_cfr_abort /
    sanity_review_findings) classifies as PASS on these added metrics and the
    all-in canary alone decides the outcome.  For the high-is-bad metrics the
    healthy default is 0.0; for strong_continue (low-is-bad) it is 1.0 — a 0.0
    default would spuriously FAIL every legacy probe.
    """
    specs = [
        ("preflop_pfr", metrics.get("preflop_pfr", 0.0),
         CANARY_PFR_WARN, CANARY_PFR_FAIL, "pct", "high"),
        ("avg_preflop_raise", metrics.get("preflop_avg_raise", 0.0),
         CANARY_AVG_RAISE_WARN, CANARY_AVG_RAISE_FAIL, "x", "high"),
        ("strong_all_in", metrics.get("strong_all_in", 0.0),
         CANARY_STRONG_ALL_IN_WARN, CANARY_STRONG_ALL_IN_FAIL, "pct", "high"),
        ("strong_continue", metrics.get("strong_continue", 1.0),
         CANARY_STRONG_CONTINUE_WARN, CANARY_STRONG_CONTINUE_FAIL, "pct", "low"),
    ]
    failed: list[str] = []
    warned: list[str] = []
    for label, value, warn_t, fail_t, unit, direction in specs:
        if unit == "pct":
            shown, warn_s, fail_s = (f"{value:.1%}", f"{warn_t:.0%}", f"{fail_t:.0%}")
        else:
            shown, warn_s, fail_s = (f"{value:.1f}x", f"{warn_t:.0f}x", f"{fail_t:.0f}x")
        if direction == "high":
            if value >= fail_t:
                failed.append(f"{label}={shown} (>= {fail_s})")
            elif value >= warn_t:
                warned.append(f"{label}={shown} (>= {warn_s})")
        else:  # low is bad
            if value < fail_t:
                failed.append(f"{label}={shown} (< {fail_s})")
            elif value < warn_t:
                warned.append(f"{label}={shown} (< {warn_s})")
    if failed:
        return "FAIL", failed, warned
    if warned:
        return "WARN", failed, warned
    return "PASS", failed, warned


def decide_canary_status(canary: dict, iteration: int,
                         deploy_iteration: int) -> tuple:
    """Combine the all-in canary and the extra health metrics into one status.

    Every metric is only ENFORCED once the model is mature enough to make the
    probe meaningful: ``iteration >= deploy_iteration`` (the same boundary the
    all-in row uses to leave the inference mask).  Before that, metrics are
    computed and reported but do NOT change the status.  In particular, the
    all-in output row is intentionally excluded from regret targets during the
    shadow phase, so shared-encoder updates can make its untrained raw score
    transiently dominate.  Enforcing that score caused seed-dependent aborts
    at the first checkpoint even though inference still masked all-in.

    Returns ``(status, base_status, extra_status, extra_failed, extra_warned,
    metrics_enforced)``.
    """
    base_status = classify_canary(
        canary.get("raw_all_in", 0.0), canary.get("search_all_in", 0.0))
    extra_status, extra_failed, extra_warned = classify_extra_canary_metrics(canary)
    metrics_enforced = iteration >= deploy_iteration
    diagnostic_status = _worst_canary_status(base_status, extra_status)
    status = diagnostic_status if metrics_enforced else "PASS"
    return (status, base_status, extra_status, extra_failed, extra_warned,
            metrics_enforced)


def save_promoted_checkpoint(path: str, iteration: int, bot: DeepCFRBot,
                             optimizer: torch.optim.Optimizer, losses: dict,
                             status: str, *, nonfinite_skips: int = 0,
                             shadow_only: bool | None = None) -> str:
    """Save according to canary promotion rules and return the written path."""
    extra = {"nonfinite_skips": nonfinite_skips, "shadow_only": shadow_only}
    if status == "PASS":
        save_checkpoint(path, iteration, bot, optimizer, losses, **extra)
        save_checkpoint(safe_checkpoint_path(path), iteration, bot, optimizer,
                        losses, **extra)
        return path
    if status == "WARN":
        side_path = warn_checkpoint_path(path, iteration)
        save_checkpoint(side_path, iteration, bot, optimizer, losses, **extra)
        return side_path
    raise RuntimeError(f"cannot save checkpoint for status {status!r}")


def load_checkpoint(path: str, bot: DeepCFRBot,
                    optimizer: torch.optim.Optimizer,
                    meta_out: dict | None = None) -> int:
    """Restore training state. Returns iteration number.

    When ``meta_out`` is given, run-level metadata that does not belong on the
    bot (currently the cumulative ``nonfinite_skips`` counter) is copied into
    it so a resumed run can keep counting where the prior run stopped.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if meta_out is not None:
        meta_out["nonfinite_skips"] = int(ckpt.get("nonfinite_skips", 0) or 0)
    bot.network.load_state_dict(ckpt["network_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    bot.all_in_warmup_iterations = int(
        ckpt.get("all_in_warmup_iterations", bot.all_in_warmup_iterations)
    )
    bot.all_in_deploy_iteration = int(
        ckpt.get("all_in_deploy_iteration", bot.all_in_deploy_iteration)
    )
    bot.all_in_full_release_iteration = int(
        ckpt.get(
            "all_in_full_release_iteration",
            bot.all_in_full_release_iteration,
        )
    )
    iteration = ckpt["iteration"]
    print(f"[train_deep_cfr] Resumed from {path} at iteration {iteration}")
    print(
        f"[Resume] Loaded checkpoint from iteration {iteration}.\n"
        f"  Note: replay buffers (regret/value/sizing) are NOT persisted in\n"
        f"  checkpoints. Buffers will refill from scratch over the next ~1000\n"
        f"  iterations before gradient steps resume at full batch size."
    )
    return iteration


def clear_all_in_optimizer_state(
    optimizer: torch.optim.Optimizer,
    bot: DeepCFRBot,
    all_in_idx: int,
) -> None:
    """Clear Adam momentum for the detoxed all-in regret output row."""
    final_linear = None
    for module in bot.network.regret_head.mlp.modules():
        if isinstance(module, nn.Linear):
            final_linear = module
    if final_linear is None:
        return

    for param, row_mode in (
        (final_linear.weight, True),
        (final_linear.bias, False),
    ):
        if param is None or param not in optimizer.state:
            continue
        for value in optimizer.state[param].values():
            if not torch.is_tensor(value):
                continue
            if row_mode and value.dim() >= 2 and value.shape[0] > all_in_idx:
                value[all_in_idx].zero_()
            elif not row_mode and value.dim() >= 1 and value.shape[0] > all_in_idx:
                value[all_in_idx].zero_()


# ═══════════════════════════════════════════════════════════════════════════════
#  Collapse canary
# ═══════════════════════════════════════════════════════════════════════════════

def _canary_is_all_in(action, view: PlayerView) -> bool:
    if action.type == "all_in":
        return True
    if action.type not in ("bet", "raise"):
        return False
    return action.amount is not None and action.amount >= view.max_raise


_CANARY_POSITIONS = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
_CANARY_STRONG_POSITIONS = ["CO", "BTN"]
# Premium hands for the strong-hand all-in canary (mirrors probe_deep_cfr.py).
_CANARY_STRONG_HANDS = [
    [("A", "h"), ("A", "s")],
    [("K", "h"), ("K", "s")],
    [("A", "h"), ("K", "h")],
]


def _canary_preflop_view(hole, position: str) -> PlayerView:
    """Build the standard open-raise-or-fold preflop spot used by the canary."""
    opponents = [f"opp{i}" for i in range(1, N_SEATS)]
    stacks = {"hero": START_CHIPS}
    for opp in opponents:
        stacks[opp] = START_CHIPS
    legal = [
        {"type": "fold"},
        {"type": "call"},
        {"type": "raise", "min": BIG_BLIND * 2, "max": START_CHIPS},
    ]
    return PlayerView(
        me="hero",
        street="preflop",
        position=position,
        hole_cards=list(hole),
        board=[],
        pot=SMALL_BLIND + BIG_BLIND,
        to_call=BIG_BLIND,
        min_raise=BIG_BLIND * 2,
        max_raise=START_CHIPS,
        legal_actions=legal,
        stacks=stacks,
        opponents=opponents,
        history=[],
    )


def _canary_views(n: int, seed: int) -> list[PlayerView]:
    rng = random.Random(seed)
    views = []
    for _ in range(n):
        deck = list(_FULL_DECK)
        rng.shuffle(deck)
        hole = [deck.pop(), deck.pop()]
        # rng order preserved (shuffle then choice) so raw/search all-in
        # frequencies are byte-for-byte identical to the pre-refactor canary.
        views.append(_canary_preflop_view(hole, rng.choice(_CANARY_POSITIONS)))
    return views


def _canary_strong_views(n: int, seed: int) -> list[PlayerView]:
    """Deterministic AA/KK/AKs spots for the strong-hand all-in metric."""
    rng = random.Random(seed)
    views = []
    for i in range(n):
        hole = _CANARY_STRONG_HANDS[i % len(_CANARY_STRONG_HANDS)]
        views.append(_canary_preflop_view(hole, rng.choice(_CANARY_STRONG_POSITIONS)))
    return views


def _canary_collect_stats(
    bot: DeepCFRBot,
    views: list[PlayerView],
    *,
    search_depth: int,
    seed: int,
    current_iteration: int,
) -> dict:
    """Run the bot over ``views`` under deterministic inference settings.

    Returns ``{"total", "all_in", "pfr", "continue", "sizes"}``.  All bot inference flags,
    the opponent tracker, the module RNG AND the bot's per-instance RNG
    (``bot._rng``, which drives action sampling) are saved and restored, so
    calling this never perturbs the surrounding training loop.  Both RNGs are
    also seeded from ``seed`` so the probe is fully reproducible for a given
    network — without seeding bot._rng, the sampled actions would still depend
    on the training loop's RNG state at checkpoint time.  ``pfr`` counts any
    bet/raise/all-in (matching probe_deep_cfr); ``continue`` counts any non-fold
    action (check/call/bet/raise/all_in — the fold-collapse complement); ``sizes``
    collects bet/raise amounts as a fraction of pot (matching probe_deep_cfr's
    avg-size).
    """
    old_training = bot.network.training
    old_inference_mode = bot.inference_mode
    old_weights_loaded = bot._weights_loaded
    old_search_depth = bot.search_depth
    old_training_iteration = bot.training_iteration
    old_guardrails_disabled = bot._all_in_guardrails_disabled
    old_opp_stats = bot._opp_stats
    old_history_len = bot._last_history_len
    old_history_snapshot = bot._last_history_snapshot
    old_last_hand_id = bot._last_hand_id
    old_random_state = random.getstate()
    old_bot_rng_state = bot._rng.getstate()

    stats = {"total": 0, "all_in": 0, "pfr": 0, "continue": 0, "sizes": []}
    try:
        random.seed(seed)
        bot._rng.seed(seed)
        bot.network.eval()
        bot.inference_mode = True
        bot._weights_loaded = True
        bot.search_depth = search_depth
        bot.training_iteration = current_iteration
        bot._all_in_guardrails_disabled = True
        bot._opp_stats = None
        bot._last_history_len = 0
        bot._last_history_snapshot = []

        for view in views:
            action = bot.act(view)
            stats["total"] += 1
            if _canary_is_all_in(action, view):
                stats["all_in"] += 1
            if action.type in ("bet", "raise", "all_in"):
                stats["pfr"] += 1
            if action.type != "fold":
                stats["continue"] += 1
            if action.type in ("bet", "raise") and action.amount is not None:
                stats["sizes"].append(action.amount / max(view.pot, 1))
    finally:
        random.setstate(old_random_state)
        bot._rng.setstate(old_bot_rng_state)
        bot.inference_mode = old_inference_mode
        bot._weights_loaded = old_weights_loaded
        bot.search_depth = old_search_depth
        bot.training_iteration = old_training_iteration
        bot._all_in_guardrails_disabled = old_guardrails_disabled
        bot._opp_stats = old_opp_stats
        bot._last_history_len = old_history_len
        bot._last_history_snapshot = old_history_snapshot
        bot._last_hand_id = old_last_hand_id
        if old_training:
            bot.network.train()
        else:
            bot.network.eval()

    return stats


def quick_canary_probe(bot: DeepCFRBot, device: torch.device,
                       n: int = 50, seed: int = 20260428,
                       current_iteration: int = 0) -> dict[str, float]:
    """Return the checkpoint health metrics for the collapse canary.

    Keys: ``raw_all_in`` / ``search_all_in`` (the original all-in frequencies,
    numerically unchanged — same views and seeds as before), plus ``preflop_pfr``
    and ``preflop_avg_raise`` (raw policy over the random preflop spots) and
    ``strong_all_in`` / ``strong_continue`` (raw policy over AA/KK/AKs spots;
    ``strong_continue`` is the non-fold rate — the fold-collapse gate).
    """
    _ = device  # kept for call-site/API clarity; bot.act handles device moves.
    views = _canary_views(n, seed)
    raw = _canary_collect_stats(
        bot, views, search_depth=0, seed=seed + 1,
        current_iteration=current_iteration,
    )
    search = _canary_collect_stats(
        bot, views, search_depth=bot.search_depth, seed=seed + 2,
        current_iteration=current_iteration,
    )
    strong = _canary_collect_stats(
        bot, _canary_strong_views(n, seed), search_depth=0, seed=seed + 3,
        current_iteration=current_iteration,
    )
    n_raw = max(raw["total"], 1)
    n_strong = max(strong["total"], 1)
    avg_raise = sum(raw["sizes"]) / len(raw["sizes"]) if raw["sizes"] else 0.0
    return {
        "raw_all_in": raw["all_in"] / n_raw,
        "search_all_in": search["all_in"] / max(search["total"], 1),
        "preflop_pfr": raw["pfr"] / n_raw,
        "preflop_avg_raise": avg_raise,
        "strong_all_in": strong["all_in"] / n_strong,
        "strong_continue": strong["continue"] / n_strong,
    }


def format_canary_metrics(canary: dict) -> str:
    """Render the one-line canary metrics summary used by the live
    ``[CANARY]`` / ``[WARN]`` / abort log lines.

    Pure function of the probe dict so the rendering — which MUST surface
    ``strong_continue`` (the fold-collapse %) alongside the all-in metrics — is
    unit-testable without a training run.  ``strong_continue`` defaults to 1.0
    (healthy) when absent so a legacy two-key probe still renders cleanly.
    """
    return (
        f"search={canary.get('search_all_in', 0.0):.1%}, "
        f"raw={canary.get('raw_all_in', 0.0):.1%}, "
        f"PFR={canary.get('preflop_pfr', 0.0):.1%}, "
        f"avg_raise={canary.get('preflop_avg_raise', 0.0):.1f}x, "
        f"strong_all_in={canary.get('strong_all_in', 0.0):.1%}, "
        f"strong_continue={canary.get('strong_continue', 1.0):.1%}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Progress printing
# ═══════════════════════════════════════════════════════════════════════════════

def print_progress(t: int, regret_buf, value_buf, sizing_buf, losses,
                   elapsed: float = 0.0):
    r = losses["regret"][-1] if losses["regret"] else 0.0
    v = losses["value"][-1] if losses["value"] else 0.0
    s = losses["sizing"][-1] if losses["sizing"] else 0.0
    rate = t / elapsed if elapsed > 0 else 0
    print(
        f"  iter={t:>7}  "
        f"bufs=({len(regret_buf)},{len(value_buf)},{len(sizing_buf)})  "
        f"loss=(r={r:.4f}, v={v:.4f}, s={s:.4f})  "
        f"rate={rate:.1f} it/s",
        flush=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Main training loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_training(args) -> dict:
    """Core training loop. Returns dict with final state info."""
    device = pick_device(args.device)
    config = DeepCFRConfig.small() if args.variant == "small" else DeepCFRConfig.large()

    bot = DeepCFRBot(config=config, inference_mode=False,
                     aivat_sims=args.aivat_sims)
    bot.all_in_warmup_iterations = args.all_in_warmup_iterations
    bot.all_in_deploy_iteration = args.all_in_deploy_iteration
    bot.all_in_full_release_iteration = args.all_in_full_release_iteration
    bot.network.to(device)
    bot.network.train()

    optimizer = torch.optim.Adam(bot.network.parameters(), lr=args.lr)

    regret_buf = ReservoirBuffer(capacity=1_000_000)
    value_buf = ReservoirBuffer(capacity=1_000_000)
    sizing_buf = ReservoirBuffer(capacity=1_000_000)

    # Cumulative + consecutive counters for the finiteness guard (B2/I5).
    # "total" persists across resumes via checkpoint metadata; "consecutive"
    # resets on any successful finite optimizer step.
    nonfinite = {"total": 0, "consecutive": 0}

    start_iter = 0
    if args.resume and os.path.exists(args.resume):
        resume_meta: dict = {}
        start_iter = load_checkpoint(args.resume, bot, optimizer,
                                     meta_out=resume_meta)
        nonfinite["total"] = int(resume_meta.get("nonfinite_skips", 0))
        bot.all_in_warmup_iterations = args.all_in_warmup_iterations
        bot.all_in_deploy_iteration = args.all_in_deploy_iteration
        bot.all_in_full_release_iteration = args.all_in_full_release_iteration
        if args.detox_all_in_on_resume:
            layer_name, all_in_idx = bot.detox_all_in_regret_output()
            clear_all_in_optimizer_state(optimizer, bot, all_in_idx)
            print(
                f"[Detox] Reset all-in regret output at {layer_name}[{all_in_idx}] "
                "and cleared optimizer momentum for that row."
            )

    losses = {"regret": [], "value": [], "sizing": []}
    gradient_steps_taken = 0
    optimizer_steps_taken = 0
    head_steps = {"regret": 0, "value": 0, "sizing": 0}

    # Signal handling (B5/M4): SIGINT *and* SIGTERM both request a graceful
    # stop — finish the current traversal, save a final checkpoint, and report
    # status="interrupted" so main() can exit 128+signum (130/143).  Pre-fix,
    # SIGINT exited 0 with status "complete" (an orchestrator keying on the
    # exit code would treat an under-trained model as finished) and SIGTERM
    # was untrapped (an OS kill lost the in-flight checkpoint).
    interrupted = {"flag": False, "signum": None}

    def _signal_handler(signum, frame):
        interrupted["flag"] = True
        interrupted["signum"] = signum
        print(f"\n[train_deep_cfr] {signal.Signals(signum).name} received — "
              f"saving checkpoint …", flush=True)

    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print("=" * 70)
    print("TRAINING DEEP CFR PLUS")
    print("=" * 70)
    print(f"  Variant:            {args.variant}")
    print(f"  Iterations:         {args.iterations}")
    print(f"  Update interval:    {args.update_interval}")
    print(f"  Checkpoint interval:{args.checkpoint_interval}")
    print(f"  Batch size:         {args.batch_size}")
    print(f"  Learning rate:      {args.lr}")
    print(f"  AIVAT sims:         {args.aivat_sims}")
    print(f"  Curriculum:         players {CURRICULUM_PLAYER_COUNTS[0]}-"
          f"{CURRICULUM_PLAYER_COUNTS[-1]}, "
          f"{CURRICULUM_MIN_DEPTH_BB}-{CURRICULUM_MAX_DEPTH_BB} BB "
          f"log-uniform (50% shared / 50% per-seat depths)")
    print(f"  Exploration eps:    0.30 → 0.05")
    print(f"  All-in shadow:      before iteration {args.all_in_warmup_iterations}")
    print(f"  All-in deploy gate: before iteration {args.all_in_deploy_iteration}")
    print(f"  All-in full release:{args.all_in_full_release_iteration}")
    print(f"  All-in detox:       {'enabled' if args.detox_all_in_on_resume else 'disabled'}")
    print(f"  Collapse canary:    {'disabled' if args.disable_collapse_canary else 'enabled'}")
    print(f"  Device:             {device}")
    print(f"  Save path:          {args.save_path}")
    total_params = sum(p.numel() for p in bot.network.parameters())
    print(f"  Network params:     {total_params:,}")
    print("=" * 70)
    print()

    t0 = _time.monotonic()
    abort_without_save = False
    nonfinite_abort = False
    last_checkpoint_iter = None
    last_checkpoint_written = None
    # Iteration accounting (4.1): the number of FULLY completed iterations.
    # The loop variable cannot be trusted for this — a signal breaks at the
    # TOP of iteration t+1, where the loop variable already reads t+1 even
    # though only t iterations ran (and it does not exist at all if the
    # signal lands before the first iteration).  Every report and checkpoint
    # stamp below uses this counter, so --resume continues exactly where the
    # run stopped instead of silently skipping one iteration.
    completed_iter = start_iter

    def final_shadow_stamp(iteration: int) -> bool | None:
        """Shadow-only stamp decision for a FINAL checkpoint (B4/I9).

        A final checkpoint below the all-in deploy gate yields a model whose
        inference permanently masks all-in.  Warn loudly and stamp the
        checkpoint metadata, but still save — smoke runs are intentionally
        short and must keep working (warn-and-stamp, NOT refuse).  Returns
        True (stamp) or None (omit the key entirely).
        """
        if iteration >= bot.all_in_deploy_iteration:
            return None
        print(
            f"\n[WARN] FINAL checkpoint at iteration {iteration} is below "
            f"the all-in deploy gate ({bot.all_in_deploy_iteration}).\n"
            f"       Inference for this model will PERMANENTLY mask the "
            f"all-in action — it is a SHADOW-ONLY artifact, not a "
            f"deployable bot.\n"
            f"       Stamping checkpoint metadata \"shadow_only\": true. "
            f"Train with --iterations >= "
            f"{bot.all_in_deploy_iteration} (TRAINING_PLAN step 7 uses "
            f"1,000,000) for a deployable model.\n",
            flush=True,
        )
        return True

    def save_emergency_checkpoint(iteration: int, reason: str) -> str:
        """Final save for an interrupted/aborted run — NEVER runs the canary.

        SIGINT/SIGTERM and the non-finite-loss abort must always preserve
        work for --resume and post-mortem.  Routing them through
        checkpoint_with_canary would (a) lose the checkpoint outright on a
        FAIL verdict and (b) spend ~150 bot.act() probe calls inside a
        SIGTERM grace window where a supervisor may escalate to SIGKILL.
        The canary still gates every PROMOTED checkpoint: the deploy-grade
        artifacts remain the canary-vetted save/safe pair, and this save
        only refreshes args.save_path (it never touches the .safe copy).
        """
        shadow_only = final_shadow_stamp(iteration)  # the run ends here
        save_checkpoint(args.save_path, iteration, bot, optimizer, losses,
                        nonfinite_skips=nonfinite["total"],
                        shadow_only=shadow_only)
        print(f"[train_deep_cfr] Emergency checkpoint ({reason}) saved to "
              f"{args.save_path} — collapse canary skipped.", flush=True)
        return args.save_path

    def checkpoint_with_canary(iteration: int, *,
                               final: bool = False) -> tuple[str, str]:
        shadow_only = final_shadow_stamp(iteration) if final else None
        if args.disable_collapse_canary:
            save_checkpoint(args.save_path, iteration, bot, optimizer, losses,
                            nonfinite_skips=nonfinite["total"],
                            shadow_only=shadow_only)
            return "DISABLED", args.save_path

        canary = quick_canary_probe(bot, device, current_iteration=iteration)
        raw_all_in = canary["raw_all_in"]
        search_all_in = canary["search_all_in"]

        # All canary metrics are diagnostic until the model reaches the deploy
        # boundary.  Before then, all-in is masked at inference and its output
        # row is intentionally untrained during the shadow phase; treating that
        # raw score as deploy-mature caused false aborts at early checkpoints.
        (status, base_status, extra_status, extra_failed, extra_warned,
         metrics_enforced) = decide_canary_status(
            canary, iteration, bot.all_in_deploy_iteration)

        phase = all_in_phase_for_iteration(
            iteration,
            bot.all_in_warmup_iterations,
            bot.all_in_full_release_iteration,
        )
        metrics_str = format_canary_metrics(canary)
        diagnostic_status = _worst_canary_status(base_status, extra_status)
        defer_note = "" if metrics_enforced else (
            f" [all health metrics reported only; enforced at iter >= "
            f"{bot.all_in_deploy_iteration}; would be {diagnostic_status}]")
        if status == "FAIL":
            reasons: list[str] = []
            if base_status == "FAIL":
                reasons.append(
                    f"all-in (search={search_all_in:.1%} > "
                    f"{CANARY_FAIL_SEARCH_MIN:.0%} or raw={raw_all_in:.1%} > "
                    f"{CANARY_FAIL_RAW_MIN:.0%})")
            if metrics_enforced:
                reasons.extend(extra_failed)
            # Header is generic ("collapse canary") so a fold-collapse abort
            # (strong_continue too low) is not mislabeled as an all-in collapse;
            # the reason list names the specific tripping metric(s).
            raise RuntimeError(
                f"Collapse canary FAILED at iter {iteration}: phase={phase} "
                f"{metrics_str} -- FAILED: {'; '.join(reasons)}"
            )
        if status == "WARN":
            reasons = []
            if base_status == "WARN":
                reasons.append("all-in freq")
            if metrics_enforced:
                reasons.extend(extra_warned)
            print(
                f"[WARN] iter {iteration}: phase={phase} {metrics_str}{defer_note} "
                f"-- side checkpoint only ({'; '.join(reasons)})",
                flush=True,
            )
        else:
            print(
                f"[CANARY] iter {iteration}: phase={phase} {metrics_str}{defer_note}",
                flush=True,
            )
        written = save_promoted_checkpoint(
            args.save_path,
            iteration,
            bot,
            optimizer,
            losses,
            status,
            nonfinite_skips=nonfinite["total"],
            shadow_only=shadow_only,
        )
        return status, written

    try:
        for t in range(start_iter + 1, args.iterations + 1):
            if interrupted["flag"]:
                break

            # Curriculum sampling (Fix 5 / I6): random player count 2-6 and
            # 10-200 BB stack depths so heads-up / short-stack endgames are
            # in-distribution, with the original hero-seat rotation.
            state, hero_seat = sample_curriculum_state(t)

            bot.network.eval()
            eps = epsilon_for_iteration(t, args.iterations)
            allow_all_in = allow_all_in_for_iteration(
                t, bot.all_in_warmup_iterations)
            all_in_policy_probability = all_in_policy_probability_for_iteration(
                t,
                bot.all_in_warmup_iterations,
                bot.all_in_full_release_iteration,
            )
            bot._cfr_recurse(
                state, hero_seat, depth=bot._MAX_CFR_DEPTH,
                iteration=t,
                regret_buf=regret_buf,
                value_buf=value_buf,
                sizing_buf=sizing_buf,
                exploration_epsilon=eps,
                allow_all_in=allow_all_in,
                all_in_policy_probability=all_in_policy_probability,
            )
            # Iteration t's traversal is done — count it.  The gradient step
            # and checkpoint below are interval-based aggregates over the
            # buffers, not part of "did iteration t run", so an abort inside
            # them still reports t completed iterations.
            completed_iter = t

            # Periodic gradient steps
            if t % args.update_interval == 0:
                bot.network.train()
                r_loss, v_loss, s_loss, step_info = train_step(
                    bot.network, optimizer,
                    regret_buf, value_buf, sizing_buf,
                    args.batch_size, device,
                )
                if step_info.get("nonfinite_skip"):
                    # Finiteness guard tripped (B2/I5): parameters were left
                    # untouched.  Log the three component losses so the bad
                    # head is identifiable, and keep the NaN out of the loss
                    # history (checkpoints store that history).
                    nonfinite["total"] += 1
                    nonfinite["consecutive"] += 1
                    print(
                        f"[WARN] iter {t}: non-finite loss — optimizer step "
                        f"SKIPPED (r={r_loss}, v={v_loss}, s={s_loss}; "
                        f"consecutive={nonfinite['consecutive']}, "
                        f"total={nonfinite['total']})",
                        flush=True,
                    )
                    if nonfinite["consecutive"] >= NONFINITE_SKIPS_ABORT_THRESHOLD:
                        print(
                            f"[ABORT] {nonfinite['consecutive']} consecutive "
                            f"non-finite losses (threshold "
                            f"{NONFINITE_SKIPS_ABORT_THRESHOLD}) — training "
                            f"signal is broken, aborting. Parameters are "
                            f"finite (every bad step was skipped); the final "
                            f"checkpoint is still saved.",
                            flush=True,
                        )
                        nonfinite_abort = True
                        break
                else:
                    losses["regret"].append(r_loss)
                    losses["value"].append(v_loss)
                    losses["sizing"].append(s_loss)
                    if step_info["did_step"]:
                        optimizer_steps_taken += 1
                        # A successful finite step ends any non-finite streak.
                        nonfinite["consecutive"] = 0
                    for head, trained in step_info["heads_trained"].items():
                        if trained:
                            head_steps[head] += 1
                    if step_info["heads_trained"]["regret"]:
                        gradient_steps_taken += 1

            # Checkpoint.  A periodic save landing exactly on the last
            # iteration IS the run's final checkpoint — mark it as such so
            # the shadow-only stamp/warning (B4/I9) cannot be skipped just
            # because --iterations is a multiple of --checkpoint-interval.
            if t % args.checkpoint_interval == 0:
                try:
                    _, last_checkpoint_written = checkpoint_with_canary(
                        t, final=(t >= args.iterations))
                    last_checkpoint_iter = t
                except RuntimeError as exc:
                    # All-in collapse canary tripped mid-run.  Break out of the
                    # loop rather than re-raising: the finally block prints the
                    # ABORT footer and the post-loop `abort_without_save` block
                    # returns status="aborted" (which main() maps to a nonzero
                    # exit).  Re-raising skipped that documented return path.
                    print(f"[ABORT] {exc}", flush=True)
                    abort_without_save = True
                    break

            # Progress
            if t % max(1, args.update_interval) == 0:
                elapsed = _time.monotonic() - t0
                print_progress(t, regret_buf, value_buf, sizing_buf, losses,
                               elapsed)

    except KeyboardInterrupt:
        # Belt-and-braces: our handler replaces the default SIGINT behavior,
        # so this only triggers for a KeyboardInterrupt raised by other means.
        interrupted["flag"] = True
        interrupted["signum"] = interrupted["signum"] or signal.SIGINT
        print("\n[train_deep_cfr] KeyboardInterrupt — saving checkpoint …")
    finally:
        # Final save.  completed_iter — NOT the loop variable — is the number
        # of fully completed iterations (see its definition above), so the
        # report and checkpoint metadata never claim an iteration that only
        # started.
        final_iter = completed_iter
        if interrupted["flag"] or nonfinite_abort:
            # Emergency save (4.1): unconditional and canary-free.  This may
            # re-save an iteration a periodic checkpoint already covered —
            # cheap, and it guarantees args.save_path holds the latest state
            # with the correct final shadow_only stamp for --resume.
            #
            # Checked BEFORE abort_without_save on purpose (4.2): a signal
            # can land while a periodic canary probe is mid-flight, and when
            # that probe then FAILs, both flags are set at once.  The run is
            # still an interrupt — the result tail below reports
            # status="interrupted" — so the save policy must match (pre-4.2
            # the abort branch won and the "interrupted" run saved nothing).
            # The FAILed canary still did its job: it kept this network out
            # of the promoted save/.safe pair; this save only refreshes
            # args.save_path.
            reason = ("nonfinite_abort" if nonfinite_abort else
                      signal.Signals(int(interrupted["signum"]
                                         or signal.SIGINT)).name)
            last_checkpoint_written = save_emergency_checkpoint(
                final_iter, reason)
            if abort_without_save:
                # Post-mortem breadcrumb for the double-flag case above.
                print("[train_deep_cfr] NOTE: a periodic collapse-canary "
                      "FAIL was also observed this run — the emergency "
                      "checkpoint is for --resume/post-mortem only; the "
                      ".safe copy remains the last canary-vetted artifact.",
                      flush=True)
        elif abort_without_save:
            # An actual canary probe FAILED on this exact network at a
            # periodic checkpoint, and NO emergency (signal / non-finite
            # abort) is active: the policy is collapsed, so refusing to
            # save it (and stopping) IS the intended behavior.
            print(
                "[ABORT] Final checkpoint not saved because the all-in "
                "collapse canary tripped.",
                flush=True,
            )
        elif final_iter != last_checkpoint_iter:
            try:
                _, last_checkpoint_written = checkpoint_with_canary(
                    final_iter, final=True)
            except RuntimeError as exc:
                print(f"[ABORT] Final checkpoint not saved: {exc}", flush=True)
                abort_without_save = True
        else:
            _ = last_checkpoint_written
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)

    elapsed_total = _time.monotonic() - t0
    if gradient_steps_taken == 0:
        print(
            f"[WARN] Training completed with 0 gradient steps. "
            f"Buffer sizes at end: regret={len(regret_buf)}, "
            f"value={len(value_buf)}, sizing={len(sizing_buf)}. "
            f"Likely cause: --batch-size {args.batch_size} too large for "
            f"--iterations {args.iterations}. Try --batch-size 32 for short runs."
        )

    if interrupted["flag"]:
        # SIGINT/SIGTERM: the finally block above already saved the final
        # checkpoint via the canary-free emergency path (4.1) — an interrupt
        # can never lose work to a FAILing canary probe, even one that FAILs
        # while the interrupt arrives (the emergency branch outranks
        # abort_without_save, 4.2).  Report "interrupted" — never
        # "complete" — and let main() exit 128+signum.
        signum = int(interrupted["signum"] or signal.SIGINT)
        print(f"\n{'='*70}")
        print(f"Training interrupted ({signal.Signals(signum).name}).")
        print(f"  Iterations reached: {final_iter}")
        print(f"  Gradient steps:     {gradient_steps_taken}")
        print(f"  Optimizer steps:    {optimizer_steps_taken}")
        print(f"  Non-finite skips:   {nonfinite['total']}")
        print(f"  Wall-clock:         {elapsed_total:.1f}s")
        print(f"  Last checkpoint:    {last_checkpoint_written or '<none>'}")
        print(f"{'='*70}")
        return {
            "status": "interrupted",
            "signal": signum,
            "final_iter": final_iter,
            "gradient_steps": gradient_steps_taken,
            "gradient_steps_taken": gradient_steps_taken,
            "optimizer_steps_taken": optimizer_steps_taken,
            "head_steps": head_steps,
            "nonfinite_skips": nonfinite["total"],
            "losses": losses,
            "regret_buf": regret_buf,
            "value_buf": value_buf,
            "sizing_buf": sizing_buf,
            "bot": bot,
            "optimizer": optimizer,
            "elapsed": elapsed_total,
            "checkpoint_saved": last_checkpoint_written,
        }

    if abort_without_save or nonfinite_abort:
        # Two distinct abort modes share the "aborted" status:
        #   collapse_canary — policy collapsed; final checkpoint NOT saved.
        #   nonfinite_loss  — too many consecutive non-finite losses; the
        #                     parameters are finite (bad steps were skipped),
        #                     so the finally block above did save a final
        #                     checkpoint.
        abort_reason = "collapse_canary" if abort_without_save else "nonfinite_loss"
        print(f"\n{'='*70}")
        print(f"Training aborted ({abort_reason}).")
        print(f"  Iterations reached: {final_iter}")
        print(f"  Gradient steps:     {gradient_steps_taken}")
        print(f"  Optimizer steps:    {optimizer_steps_taken}")
        print(f"  Non-finite skips:   {nonfinite['total']}")
        print(f"  Wall-clock:         {elapsed_total:.1f}s")
        print(f"  Last checkpoint:    {last_checkpoint_written or '<none>'}")
        print(f"{'='*70}")
        return {
            "status": "aborted",
            "abort_reason": abort_reason,
            "final_iter": final_iter,
            "gradient_steps": gradient_steps_taken,
            "gradient_steps_taken": gradient_steps_taken,
            "optimizer_steps_taken": optimizer_steps_taken,
            "head_steps": head_steps,
            "nonfinite_skips": nonfinite["total"],
            "losses": losses,
            "regret_buf": regret_buf,
            "value_buf": value_buf,
            "sizing_buf": sizing_buf,
            "bot": bot,
            "optimizer": optimizer,
            "elapsed": elapsed_total,
            "checkpoint_saved": last_checkpoint_written,
        }

    print(f"\n{'='*70}")
    print(f"Training complete.")
    print(f"  Iterations:       {final_iter}")
    print(f"  Gradient steps:   {gradient_steps_taken}")
    print(f"  Optimizer steps:  {optimizer_steps_taken}")
    print(f"  Head steps:       regret={head_steps['regret']}, value={head_steps['value']}, sizing={head_steps['sizing']}")
    print(f"  Non-finite skips: {nonfinite['total']}")
    print(f"  Buffer sizes:     regret={len(regret_buf)}, value={len(value_buf)}, sizing={len(sizing_buf)}")
    print(f"  Wall-clock:       {elapsed_total:.1f}s")
    print(f"  Checkpoint saved: {last_checkpoint_written or args.save_path}")
    print(f"{'='*70}")

    return {
        "status": "complete",
        "final_iter": final_iter,
        "gradient_steps": gradient_steps_taken,
        "gradient_steps_taken": gradient_steps_taken,
        "optimizer_steps_taken": optimizer_steps_taken,
        "head_steps": head_steps,
        "nonfinite_skips": nonfinite["total"],
        "losses": losses,
        "regret_buf": regret_buf,
        "value_buf": value_buf,
        "sizing_buf": sizing_buf,
        "bot": bot,
        "optimizer": optimizer,
        "elapsed": elapsed_total,
        "checkpoint_saved": last_checkpoint_written or args.save_path,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train Deep CFR Plus bot via external-sampling CFR traversals")
    parser.add_argument("--variant", choices=["small", "large"], required=True)
    # Default raised from 100k (B4/I9): 100k sat BELOW the 150k all-in deploy
    # gate, so a default run produced a checkpoint whose inference permanently
    # masks all-in.  1M matches TRAINING_PLAN step 7.  Short runs still work —
    # an under-gate FINAL checkpoint is loudly warned about and stamped
    # "shadow_only" instead of being refused (smoke tests run ~100 iterations).
    parser.add_argument(
        "--iterations",
        type=int,
        default=1_000_000,
        help="Training iterations (default 1,000,000; below "
             f"--all-in-deploy-iteration the final model masks all-in)",
    )
    parser.add_argument("--update-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--aivat-sims", type=int, default=500)
    parser.add_argument(
        "--all-in-warmup-iterations",
        type=int,
        default=ALL_IN_WARMUP_ITERATIONS,
        help="Shadow-train all-in before this iteration; staged exposure starts here",
    )
    parser.add_argument(
        "--all-in-deploy-iteration",
        type=int,
        default=ALL_IN_DEPLOY_ITERATION,
        help="Tournament inference masks all-in before this checkpoint iteration",
    )
    parser.add_argument(
        "--all-in-full-release-iteration",
        type=int,
        default=ALL_IN_FULL_RELEASE_ITERATION,
        help="Iteration where staged all-in self-play and inference caps end",
    )
    parser.add_argument(
        "--detox-all-in-on-resume",
        action="store_true",
        help="On resume, reset only the all-in regret output row before continuing",
    )
    parser.add_argument(
        "--disable-collapse-canary",
        action="store_true",
        help="Skip checkpoint all-in collapse probes; intended only for smoke tests",
    )
    parser.add_argument("--save-path", type=str, default="models/deep_cfr_v1.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    result = run_training(args)
    status = result.get("status")
    if status == "interrupted":
        # Conventional shell exit code for death-by-signal: 128 + signum
        # (130 = SIGINT, 143 = SIGTERM), so orchestrators can tell an
        # interrupted run from a completed (0) or aborted (1) one.
        raise SystemExit(128 + int(result.get("signal") or signal.SIGINT))
    if status != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
