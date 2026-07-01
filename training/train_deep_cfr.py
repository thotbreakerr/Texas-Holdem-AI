"""
training/train_deep_cfr.py — multiway Deep CFR-inspired schema-v2 trainer
-------------------------------------------------------------------------
Collects external-sampling traversals under a frozen advantage policy, then
reinitializes/refits the advantage network at round boundaries. Deployment
uses a separately trained reach-weighted average-strategy network.

Usage:
    python training/train_deep_cfr.py --variant small --iterations 50 \
        --update-interval 10 --checkpoint-interval 25 --batch-size 32 \
        --aivat-sims 50 --save-path /tmp/deep_cfr_smoke.pt
    python training/train_deep_cfr.py --variant large --iterations 1000000 \\
        --curriculum-profile sixmax --save-path models/deep_cfr_v2.pt
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
    ABSTRACT_ACTIONS, NUM_ACTIONS, DEEP_CFR_SCHEMA_VERSION,
    _abstract_to_concrete, _is_effective_all_in, _infer_big_blind,
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
CURRICULUM_PROFILES = {
    "sixmax": {
        2: 0.125,
        3: 0.10,
        4: 0.125,
        5: 0.15,
        6: 0.50,
    },
}
CURRICULUM_MIN_DEPTH_BB = 10     # shortest stack: 10 big blinds
CURRICULUM_MAX_DEPTH_BB = 200    # deepest stack: 200 big blinds
CURRICULUM_SHARED_DEPTH_PROB = 0.5  # 50% one depth for all seats, 50% per-seat
DEFAULT_ROUND_SIZE = 25_000
DEFAULT_CANARY_ENFORCE_ITERATION = 100_000
DEFAULT_CANARY_FAIL_PATIENCE = 3
PILOT_GATE_ITERATIONS = (100_000, 150_000)
CANARY_WARN_SEARCH_MIN = 0.10
CANARY_WARN_RAW_MIN = 0.10
CANARY_FAIL_SEARCH_MIN = 0.25
CANARY_FAIL_RAW_MIN = 0.25
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


def pilot_health_failures(metrics: dict) -> list[str]:
    """Return rollout-gate failures for the 100k and 150k pilot probes."""
    checks = [
        ("raw_all_in", metrics.get("raw_all_in", 1.0), 0.10, "high"),
        ("search_all_in", metrics.get("search_all_in", 1.0), 0.15, "high"),
        ("strong_continue", metrics.get("strong_continue", 0.0), 0.80, "low"),
        ("normal_action_mass", metrics.get("normal_action_mass", 0.0), 0.30, "low"),
    ]
    failures = []
    for name, value, threshold, direction in checks:
        failed = value >= threshold if direction == "high" else value < threshold
        if failed:
            comparator = ">=" if direction == "high" else "<"
            failures.append(
                f"{name}={value:.1%} ({comparator} {threshold:.0%})")
    return failures


def epsilon_for_iteration(t: int, total: int) -> float:
    """Anneal training-time opponent exploration from 0.30 to 0.05."""
    progress = min(1.0, t / max(total, 1))
    return 0.30 - 0.25 * progress


# ═══════════════════════════════════════════════════════════════════════════════
#  Device selection
# ═══════════════════════════════════════════════════════════════════════════════

def pick_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
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


def sample_curriculum_state(
    iteration: int,
    rng=random,
    profile: str = "sixmax",
):
    """Sample a six-player-heavy tournament state."""
    weights = CURRICULUM_PROFILES[profile]
    roll = rng.random()
    cumulative = 0.0
    n_seats = CURRICULUM_PLAYER_COUNTS[-1]
    for count in CURRICULUM_PLAYER_COUNTS:
        cumulative += weights[count]
        if roll < cumulative:
            n_seats = count
            break
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
    optimizers,
    regret_buf: ReservoirBuffer,
    value_buf: ReservoirBuffer,
    sizing_buf: ReservoirBuffer,
    batch_size: int,
    device: torch.device,
    strategy_buf: ReservoirBuffer | None = None,
) -> tuple:
    """Run one independent optimizer step per ready schema-v2 objective."""
    network.train()
    if not isinstance(optimizers, dict):
        optimizers = {
            "advantage": optimizers,
            "strategy": optimizers,
            "value": optimizers,
            "sizing": optimizers,
        }
    if strategy_buf is None:
        strategy_buf = ReservoirBuffer(capacity=1)
    losses = {"regret": 0.0, "strategy": 0.0, "value": 0.0, "sizing": 0.0}
    heads_trained = {
        "regret": False,
        "strategy": False,
        "value": False,
        "sizing": False,
    }
    any_step = False
    nonfinite_skip = False

    def _step(name: str, module: nn.Module, loss: torch.Tensor) -> None:
        nonlocal any_step, nonfinite_skip
        losses[name] = float(loss.detach().item())
        heads_trained[name] = True
        if not bool(torch.isfinite(loss).item()):
            nonfinite_skip = True
            return
        optimizer_key = "advantage" if name == "regret" else name
        optimizer = optimizers[optimizer_key]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=10.0)
        optimizer.step()
        any_step = True

    if len(regret_buf) >= batch_size:
        samples = regret_buf.sample(batch_size)
        batch = _stack_input_dicts([s[0] for s in samples], device)
        targets = torch.stack([s[1] for s in samples]).to(device)
        masks = torch.stack([s[2] for s in samples]).to(device)
        weights = torch.tensor(
            [s[3] for s in samples], dtype=torch.float32, device=device)
        weights = weights / (weights.sum() + 1e-8)
        preds = network.advantage_forward(batch)
        elementwise = nn.functional.smooth_l1_loss(
            preds * masks, targets * masks, reduction="none") * masks
        per_sample = elementwise.sum(dim=1) / masks.sum(dim=1).clamp(min=1.0)
        _step("regret", network.advantage, (per_sample * weights).sum())

    if len(strategy_buf) >= batch_size:
        samples = strategy_buf.sample(batch_size)
        batch = _stack_input_dicts([s[0] for s in samples], device)
        targets = torch.stack([s[1] for s in samples]).to(device)
        masks = torch.stack([s[2] for s in samples]).to(device)
        weights = torch.tensor(
            [s[3] for s in samples], dtype=torch.float32, device=device)
        weights = weights / (weights.sum() + 1e-8)
        logits = network.strategy_forward(batch).masked_fill(masks <= 0, -1e9)
        log_probs = nn.functional.log_softmax(logits, dim=1)
        per_sample = -(targets * log_probs * masks).sum(dim=1)
        _step("strategy", network.strategy, (per_sample * weights).sum())

    if len(value_buf) >= batch_size:
        samples = value_buf.sample(batch_size)
        batch = _stack_input_dicts([s[0] for s in samples], device)
        targets = torch.tensor(
            [s[1] for s in samples], dtype=torch.float32, device=device)
        _step(
            "value",
            network.value,
            nn.functional.smooth_l1_loss(network.value_forward(batch), targets),
        )

    sizing_batch = min(batch_size, len(sizing_buf))
    if sizing_batch >= min(batch_size, 16):
        samples = sizing_buf.sample(sizing_batch)
        batch = _stack_input_dicts([s[0] for s in samples], device)
        targets = torch.tensor(
            [s[1] for s in samples], dtype=torch.float32, device=device)
        _step(
            "sizing",
            network.sizing,
            nn.functional.mse_loss(network.sizing_forward(batch), targets),
        )

    return losses["regret"], losses["value"], losses["sizing"], {
        "strategy_loss": losses["strategy"],
        "heads_trained": heads_trained,
        "did_step": any_step,
        "nonfinite_skip": nonfinite_skip,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Checkpoint save/load (atomic via .tmp + os.replace)
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(path: str, iteration: int, bot: DeepCFRBot,
                    optimizer, losses: dict,
                    *, nonfinite_skips: int = 0,
                    buffers: dict | None = None,
                    canary_fail_streak: int = 0,
                    round_index: int = 0,
                    last_fit_iteration: int = 0,
                    pilot_gates_completed=(),
                    canary_status: str = "UNKNOWN",
                    last_canary_metrics: dict | None = None,
                    curriculum_profile: str = "sixmax"):
    """Atomically save a complete, resumable schema-v2 training snapshot."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    if not isinstance(optimizer, dict):
        optimizer = {"legacy": optimizer}
    if buffers is None:
        buffers = {
            "regret": bot.regret_buffer,
            "strategy": bot.strategy_buffer,
            "value": bot.value_buffer,
            "sizing": bot.sizing_buffer,
        }
    payload = {
        "schema_version": DEEP_CFR_SCHEMA_VERSION,
        "algorithm": "multiway_deep_cfr_inspired",
        "iteration": iteration,
        "config": bot.config,
        "network_state_dict": bot.network.state_dict(),
        "optimizer_state_dicts": {
            name: opt.state_dict() for name, opt in optimizer.items()
        },
        "reservoirs": {
            name: buffer.state_dict() for name, buffer in buffers.items()
        },
        "curriculum_profile": curriculum_profile,
        "curriculum_weights": CURRICULUM_PROFILES[curriculum_profile],
        "round_index": int(round_index),
        "last_fit_iteration": int(last_fit_iteration),
        "pilot_gates_completed": sorted(
            int(item) for item in pilot_gates_completed),
        "canary_status": str(canary_status),
        "last_canary_metrics": dict(last_canary_metrics or {}),
        "canary_fail_streak": int(canary_fail_streak),
        "python_random_state": random.getstate(),
        "bot_random_state": bot._rng.getstate(),
        "torch_random_state": torch.random.get_rng_state(),
        "losses": {k: v[-100:] for k, v in losses.items()},  # last 100 only
        "nonfinite_skips": int(nonfinite_skips),
    }
    if torch.cuda.is_available():
        payload["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
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
    if search_all_in >= CANARY_FAIL_SEARCH_MIN or raw_all_in >= CANARY_FAIL_RAW_MIN:
        return "FAIL"
    if search_all_in >= CANARY_WARN_SEARCH_MIN or raw_all_in >= CANARY_WARN_RAW_MIN:
        return "WARN"
    return "PASS"


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
    """Combine health metrics and defer enforcement until model maturity."""
    base_status = classify_canary(
        canary.get("raw_all_in", 0.0), canary.get("search_all_in", 0.0))
    extra_status, extra_failed, extra_warned = classify_extra_canary_metrics(canary)
    metrics_enforced = iteration >= deploy_iteration
    diagnostic_status = _worst_canary_status(base_status, extra_status)
    status = diagnostic_status if metrics_enforced else "PASS"
    return (status, base_status, extra_status, extra_failed, extra_warned,
            metrics_enforced)


def save_promoted_checkpoint(path: str, iteration: int, bot: DeepCFRBot,
                             optimizer, losses: dict,
                             status: str, *, nonfinite_skips: int = 0,
                             **checkpoint_meta) -> str:
    """Save according to canary promotion rules and return the written path."""
    extra = {
        "nonfinite_skips": nonfinite_skips,
        **checkpoint_meta,
    }
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
                    optimizer,
                    meta_out: dict | None = None,
                    buffers: dict | None = None) -> int:
    """Restore a complete schema-v2 training snapshot."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("schema_version") != DEEP_CFR_SCHEMA_VERSION:
        raise RuntimeError(
            f"Cannot resume {path!r}: expected Deep CFR schema "
            f"{DEEP_CFR_SCHEMA_VERSION}, found {ckpt.get('schema_version', 1)}. "
            "Legacy v1 checkpoints are postmortem-only."
        )
    if "reservoirs" not in ckpt:
        raise RuntimeError("schema-v2 resume requires persisted reservoirs")
    if not isinstance(optimizer, dict):
        optimizer = {"legacy": optimizer}
    optimizer_states = ckpt.get("optimizer_state_dicts", {})
    missing_optimizers = sorted(set(optimizer) - set(optimizer_states))
    if missing_optimizers:
        raise RuntimeError(
            f"checkpoint is missing optimizer states: {missing_optimizers}")
    if buffers is None:
        buffers = {
            "regret": bot.regret_buffer,
            "strategy": bot.strategy_buffer,
            "value": bot.value_buffer,
            "sizing": bot.sizing_buffer,
        }
    missing_buffers = sorted(set(buffers) - set(ckpt["reservoirs"]))
    if missing_buffers:
        raise RuntimeError(
            f"checkpoint is missing reservoir snapshots: {missing_buffers}")
    if meta_out is not None:
        meta_out["nonfinite_skips"] = int(ckpt.get("nonfinite_skips", 0) or 0)
        meta_out["canary_fail_streak"] = int(
            ckpt.get("canary_fail_streak", 0) or 0)
        meta_out["round_index"] = int(ckpt.get("round_index", 0) or 0)
        meta_out["last_fit_iteration"] = int(
            ckpt.get("last_fit_iteration", ckpt.get("iteration", 0)) or 0)
        meta_out["pilot_gates_completed"] = set(
            int(item) for item in ckpt.get("pilot_gates_completed", []))
        meta_out["curriculum_profile"] = ckpt.get(
            "curriculum_profile", "sixmax")
    bot.network.load_state_dict(ckpt["network_state_dict"])
    for name, opt in optimizer.items():
        opt.load_state_dict(optimizer_states[name])
    for name, buffer in buffers.items():
        buffer.load_state_dict(ckpt["reservoirs"][name])
    if "python_random_state" in ckpt:
        random.setstate(ckpt["python_random_state"])
    if "bot_random_state" in ckpt:
        bot._rng.setstate(ckpt["bot_random_state"])
    if "torch_random_state" in ckpt:
        torch.random.set_rng_state(ckpt["torch_random_state"])
    if "torch_cuda_random_state_all" in ckpt and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["torch_cuda_random_state_all"])
    iteration = ckpt["iteration"]
    print(f"[train_deep_cfr] Resumed from {path} at iteration {iteration}")
    print(
        f"[Resume] Loaded schema-v2 networks, optimizers, four reservoirs, "
        f"and RNG state from iteration {iteration}."
    )
    return iteration


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
    current_iteration: int,
    use_advantage: bool = False,
) -> dict:
    """Aggregate expected behavior directly from policy probabilities."""
    _ = current_iteration
    stats = {
        "total": 0,
        "all_in": 0.0,
        "pfr": 0.0,
        "continue": 0.0,
        "normal_action_mass": 0.0,
        "size_numerator": 0.0,
        "size_weight": 0.0,
    }
    if hasattr(bot, "network"):
        bot.network.eval()
    for view in views:
        strategy, legal_mask, sizing = bot.policy_probabilities(
            view,
            search_depth=search_depth,
            use_advantage=use_advantage,
        )
        stats["total"] += 1
        fold_idx = ABSTRACT_ACTIONS.index("fold")
        stats["continue"] += 1.0 - (
            strategy[fold_idx] if fold_idx in legal_mask else 0.0)
        for action_idx in legal_mask:
            probability = float(strategy[action_idx])
            label = ABSTRACT_ACTIONS[action_idx]
            action = _abstract_to_concrete(
                action_idx,
                view.legal_actions,
                view.pot,
                sizing_frac=sizing,
                street=view.street,
                big_blind=_infer_big_blind(view),
            )
            effective_all_in = _is_effective_all_in(
                action, view.legal_actions)
            if effective_all_in:
                stats["all_in"] += probability
            if label in {
                "bet_33", "bet_50", "bet_67", "bet_75", "bet_100", "all_in",
            }:
                stats["pfr"] += probability
            if label == "check_call" or (
                label.startswith("bet_") and not effective_all_in
            ):
                stats["normal_action_mass"] += probability
            if (
                label.startswith("bet_")
                and not effective_all_in
                and action.amount is not None
            ):
                stats["size_numerator"] += (
                    probability * action.amount / max(view.pot, 1))
                stats["size_weight"] += probability
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
        bot, views, search_depth=0, current_iteration=current_iteration)
    search = _canary_collect_stats(
        bot, views, search_depth=bot.search_depth,
        current_iteration=current_iteration)
    strong = _canary_collect_stats(
        bot, _canary_strong_views(n, seed), search_depth=0,
        current_iteration=current_iteration)
    advantage = _canary_collect_stats(
        bot, views, search_depth=0, current_iteration=current_iteration,
        use_advantage=True)
    n_raw = max(raw["total"], 1)
    n_strong = max(strong["total"], 1)
    avg_raise = (
        raw["size_numerator"] / raw["size_weight"]
        if raw["size_weight"] > 0 else 0.0
    )
    return {
        "raw_all_in": raw["all_in"] / n_raw,
        "search_all_in": search["all_in"] / max(search["total"], 1),
        "preflop_pfr": raw["pfr"] / n_raw,
        "preflop_avg_raise": avg_raise,
        "strong_all_in": strong["all_in"] / n_strong,
        "strong_continue": strong["continue"] / n_strong,
        "normal_action_mass": raw["normal_action_mass"] / n_raw,
        "advantage_raw_all_in": advantage["all_in"] / n_raw,
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
        f"strong_continue={canary.get('strong_continue', 1.0):.1%}, "
        f"normal_mass={canary.get('normal_action_mass', 0.0):.1%}, "
        f"adv_raw={canary.get('advantage_raw_all_in', 0.0):.1%}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Progress printing
# ═══════════════════════════════════════════════════════════════════════════════

def print_progress(t: int, regret_buf, strategy_buf, value_buf, sizing_buf, losses,
                   elapsed: float = 0.0):
    r = losses["regret"][-1] if losses["regret"] else 0.0
    p = losses["strategy"][-1] if losses["strategy"] else 0.0
    v = losses["value"][-1] if losses["value"] else 0.0
    s = losses["sizing"][-1] if losses["sizing"] else 0.0
    rate = t / elapsed if elapsed > 0 else 0
    print(
        f"  iter={t:>7}  "
        f"bufs=({len(regret_buf)},{len(strategy_buf)},"
        f"{len(value_buf)},{len(sizing_buf)})  "
        f"loss=(a={r:.4f}, p={p:.4f}, v={v:.4f}, s={s:.4f})  "
        f"rate={rate:.1f} it/s",
        flush=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Main training loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_training(args) -> dict:
    """Collect frozen-policy rounds and fit independent schema-v2 networks."""
    device = pick_device(args.device)
    config = (
        DeepCFRConfig.small()
        if args.variant == "small"
        else DeepCFRConfig.large()
    )
    bot = DeepCFRBot(
        config=config, inference_mode=False, aivat_sims=args.aivat_sims)
    bot.network.to(device)

    def make_optimizers():
        return {
            "advantage": torch.optim.Adam(
                bot.network.advantage.parameters(), lr=args.lr),
            "strategy": torch.optim.Adam(
                bot.network.strategy.parameters(), lr=args.lr),
            "value": torch.optim.Adam(
                bot.network.value.parameters(), lr=args.lr),
            "sizing": torch.optim.Adam(
                bot.network.sizing.parameters(), lr=args.lr),
        }

    optimizers = make_optimizers()
    buffers = {
        "regret": ReservoirBuffer(capacity=1_000_000),
        "strategy": ReservoirBuffer(capacity=1_000_000),
        "value": ReservoirBuffer(capacity=1_000_000),
        "sizing": ReservoirBuffer(capacity=1_000_000),
    }
    bot.regret_buffer = buffers["regret"]
    bot.strategy_buffer = buffers["strategy"]
    bot.value_buffer = buffers["value"]
    bot.sizing_buffer = buffers["sizing"]

    losses = {"regret": [], "strategy": [], "value": [], "sizing": []}
    head_steps = {
        "regret": 0, "strategy": 0, "value": 0, "sizing": 0}
    nonfinite = {"total": 0, "consecutive": 0}
    canary_fail_streak = 0
    round_index = 0
    last_fit_iteration = 0
    pilot_gates_completed: set[int] = set()
    start_iter = 0

    if args.resume:
        if not os.path.exists(args.resume):
            raise RuntimeError(f"resume checkpoint not found: {args.resume}")
        resume_meta: dict = {}
        start_iter = load_checkpoint(
            args.resume,
            bot,
            optimizers,
            meta_out=resume_meta,
            buffers=buffers,
        )
        if resume_meta.get("curriculum_profile") != args.curriculum_profile:
            raise RuntimeError(
                "resume curriculum profile does not match command line")
        nonfinite["total"] = resume_meta["nonfinite_skips"]
        canary_fail_streak = resume_meta["canary_fail_streak"]
        round_index = resume_meta["round_index"]
        last_fit_iteration = resume_meta["last_fit_iteration"]
        pilot_gates_completed = resume_meta["pilot_gates_completed"]

    interrupted = {"flag": False, "signum": None}

    def _signal_handler(signum, _frame):
        interrupted["flag"] = True
        interrupted["signum"] = signum
        print(
            f"\n[train_deep_cfr] {signal.Signals(signum).name} received; "
            "saving a resumable schema-v2 checkpoint...",
            flush=True,
        )

    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print("=" * 70)
    print("TRAINING MULTIWAY DEEP CFR-INSPIRED V2")
    print("=" * 70)
    print(f"  Variant:             {args.variant}")
    print(f"  Traversals:          {args.iterations}")
    print(f"  Frozen round size:   {args.round_size}")
    print(f"  Fit-step divisor:    {args.update_interval}")
    print(f"  Checkpoint interval: {args.checkpoint_interval}")
    print(f"  Curriculum:          {args.curriculum_profile} "
          f"{CURRICULUM_PROFILES[args.curriculum_profile]}")
    print(f"  Canary enforcement:  iter >= {args.canary_enforce_iteration}")
    print(f"  Canary patience:     {args.canary_fail_patience}")
    print(f"  Device:              {device}")
    print(f"  Save path:           {args.save_path}")
    print(f"  Parameters:          "
          f"{sum(p.numel() for p in bot.network.parameters()):,}")
    print("=" * 70)

    t0 = _time.monotonic()
    completed_iter = start_iter
    gradient_steps_taken = 0
    optimizer_steps_taken = 0
    last_checkpoint_iter = None
    last_checkpoint_written = None
    abort_without_save = False
    nonfinite_abort = False

    def checkpoint_kwargs():
        return {
            "buffers": buffers,
            "canary_fail_streak": canary_fail_streak,
            "round_index": round_index,
            "last_fit_iteration": last_fit_iteration,
            "pilot_gates_completed": pilot_gates_completed,
            "curriculum_profile": args.curriculum_profile,
        }

    def save_emergency_checkpoint(iteration: int, reason: str) -> str:
        save_checkpoint(
            args.save_path,
            iteration,
            bot,
            optimizers,
            losses,
            nonfinite_skips=nonfinite["total"],
            canary_status="EMERGENCY",
            **checkpoint_kwargs(),
        )
        print(
            f"[train_deep_cfr] Emergency checkpoint ({reason}) saved to "
            f"{args.save_path}; canary skipped.",
            flush=True,
        )
        return args.save_path

    def checkpoint_with_canary(iteration: int) -> tuple[str, str]:
        nonlocal canary_fail_streak
        if args.disable_collapse_canary:
            save_checkpoint(
                args.save_path,
                iteration,
                bot,
                optimizers,
                losses,
                nonfinite_skips=nonfinite["total"],
                canary_status="DISABLED",
                **checkpoint_kwargs(),
            )
            return "DISABLED", args.save_path

        canary = quick_canary_probe(
            bot, device, current_iteration=iteration)
        pending_pilot_gates = [
            gate for gate in PILOT_GATE_ITERATIONS
            if gate <= iteration and gate not in pilot_gates_completed
        ]
        pilot_failures = (
            pilot_health_failures(canary) if pending_pilot_gates else [])
        if pilot_failures:
            raise RuntimeError(
                f"Pilot health gate {pending_pilot_gates} failed at "
                f"checkpoint iter {iteration}: "
                f"{format_canary_metrics(canary)}; "
                f"FAILED: {'; '.join(pilot_failures)}")
        pilot_gates_completed.update(pending_pilot_gates)
        (
            status,
            base_status,
            _extra_status,
            extra_failed,
            extra_warned,
            metrics_enforced,
        ) = decide_canary_status(
            canary, iteration, args.canary_enforce_iteration)
        diagnostic_status = _worst_canary_status(
            base_status, classify_extra_canary_metrics(canary)[0])
        if metrics_enforced and diagnostic_status == "FAIL":
            canary_fail_streak += 1
        elif metrics_enforced:
            canary_fail_streak = 0

        metrics_str = format_canary_metrics(canary)
        defer_note = "" if metrics_enforced else (
            f" [reported only; enforced at iter >= "
            f"{args.canary_enforce_iteration}; would be {diagnostic_status}]")
        if (
            metrics_enforced
            and diagnostic_status == "FAIL"
            and canary_fail_streak >= args.canary_fail_patience
        ):
            reasons = list(extra_failed)
            if base_status == "FAIL":
                reasons.insert(0, "raw/search all-in probability")
            raise RuntimeError(
                f"Collapse canary failed {canary_fail_streak} consecutive "
                f"checkpoints at iter {iteration}: {metrics_str}; "
                f"FAILED: {'; '.join(reasons)}")

        save_status = status
        if metrics_enforced and diagnostic_status == "FAIL":
            save_status = "WARN"
        if save_status == "WARN":
            reasons = list(extra_warned)
            if diagnostic_status == "FAIL":
                reasons = list(extra_failed) or ["raw/search all-in"]
            print(
                f"[WARN] iter {iteration}: {metrics_str}{defer_note}; "
                f"failure_streak={canary_fail_streak}/"
                f"{args.canary_fail_patience}; side checkpoint only "
                f"({'; '.join(reasons)})",
                flush=True,
            )
        else:
            print(
                f"[CANARY] iter {iteration}: {metrics_str}{defer_note}",
                flush=True,
            )
        written = save_promoted_checkpoint(
            args.save_path,
            iteration,
            bot,
            optimizers,
            losses,
            save_status,
            nonfinite_skips=nonfinite["total"],
            canary_status=(
                save_status
                if metrics_enforced
                else f"DIAGNOSTIC_{diagnostic_status}"
            ),
            last_canary_metrics=canary,
            **checkpoint_kwargs(),
        )
        return save_status, written

    def fit_round(round_end: int, round_span: int) -> bool:
        nonlocal optimizers, round_index, last_fit_iteration
        nonlocal gradient_steps_taken, optimizer_steps_taken, nonfinite_abort
        bot.network.reinitialize_advantage()
        optimizers["advantage"] = torch.optim.Adam(
            bot.network.advantage.parameters(), lr=args.lr)
        fit_steps = max(1, math.ceil(round_span / max(1, args.update_interval)))
        print(
            f"[ROUND] fitting round {round_index + 1} at iter {round_end}: "
            f"{fit_steps} steps from cumulative reservoirs",
            flush=True,
        )
        for _ in range(fit_steps):
            r_loss, v_loss, s_loss, info = train_step(
                bot.network,
                optimizers,
                buffers["regret"],
                buffers["value"],
                buffers["sizing"],
                args.batch_size,
                device,
                strategy_buf=buffers["strategy"],
            )
            if info["nonfinite_skip"]:
                nonfinite["total"] += 1
                nonfinite["consecutive"] += 1
                print(
                    f"[WARN] iter {round_end}: non-finite training loss; "
                    f"optimizer step skipped "
                    f"(a={r_loss}, p={info.get('strategy_loss', 0.0)}, "
                    f"v={v_loss}, s={s_loss}; "
                    f"consecutive={nonfinite['consecutive']})",
                    flush=True,
                )
                if (
                    nonfinite["consecutive"]
                    >= NONFINITE_SKIPS_ABORT_THRESHOLD
                ):
                    print(
                        f"[ABORT] {nonfinite['consecutive']} consecutive "
                        "non-finite training losses.",
                        flush=True,
                    )
                    nonfinite_abort = True
                    return False
                continue
            nonfinite["consecutive"] = 0
            losses["regret"].append(r_loss)
            losses["strategy"].append(info.get("strategy_loss", 0.0))
            losses["value"].append(v_loss)
            losses["sizing"].append(s_loss)
            if info["did_step"]:
                optimizer_steps_taken += 1
            for head, trained in info["heads_trained"].items():
                if trained:
                    head_steps[head] += 1
            if info["heads_trained"].get("regret", False):
                gradient_steps_taken += 1
        round_index += 1
        last_fit_iteration = round_end
        bot.network.eval()
        return True

    try:
        for t in range(start_iter + 1, args.iterations + 1):
            if interrupted["flag"]:
                break
            state, hero_seat = sample_curriculum_state(
                t, profile=args.curriculum_profile)
            bot.network.eval()
            bot._cfr_recurse(
                state,
                hero_seat,
                depth=bot._MAX_CFR_DEPTH,
                iteration=t,
                regret_buf=buffers["regret"],
                strategy_buf=buffers["strategy"],
                value_buf=buffers["value"],
                sizing_buf=buffers["sizing"],
                exploration_epsilon=epsilon_for_iteration(t, args.iterations),
            )
            completed_iter = t

            at_round_end = t - last_fit_iteration >= args.round_size
            at_final = t == args.iterations
            if at_round_end or at_final:
                span = max(1, t - last_fit_iteration)
                if not fit_round(t, span):
                    break

            if t % args.checkpoint_interval == 0:
                try:
                    _, last_checkpoint_written = checkpoint_with_canary(t)
                    last_checkpoint_iter = t
                except RuntimeError as exc:
                    print(f"[ABORT] {exc}", flush=True)
                    abort_without_save = True
                    break

            if t % max(1, args.update_interval) == 0:
                print_progress(
                    t,
                    buffers["regret"],
                    buffers["strategy"],
                    buffers["value"],
                    buffers["sizing"],
                    losses,
                    _time.monotonic() - t0,
                )
    except KeyboardInterrupt:
        interrupted["flag"] = True
        interrupted["signum"] = interrupted["signum"] or signal.SIGINT
    finally:
        final_iter = completed_iter
        if interrupted["flag"] or nonfinite_abort:
            reason = (
                "nonfinite_abort"
                if nonfinite_abort
                else signal.Signals(
                    int(interrupted["signum"] or signal.SIGINT)).name
            )
            last_checkpoint_written = save_emergency_checkpoint(
                final_iter, reason)
        elif abort_without_save:
            print(
                "[ABORT] Final checkpoint not saved because the collapse "
                "canary exhausted its failure patience.",
                flush=True,
            )
        elif final_iter != last_checkpoint_iter:
            try:
                _, last_checkpoint_written = checkpoint_with_canary(final_iter)
            except RuntimeError as exc:
                print(f"[ABORT] Final checkpoint not saved: {exc}", flush=True)
                abort_without_save = True
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)

    elapsed_total = _time.monotonic() - t0
    status = "complete"
    abort_reason = None
    if interrupted["flag"]:
        status = "interrupted"
    elif nonfinite_abort:
        status = "aborted"
        abort_reason = "nonfinite_loss"
    elif abort_without_save:
        status = "aborted"
        abort_reason = "collapse_canary"

    print(f"\n{'='*70}")
    if status == "interrupted":
        signum = int(interrupted["signum"] or signal.SIGINT)
        print(f"Training interrupted ({signal.Signals(signum).name}).")
    elif status == "aborted":
        print(f"Training aborted ({abort_reason}).")
    else:
        print("Training complete.")
    print(f"  Traversals reached: {final_iter}")
    print(f"  Frozen rounds fit:  {round_index}")
    print(f"  Advantage steps:    {gradient_steps_taken}")
    print(f"  Optimizer batches:  {optimizer_steps_taken}")
    print(f"  Canary fail streak: {canary_fail_streak}")
    print(f"  Wall-clock:         {elapsed_total:.1f}s")
    print(f"  Last checkpoint:    {last_checkpoint_written or '<none>'}")
    print(f"{'='*70}")

    result = {
        "status": status,
        "final_iter": final_iter,
        "round_index": round_index,
        "gradient_steps": gradient_steps_taken,
        "gradient_steps_taken": gradient_steps_taken,
        "optimizer_steps_taken": optimizer_steps_taken,
        "head_steps": head_steps,
        "nonfinite_skips": nonfinite["total"],
        "canary_fail_streak": canary_fail_streak,
        "losses": losses,
        "regret_buf": buffers["regret"],
        "strategy_buf": buffers["strategy"],
        "value_buf": buffers["value"],
        "sizing_buf": buffers["sizing"],
        "bot": bot,
        "optimizer": optimizers,
        "optimizers": optimizers,
        "elapsed": elapsed_total,
        "checkpoint_saved": last_checkpoint_written,
    }
    if interrupted["flag"]:
        result["signal"] = int(interrupted["signum"] or signal.SIGINT)
    if abort_reason:
        result["abort_reason"] = abort_reason
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train the schema-v2 multiway Deep CFR-inspired bot with "
            "frozen-policy external-sampling rounds"
        ))
    parser.add_argument("--variant", choices=["small", "large"], required=True)
    parser.add_argument(
        "--iterations",
        type=int,
        default=1_000_000,
        help="External-sampling traversals (default: 1,000,000)",
    )
    parser.add_argument(
        "--round-size",
        type=int,
        default=DEFAULT_ROUND_SIZE,
        help="Traversals collected under each frozen policy (default: 25,000)",
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=100,
        help="Traversals per refit optimizer batch at each round boundary",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--aivat-sims", type=int, default=500)
    parser.add_argument(
        "--curriculum-profile",
        choices=sorted(CURRICULUM_PROFILES),
        default="sixmax",
        help="Player-count curriculum profile",
    )
    parser.add_argument(
        "--canary-enforce-iteration",
        type=int,
        default=DEFAULT_CANARY_ENFORCE_ITERATION,
        help="First traversal where collapse failures count (default: 100,000)",
    )
    parser.add_argument(
        "--canary-fail-patience",
        type=int,
        default=DEFAULT_CANARY_FAIL_PATIENCE,
        help="Consecutive failing checkpoints before abort (default: 3)",
    )
    parser.add_argument(
        "--disable-collapse-canary",
        action="store_true",
        help="Skip checkpoint all-in collapse probes; intended only for smoke tests",
    )
    parser.add_argument("--save-path", type=str, default="models/deep_cfr_v2.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args(argv)
    for name in (
        "iterations",
        "round_size",
        "update_interval",
        "checkpoint_interval",
        "batch_size",
        "canary_fail_patience",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.canary_enforce_iteration < 0:
        parser.error("--canary-enforce-iteration must be non-negative")
    return args


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
