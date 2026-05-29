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

# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

N_SEATS = 6
START_CHIPS = 1000
BIG_BLIND = 10
SMALL_BLIND = 5
ALL_IN_WARMUP_ITERATIONS = 100_000
ALL_IN_DEPLOY_ITERATION = 150_000
ALL_IN_FULL_RELEASE_ITERATION = 350_000
CANARY_PASS_SEARCH_MAX = 0.15
CANARY_PASS_RAW_MAX = 0.30
CANARY_FAIL_SEARCH_MIN = 0.35
CANARY_FAIL_RAW_MIN = 0.60


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

def build_initial_state(n_seats: int = N_SEATS, hero_seat: int = 0) -> _DeepCFRGameState:
    """Construct a fresh preflop _DeepCFRGameState with random hole cards."""
    deck = list(_FULL_DECK)
    random.shuffle(deck)

    hole_cards = {}
    for seat in range(n_seats):
        hole_cards[seat] = (deck.pop(), deck.pop())

    stacks = [START_CHIPS] * n_seats
    committed = [0] * n_seats

    # Post blinds (seats 1=SB, 2=BB for 6-handed)
    sb_seat = 1 % n_seats
    bb_seat = 2 % n_seats
    sb_amt = min(SMALL_BLIND, stacks[sb_seat])
    bb_amt = min(BIG_BLIND, stacks[bb_seat])
    stacks[sb_seat] -= sb_amt
    committed[sb_seat] = sb_amt
    stacks[bb_seat] -= bb_amt
    committed[bb_seat] = bb_amt
    pot = sb_amt + bb_amt

    # Preflop action order: UTG first (seat 3 for 6-handed), then 4, 5, 0, 1, 2
    seat_order = [(3 + i) % n_seats for i in range(n_seats)]

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
        ring_order=list(range(n_seats)),
    )


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
    if total_loss.requires_grad:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=10.0)
        optimizer.step()

    return r_loss_val, v_loss_val, s_loss_val, {
        "heads_trained": heads_trained,
        "did_step": did_step,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Checkpoint save/load (atomic via .tmp + os.replace)
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(path: str, iteration: int, bot: DeepCFRBot,
                    optimizer: torch.optim.Optimizer, losses: dict):
    """Atomic save of training state."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save({
        "iteration": iteration,
        "config": bot.config,
        "network_state_dict": bot.network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "all_in_warmup_iterations": bot.all_in_warmup_iterations,
        "all_in_deploy_iteration": bot.all_in_deploy_iteration,
        "all_in_full_release_iteration": bot.all_in_full_release_iteration,
        "losses": {k: v[-100:] for k, v in losses.items()},  # last 100 only
    }, tmp_path)
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


def save_promoted_checkpoint(path: str, iteration: int, bot: DeepCFRBot,
                             optimizer: torch.optim.Optimizer, losses: dict,
                             status: str) -> str:
    """Save according to canary promotion rules and return the written path."""
    if status == "PASS":
        save_checkpoint(path, iteration, bot, optimizer, losses)
        save_checkpoint(safe_checkpoint_path(path), iteration, bot, optimizer, losses)
        return path
    if status == "WARN":
        side_path = warn_checkpoint_path(path, iteration)
        save_checkpoint(side_path, iteration, bot, optimizer, losses)
        return side_path
    raise RuntimeError(f"cannot save checkpoint for status {status!r}")


def load_checkpoint(path: str, bot: DeepCFRBot,
                    optimizer: torch.optim.Optimizer) -> int:
    """Restore training state. Returns iteration number."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
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


def _canary_views(n: int, seed: int) -> list[PlayerView]:
    rng = random.Random(seed)
    views = []
    for _ in range(n):
        deck = list(_FULL_DECK)
        rng.shuffle(deck)
        hole = [deck.pop(), deck.pop()]
        opponents = [f"opp{i}" for i in range(1, N_SEATS)]
        stacks = {"hero": START_CHIPS}
        for opp in opponents:
            stacks[opp] = START_CHIPS

        legal = [
            {"type": "fold"},
            {"type": "call"},
            {
                "type": "raise",
                "min": BIG_BLIND * 2,
                "max": START_CHIPS,
            },
        ]
        views.append(PlayerView(
            me="hero",
            street="preflop",
            position=rng.choice(["UTG", "MP", "CO", "BTN", "SB", "BB"]),
            hole_cards=hole,
            board=[],
            pot=SMALL_BLIND + BIG_BLIND,
            to_call=BIG_BLIND,
            min_raise=BIG_BLIND * 2,
            max_raise=START_CHIPS,
            legal_actions=legal,
            stacks=stacks,
            opponents=opponents,
            history=[],
        ))
    return views


def _canary_frequency_for_depth(
    bot: DeepCFRBot,
    views: list[PlayerView],
    *,
    search_depth: int,
    seed: int,
    current_iteration: int,
) -> float:
    old_training = bot.network.training
    old_inference_mode = bot.inference_mode
    old_weights_loaded = bot._weights_loaded
    old_search_depth = bot.search_depth
    old_training_iteration = bot.training_iteration
    old_guardrails_disabled = bot._all_in_guardrails_disabled
    old_opp_stats = bot._opp_stats
    old_history_len = bot._last_history_len
    old_history_snapshot = bot._last_history_snapshot
    old_random_state = random.getstate()

    all_in_count = 0
    try:
        random.seed(seed)
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
            if _canary_is_all_in(action, view):
                all_in_count += 1
    finally:
        random.setstate(old_random_state)
        bot.inference_mode = old_inference_mode
        bot._weights_loaded = old_weights_loaded
        bot.search_depth = old_search_depth
        bot.training_iteration = old_training_iteration
        bot._all_in_guardrails_disabled = old_guardrails_disabled
        bot._opp_stats = old_opp_stats
        bot._last_history_len = old_history_len
        bot._last_history_snapshot = old_history_snapshot
        if old_training:
            bot.network.train()
        else:
            bot.network.eval()

    return all_in_count / max(len(views), 1)


def quick_canary_probe(bot: DeepCFRBot, device: torch.device,
                       n: int = 50, seed: int = 20260428,
                       current_iteration: int = 0) -> dict[str, float]:
    """Return raw-policy and search-policy all-in frequencies."""
    _ = device  # kept for call-site/API clarity; bot.act handles device moves.
    views = _canary_views(n, seed)
    raw = _canary_frequency_for_depth(
        bot,
        views,
        search_depth=0,
        seed=seed + 1,
        current_iteration=current_iteration,
    )
    search = _canary_frequency_for_depth(
        bot,
        views,
        search_depth=bot.search_depth,
        seed=seed + 2,
        current_iteration=current_iteration,
    )
    return {"raw_all_in": raw, "search_all_in": search}


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

    start_iter = 0
    if args.resume and os.path.exists(args.resume):
        start_iter = load_checkpoint(args.resume, bot, optimizer)
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

    # SIGINT handler for safe checkpoint
    interrupted = [False]
    def _sigint_handler(sig, frame):
        interrupted[0] = True
        print("\n[train_deep_cfr] SIGINT received — saving checkpoint …")
    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _sigint_handler)

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
    last_checkpoint_iter = None
    last_checkpoint_written = None

    def checkpoint_with_canary(iteration: int) -> tuple[str, str]:
        if args.disable_collapse_canary:
            save_checkpoint(args.save_path, iteration, bot, optimizer, losses)
            return "DISABLED", args.save_path

        canary = quick_canary_probe(bot, device, current_iteration=iteration)
        raw_all_in = canary["raw_all_in"]
        search_all_in = canary["search_all_in"]
        status = classify_canary(raw_all_in, search_all_in)
        phase = all_in_phase_for_iteration(
            iteration,
            bot.all_in_warmup_iterations,
            bot.all_in_full_release_iteration,
        )
        if status == "FAIL":
            raise RuntimeError(
                f"All-in collapse at iter {iteration}: "
                f"phase={phase} search={search_all_in:.1%}, "
                f"raw={raw_all_in:.1%}"
            )
        if status == "WARN":
            print(
                f"[WARN] iter {iteration}: all-in freq "
                f"phase={phase} search={search_all_in:.1%}, "
                f"raw={raw_all_in:.1%} -- side checkpoint only",
                flush=True,
            )
        else:
            print(
                f"[CANARY] iter {iteration}: all-in freq "
                f"phase={phase} search={search_all_in:.1%}, "
                f"raw={raw_all_in:.1%}",
                flush=True,
            )
        written = save_promoted_checkpoint(
            args.save_path,
            iteration,
            bot,
            optimizer,
            losses,
            status,
        )
        return status, written

    try:
        for t in range(start_iter + 1, args.iterations + 1):
            if interrupted[0]:
                break

            hero_seat = (t - 1) % N_SEATS
            state = build_initial_state(n_seats=N_SEATS, hero_seat=hero_seat)

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

            # Periodic gradient steps
            if t % args.update_interval == 0:
                bot.network.train()
                r_loss, v_loss, s_loss, step_info = train_step(
                    bot.network, optimizer,
                    regret_buf, value_buf, sizing_buf,
                    args.batch_size, device,
                )
                losses["regret"].append(r_loss)
                losses["value"].append(v_loss)
                losses["sizing"].append(s_loss)
                if step_info["did_step"]:
                    optimizer_steps_taken += 1
                for head, trained in step_info["heads_trained"].items():
                    if trained:
                        head_steps[head] += 1
                if step_info["heads_trained"]["regret"]:
                    gradient_steps_taken += 1

            # Checkpoint
            if t % args.checkpoint_interval == 0:
                try:
                    _, last_checkpoint_written = checkpoint_with_canary(t)
                    last_checkpoint_iter = t
                except RuntimeError:
                    # All-in collapse canary tripped mid-run.  Break out of the
                    # loop rather than re-raising: the finally block prints the
                    # ABORT footer and the post-loop `abort_without_save` block
                    # returns status="aborted" (which main() maps to a nonzero
                    # exit).  Re-raising skipped that documented return path.
                    abort_without_save = True
                    break

            # Progress
            if t % max(1, args.update_interval) == 0:
                elapsed = _time.monotonic() - t0
                print_progress(t, regret_buf, value_buf, sizing_buf, losses,
                               elapsed)

    except KeyboardInterrupt:
        interrupted[0] = True
        print("\n[train_deep_cfr] KeyboardInterrupt — saving checkpoint …")
    finally:
        # Final save
        final_iter = min(t, args.iterations) if 't' in dir() else start_iter
        if abort_without_save:
            print(
                "[ABORT] Final checkpoint not saved because the all-in "
                "collapse canary tripped.",
                flush=True,
            )
        elif final_iter != last_checkpoint_iter:
            try:
                _, last_checkpoint_written = checkpoint_with_canary(final_iter)
            except RuntimeError as exc:
                print(f"[ABORT] Final checkpoint not saved: {exc}", flush=True)
                abort_without_save = True
        else:
            _ = last_checkpoint_written
        signal.signal(signal.SIGINT, original_handler)

    elapsed_total = _time.monotonic() - t0
    if gradient_steps_taken == 0:
        print(
            f"[WARN] Training completed with 0 gradient steps. "
            f"Buffer sizes at end: regret={len(regret_buf)}, "
            f"value={len(value_buf)}, sizing={len(sizing_buf)}. "
            f"Likely cause: --batch-size {args.batch_size} too large for "
            f"--iterations {args.iterations}. Try --batch-size 32 for short runs."
        )

    if abort_without_save:
        print(f"\n{'='*70}")
        print("Training aborted.")
        print(f"  Iterations reached: {final_iter}")
        print(f"  Gradient steps:     {gradient_steps_taken}")
        print(f"  Optimizer steps:    {optimizer_steps_taken}")
        print(f"  Wall-clock:         {elapsed_total:.1f}s")
        print(f"  Last checkpoint:    {last_checkpoint_written or '<none>'}")
        print(f"{'='*70}")
        return {
            "status": "aborted",
            "final_iter": final_iter,
            "gradient_steps": gradient_steps_taken,
            "gradient_steps_taken": gradient_steps_taken,
            "optimizer_steps_taken": optimizer_steps_taken,
            "head_steps": head_steps,
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
    parser.add_argument("--iterations", type=int, default=100_000)
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
    if result.get("status") != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
