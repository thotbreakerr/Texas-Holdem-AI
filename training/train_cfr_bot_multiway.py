"""
Train the CFR bot for multi-player deep-stack conditions.

Six CFRBot instances (all sharing the same regret table) are placed on
seats P1–P6 simultaneously.  Every hand updates regrets from all six
perspectives, giving the table broader multi-way situation coverage.

This mirrors the real gameday tournament format:
  * 6 players per table (match the real 6-group tournament)
  * 1 000 chip starting stacks
  * 5/10 blinds with 1.5× escalation every 50 hands

Saves to a *separate* profile (models/cfr_regret_deep_v2.pkl) so the existing
heads-up profile (models/cfr_regret.pkl) is never touched.

Equity-cache optimisation
-------------------------
``_quick_equity`` is computed **once** per decision point (before the
iteration loop in ``CFRBot._run_iterations``), not per-action per-iteration.
This is already the design in CFRBot; this script simply inherits it.

Convergence note
----------------
In 6-way self-play the expected win-rate for each seat is ~16.7 %.  That is the
sign of healthy convergence, not a bug.  Track ``info_sets`` and
``total_iters`` (printed every 1 000 episodes).

Checkpoint
----------
* Loads  ``--profile``  (default: models/cfr_regret_deep_v2.pkl) on startup.
* Saves every ``--save_every`` episodes (default: 500) and at the end of
  training.

Usage
-----
    python training/train_cfr_bot_multiway.py
    python training/train_cfr_bot_multiway.py --tournaments 5000 --iterations 10
    python training/train_cfr_bot_multiway.py --profile models/cfr_deep_v2.pkl
    # exact (uncapped) AIVAT leaf enumeration, pre-2026-07 behavior:
    python training/train_cfr_bot_multiway.py --leaf_enum_cap 0 --leaf_sims 200
"""

import os
import random
import sys
import time

# Add project root so imports work from any working directory.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import argparse

from core.engine import Table, Seat, InProcessBot, _FULL_DECK
from core.table_order import advance_dealer_seat_index, normalize_dealer_seat_index
from core.bot_api import BotAdapter, PlayerView, Action
from bots.cfr_bot import CFRBot
from bots import escalate_blinds


# ---------------------------------------------------------------------------
#  Thin adapter so CFRBot plugs into the engine's InProcessBot / bot_for dict
# ---------------------------------------------------------------------------

class _CFRAdapter(BotAdapter):
    """Wraps a CFRBot to satisfy the BotAdapter interface."""

    def __init__(self, bot: CFRBot):
        self.bot = bot

    def act(self, view: PlayerView) -> Action:
        return self.bot.act(view)


# ---------------------------------------------------------------------------
#  Main training function
# ---------------------------------------------------------------------------

PLAYER_IDS = ["P1", "P2", "P3", "P4", "P5", "P6"]
NUM_PLAYERS = len(PLAYER_IDS)

BASE_SB = 5
BASE_BB = 10
BLIND_ESCALATION_EVERY = 50   # hands


def _format_card(card) -> str:
    return f"{card[0]}{card[1]}"


def _preview_first_flop(rng: random.Random, n_players: int) -> tuple:
    """Preview the first hand's flop cards without advancing ``rng``."""
    clone = random.Random()
    clone.setstate(rng.getstate())
    deck = list(_FULL_DECK)
    clone.shuffle(deck)
    for _ in range(n_players * 2):
        deck.pop()
    return tuple(deck.pop() for _ in range(3))


def _format_eta(seconds: float) -> str:
    """Human-readable remaining-time estimate."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def train_cfr_bot_multiway(
    num_tournaments: int = 50_000,
    chips_per_player: int = 1_000,
    iterations: int = 10,
    save_every: int = 500,
    profile_path: str = "models/cfr_regret_deep_v2.pkl",
    seed: int | None = None,
    leaf_sims: int = 100,
    leaf_enum_cap: int | None = 120,
) -> CFRBot:
    """
    Run multi-player CFR self-play for ``num_tournaments`` episodes.

    Args:
        num_tournaments:  Number of 6-player tournament episodes.
        chips_per_player: Starting chip stack per seat (default 1 000).
        iterations:       MCCFR traversals per decision point.
        save_every:       Persist the regret table every N episodes.
        profile_path:     Path for regret-table persistence.
        seed:             Optional base seed. Episode N uses seed + N - 1.
        leaf_sims:        Preflop Monte Carlo sims per AIVAT leaf.
        leaf_enum_cap:    Max turn/flop runouts per AIVAT leaf
                          (None = exact enumeration, ~990 on the flop).

    Returns:
        The trained CFRBot instance.
    """
    print("=" * 70)
    print("TRAINING CFR BOT  (6-player deep-stack self-play)")
    print("=" * 70)
    print(f"Episodes:         {num_tournaments}")
    print(f"Players:          {NUM_PLAYERS}  ({', '.join(PLAYER_IDS)})")
    print(f"Chips per player: {chips_per_player}")
    print(f"Base blinds:      {BASE_SB}/{BASE_BB}")
    print(f"Blind escalation: every {BLIND_ESCALATION_EVERY} hands (×1.5)")
    print(f"Iterations/pt:    {iterations}  (MCCFR traversals per decision)")
    print(f"Leaf sims:        {leaf_sims}  (preflop MC per AIVAT leaf)")
    print(f"Leaf enum cap:    {leaf_enum_cap if leaf_enum_cap is not None else 'exact (no cap)'}"
          f"  (turn/flop runouts per AIVAT leaf)")
    print(f"Save every:       {save_every} episodes")
    print(f"Profile path:     {profile_path}")
    if seed is None:
        print("Deck RNG:         system entropy per episode")
    else:
        print(f"Deck RNG:         reproducible per episode from seed={seed}")
    print("=" * 70)
    print()

    if seed is not None:
        random.seed(seed)

    # ── Build bot (constructor auto-loads profile if it exists) ──────────────
    bot = CFRBot(
        iterations=iterations,
        profile_path=profile_path,
        use_average=True,
        leaf_sims=leaf_sims,
        leaf_enum_cap=leaf_enum_cap,
    )

    loaded_stats = bot.stats()
    if loaded_stats["info_sets"] > 0:
        print(
            f"Resumed from {profile_path}: "
            f"{loaded_stats['info_sets']} info sets, "
            f"{loaded_stats['total_iterations']} total iterations.\n"
        )
    else:
        print("No existing profile found — starting fresh.\n")

    # ── One shared adapter: all 6 seats reference the same CFRBot ────────────
    adapter = _CFRAdapter(bot)

    wins = {pid: 0 for pid in PLAYER_IDS}
    first_flop_previews = []

    # ── Main training loop ───────────────────────────────────────────────────
    t_train_start = time.monotonic()
    try:
        for episode in range(1, num_tournaments + 1):
            ep_start = time.monotonic()
            episode_seed = seed + episode - 1 if seed is not None else None
            table_rng = (
                random.Random(episode_seed)
                if episode_seed is not None
                else random.Random()
            )
            if len(first_flop_previews) < 10:
                first_flop_previews.append(
                    (episode, _preview_first_flop(table_rng, NUM_PLAYERS))
                )
            table = Table(rng=table_rng)

            # Fresh chip stacks each episode
            seats = [Seat(player_id=pid, chips=chips_per_player) for pid in PLAYER_IDS]

            # All seats share the same adapter (and therefore the same regret table)
            bots = {pid: InProcessBot(adapter) for pid in PLAYER_IDS}

            # ── Play hands until one player remains ──────────────────────────
            dealer_index = 0
            hand_count = 0
            winner = None

            while True:
                active_seats = [s for s in seats if s.chips > 0]
                if len(active_seats) <= 1:
                    winner = active_seats[0].player_id if active_seats else None
                    break

                # Blind escalation: 1.5× every BLIND_ESCALATION_EVERY hands
                sb, bb = escalate_blinds(
                    hand_count + 1,
                    BASE_SB,
                    BASE_BB,
                    BLIND_ESCALATION_EVERY,
                )
                dealer_index = normalize_dealer_seat_index(seats, dealer_index)
                if dealer_index is None:
                    winner = None
                    break

                table.play_hand(
                    seats=seats,
                    small_blind=sb,
                    big_blind=bb,
                    dealer_index=dealer_index,
                    bot_for={s.player_id: bots[s.player_id] for s in active_seats},
                    on_event=None,
                    log_decisions=False,
                )

                next_dealer = advance_dealer_seat_index(seats, dealer_index)
                if next_dealer is not None:
                    dealer_index = next_dealer
                hand_count += 1

                if hand_count > 10_000:      # safety cap
                    winner = max(seats, key=lambda s: s.chips).player_id
                    break

            # ── Episode bookkeeping ──────────────────────────────────────────
            if winner and winner in wins:
                wins[winner] += 1

            # ── Periodic save ────────────────────────────────────────────────
            if episode % save_every == 0:
                bot.save(profile_path)

            # ── Progress report: every episode for the first 10 (heartbeat
            # with wall time + ETA so a slow config is visible immediately),
            # then every 100 episodes (first 1k) or 1000. ────────────────
            report_interval = 100 if episode <= 1_000 else 1_000
            if episode <= 10 or episode % report_interval == 0:
                s = bot.stats()
                win_rates = "  ".join(
                    f"{pid}={wins[pid]/episode:.1%}" for pid in PLAYER_IDS
                )
                ep_time = time.monotonic() - ep_start
                avg_ep = (time.monotonic() - t_train_start) / episode
                eta = _format_eta(avg_ep * (num_tournaments - episode))
                print(
                    f"  ep={episode:>7}  "
                    f"{ep_time:6.1f}s/ep  ETA={eta:<7}  "
                    f"info_sets={s['info_sets']:<7}  "
                    f"total_iters={s['total_iterations']:<10}  "
                    f"recursion_calls={s.get('recursion_calls', '?'):<8}  "
                    f"{win_rates}",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\n[Training interrupted by user — saving checkpoint …]")
    finally:
        # ── End of training (normal finish or Ctrl+C) ─────────────────────────
        bot.save(profile_path)

        final_stats = bot.stats()
        print(f"\n{'=' * 70}")
        print(f"Training complete.")
        print(f"  Episodes:       {num_tournaments}")
        for pid in PLAYER_IDS:
            ep_played = sum(wins.values())  # episodes where a winner was recorded
            wr = wins[pid] / num_tournaments if num_tournaments > 0 else 0.0
            print(f"  {pid} wins:      {wins[pid]} / {num_tournaments}  ({wr:.1%})")
        print(f"  Info sets:      {final_stats['info_sets']}")
        print(f"  Total iters:    {final_stats['total_iterations']}")
        print(f"  Profile saved:  {profile_path}")
        if first_flop_previews:
            preview_text = ", ".join(
                f"ep{ep}:{'/'.join(_format_card(c) for c in flop)}"
                for ep, flop in first_flop_previews
            )
            print(f"  First flop previews: {preview_text}")
        print(f"{'=' * 70}")

    return bot


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CFR bot via 6-player deep-stack self-play (MCCFR)"
    )
    parser.add_argument(
        "--tournaments", type=int, default=50_000,
        help="Number of 6-player tournament episodes (default: 50000)"
    )
    parser.add_argument(
        "--chips", type=int, default=1_000,
        help="Starting chips per player (default: 1000)"
    )
    parser.add_argument(
        "--iterations", type=int, default=10,
        help="MCCFR traversals per decision point (default: 10; each "
             "traversal fully recurses the tree, so this is expensive)"
    )
    parser.add_argument(
        "--leaf_sims", type=int, default=100,
        help="Preflop Monte Carlo sims per AIVAT leaf (default: 100)"
    )
    parser.add_argument(
        "--leaf_enum_cap", type=int, default=120,
        help="Max turn/flop runouts settled per AIVAT leaf (default: 120; "
             "0 = exact enumeration, ~990 boards per flop leaf — slow)"
    )
    parser.add_argument(
        "--save_every", type=int, default=500,
        help="Save regret table every N episodes (default: 500)"
    )
    parser.add_argument(
        "--profile", type=str, default="models/cfr_regret_deep_v2.pkl",
        help="Path for regret-table persistence (default: models/cfr_regret_deep_v2.pkl)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Optional base seed for reproducible per-episode deck RNG"
    )
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)

    train_cfr_bot_multiway(
        num_tournaments=args.tournaments,
        chips_per_player=args.chips,
        iterations=args.iterations,
        save_every=args.save_every,
        profile_path=args.profile,
        seed=args.seed,
        leaf_sims=args.leaf_sims,
        leaf_enum_cap=args.leaf_enum_cap if args.leaf_enum_cap > 0 else None,
    )
