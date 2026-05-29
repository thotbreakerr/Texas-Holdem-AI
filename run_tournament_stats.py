# run_tournament_stats.py — Run multiple tournaments and track detailed statistics

import argparse
import csv
import io
import os
import random
import time as _time
from collections import defaultdict
from contextlib import redirect_stdout
from multiprocessing import Pool

from core.engine import Table, Seat
from core.table_order import advance_dealer_seat_index, normalize_dealer_seat_index
from bots import parse_players, escalate_blinds, create_bot


# ── Single silent tournament ──────────────────────────────────────────────────

def maybe_disable_deep_cfr_search(adapter, bot_type: str):
    """Disable DeepCFR real-time search on an adapter, if it wraps DeepCFRBot."""
    key = bot_type.strip().lower()
    if not key.startswith(("deep_cfr", "deepcfr", "deep_cfr_bot")):
        return adapter
    inner = getattr(adapter, "bot", None)
    if hasattr(inner, "search_depth"):
        inner.search_depth = 0
    return adapter


def _advance_dealer(dealer_index: int, active_count: int) -> int:
    """Advance the dealer button by one in the ACTIVE seat circle.

    Bug fingerprint (now fixed): using % len(seats_total) instead of
    % active_count produces sequences with duplicate consecutive indices
    after eliminations (e.g. with 6 seats total but 5 active:
    [0,1,2,3,4,0,0,1,2,3] — note the double-0 at positions 5 and 6).
    Correct rotation: [0,1,2,3,4,0,1,2,3,4].
    """
    return (dealer_index + 1) % active_count

def finalize_finish_order(seats, finish_order, total_players, hand_count):
    """Append survivor results after a natural finish or safety stop."""
    survivors = [
        s for s in seats
        if s.chips > 0 and not any(e[0] == s.player_id for e in finish_order)
    ]

    if len(survivors) == 1:
        finish_order.append((survivors[0].player_id, 1, hand_count, survivors[0].chips))
    elif len(survivors) > 1:
        if not finish_order:
            # Match run_local_match.py: no eliminations means unfinished/unranked.
            for s in survivors:
                finish_order.append((s.player_id, 0, hand_count, s.chips))
        else:
            sorted_survivors = sorted(survivors, key=lambda s: s.chips, reverse=True)
            for rank_idx, s in enumerate(sorted_survivors):
                finish_order.append((s.player_id, rank_idx + 1, hand_count, s.chips))

    return finish_order

def run_silent_tournament(args_tuple):
    """Run one tournament silently. Accepts a tuple for multiprocessing.Pool.map.

    Returns dict with: winner, hand_count, finish_order [(pid, position, hand#, chips_at_elim)].
    """
    (player_specs, chips, base_sb, base_bb, blind_increase_every,
     max_hands, seed, disable_search) = args_tuple

    if seed is not None:
        random.seed(seed)

    # Rebuild bots in this process (can't pickle adapters across processes)
    bots = {}
    for pid, btype, _ in player_specs:
        adapter = create_bot(btype)
        if disable_search:
            adapter = maybe_disable_deep_cfr_search(adapter, btype)
        bots[pid] = adapter

    seats = [Seat(player_id=pid, chips=chips) for pid, _, _ in player_specs]
    table_rng = random.Random(seed) if seed is not None else random.Random()
    table = Table(rng=table_rng)
    dealer_index = 0
    hand_count = 0
    total_players = len(seats)
    finish_order: list[tuple[str, int, int, int]] = []  # (pid, pos, hand#, chips_at_elim)

    with redirect_stdout(io.StringIO()):
        while True:
            active_players = [s for s in seats if s.chips > 0]
            if len(active_players) <= 1:
                break

            hand_count += 1
            sb, bb = escalate_blinds(hand_count, base_sb, base_bb, blind_increase_every)
            active_seats = [s for s in seats if s.chips > 0]
            active_bots = {s.player_id: bots[s.player_id] for s in active_seats}
            dealer_index = normalize_dealer_seat_index(seats, dealer_index)
            if dealer_index is None:
                break

            table.play_hand(
                seats=seats,
                small_blind=sb,
                big_blind=bb,
                dealer_index=dealer_index,
                bot_for=active_bots,
                on_event=None,
            )
            next_dealer = advance_dealer_seat_index(seats, dealer_index)
            if next_dealer is not None:
                dealer_index = next_dealer

            # Track eliminations
            for s in seats:
                if s.chips <= 0 and not any(e[0] == s.player_id for e in finish_order):
                    pos = total_players - len(finish_order)
                    finish_order.append((s.player_id, pos, hand_count, 0))

            if hand_count >= max_hands:
                break

    finalize_finish_order(seats, finish_order, total_players, hand_count)

    winner = None
    for pid, pos, _, _ in finish_order:
        if pos == 1:
            winner = pid
            break

    return {
        "winner": winner,
        "hand_count": hand_count,
        "finish_order": finish_order,
    }


# ── Batch runner ──────────────────────────────────────────────────────────────

# ── GUI code run in a subprocess (macOS requires tkinter on the main thread) ──

_G   = "\033[92m"
_B   = "\033[1m"
_R   = "\033[0m"
_W   = 40


def _bar(done, total):
    filled = int(_W * done / total) if total else 0
    pct    = done / total * 100 if total else 0
    bar    = "█" * filled + "░" * (_W - filled)
    print(f"\r  {_G}{_B}[{bar}]{_R} {_G}{pct:5.1f}%{_R}  {done}/{total}",
          end="", flush=True)


def run_tournament_batch(player_spec_str, num_tournaments, chips, base_sb, base_bb,
                         blind_increase_every, max_hands, parallel, output_csv, seed,
                         disable_search=False):
    player_specs = parse_players(player_spec_str)
    if len(player_specs) < 2:
        print("Error: need at least 2 players.")
        return

    pids = [pid for pid, _, _ in player_specs]
    bot_types = {pid: btype for pid, btype, _ in player_specs}

    print("=" * 75)
    print(f"RUNNING {num_tournaments} TOURNAMENTS")
    print("=" * 75)
    print(f"Players: {', '.join(f'{pid}={btype}' for pid, btype, _ in player_specs)}")
    print(f"Chips: {chips}  |  Blinds: {base_sb}/{base_bb}  |  "
          f"Escalation every {blind_increase_every} hands")
    if parallel > 1:
        print(f"Parallel workers: {parallel}")
    if disable_search:
        print("DeepCFR search: disabled (raw regret-matched policy)")
    print("=" * 75)
    print()

    # Build args tuples for each tournament
    tasks = []
    for i in range(num_tournaments):
        t_seed = (seed + i) if seed is not None else None
        tasks.append((player_specs, chips, base_sb, base_bb,
                      blind_increase_every, max_hands, t_seed, disable_search))

    # Run tournaments
    results = []
    _bar(0, num_tournaments)
    if parallel > 1:
        with Pool(processes=parallel) as pool:
            for i, res in enumerate(pool.imap_unordered(run_silent_tournament, tasks), 1):
                results.append(res)
                _bar(i, num_tournaments)
    else:
        for i, task in enumerate(tasks, 1):
            res = run_silent_tournament(task)
            results.append(res)
            _bar(i, num_tournaments)
    print()

    # ── Aggregate statistics ──────────────────────────────────────────────────

    wins = defaultdict(int)
    finish_positions = defaultdict(list)       # pid -> [positions]
    chips_at_elimination = defaultdict(list)   # pid -> [chips when eliminated]
    hands_survived = defaultdict(list)         # pid -> [hand# when eliminated]
    h2h_wins = defaultdict(lambda: defaultdict(int))  # pid_a -> pid_b -> count a beat b

    hand_counts = []

    for res in results:
        hand_counts.append(res["hand_count"])
        if res["winner"] in pids:
            wins[res["winner"]] += 1

        fo = res["finish_order"]  # [(pid, pos, hand#, chips)]

        for pid, pos, hand, elim_chips in fo:
            finish_positions[pid].append(pos)
            hands_survived[pid].append(hand)
            if pos > 1:
                chips_at_elimination[pid].append(elim_chips)

            # Head-to-head: count wins against each opponent who finished worse
            for other_pid, other_pos, _, _ in fo:
                if other_pid != pid and pos < other_pos:
                    h2h_wins[pid][other_pid] += 1

    # ── Print results ─────────────────────────────────────────────────────────

    print("\n" + "=" * 75)
    print("LEADERBOARD (sorted by win rate)")
    print("=" * 75)

    header = (f"{'#':<4} {'Player':<8} {'Bot':<14} {'Wins':>6} {'Win%':>7} "
              f"{'Avg Pos':>8} {'Avg Elim $':>10} "
              f"{'Hands (avg)':>12} {'(min)':>7} {'(max)':>7}")
    print(header)
    print("-" * 75)

    # Sort by win rate desc
    sorted_pids = sorted(pids, key=lambda p: wins[p], reverse=True)

    for rank, pid in enumerate(sorted_pids, 1):
        btype = bot_types[pid]
        w = wins[pid]
        wr = (w / num_tournaments) * 100
        avg_pos = sum(finish_positions[pid]) / len(finish_positions[pid])
        elim_chips = chips_at_elimination[pid]
        avg_elim = sum(elim_chips) / len(elim_chips) if elim_chips else 0
        hs = hands_survived[pid]
        avg_h = sum(hs) / len(hs) if hs else 0
        min_h = min(hs) if hs else 0
        max_h = max(hs) if hs else 0

        print(f"{rank:<4} {pid:<8} {btype:<14} {w:>6} {wr:>6.1f}% "
              f"{avg_pos:>8.2f} {avg_elim:>10.0f} "
              f"{avg_h:>12.1f} {min_h:>7} {max_h:>7}")

    # Hand count stats
    print(f"\n{'Tournaments:':<25} {num_tournaments}")
    print(f"{'Avg hands/tournament:':<25} {sum(hand_counts)/len(hand_counts):.1f}")
    print(f"{'Shortest:':<25} {min(hand_counts)}")
    print(f"{'Longest:':<25} {max(hand_counts)}")

    # Head-to-head matrix
    print("\n" + "=" * 75)
    print("HEAD-TO-HEAD WIN RATES")
    print("=" * 75)

    # Header row
    col_w = 10
    print(f"{'':>{col_w}}", end="")
    for pid in pids:
        print(f"{pid:>{col_w}}", end="")
    print()

    for pid_a in pids:
        print(f"{pid_a:>{col_w}}", end="")
        for pid_b in pids:
            if pid_a == pid_b:
                print(f"{'---':>{col_w}}", end="")
            else:
                total = h2h_wins[pid_a][pid_b] + h2h_wins[pid_b][pid_a]
                if total > 0:
                    rate = (h2h_wins[pid_a][pid_b] / total) * 100
                    print(f"{rate:>{col_w - 1}.0f}%", end="")
                else:
                    print(f"{'N/A':>{col_w}}", end="")
        print()

    print("=" * 75)

    # ── CSV output ────────────────────────────────────────────────────────────

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tournament", "winner", "hands",
                             *[f"{pid}_position" for pid in pids],
                             *[f"{pid}_hands_survived" for pid in pids]])
            for i, res in enumerate(results, 1):
                fo = {pid: (pos, hand) for pid, pos, hand, _ in res["finish_order"]}
                row = [i, res["winner"], res["hand_count"]]
                for pid in pids:
                    pos, hand = fo.get(pid, (0, 0))
                    row.append(pos)
                for pid in pids:
                    pos, hand = fo.get(pid, (0, 0))
                    row.append(hand)
                writer.writerow(row)
        print(f"\nResults saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple Texas Hold'em tournaments and track statistics")
    parser.add_argument("--players", type=str,
                        default="smart,cfr,gto,icm,exploitative,opponentmodel,mc200",
                        help="Comma-separated bot types (default: smart,cfr,gto,icm,exploitative,opponentmodel,mc200)")
    parser.add_argument("--tournaments", type=int, default=100,
                        help="Number of tournaments (default: 100)")
    parser.add_argument("--chips", type=int, default=500,
                        help="Starting chips per player (default: 500)")
    parser.add_argument("--sb", type=int, default=1,
                        help="Starting small blind (default: 1)")
    parser.add_argument("--bb", type=int, default=2,
                        help="Starting big blind (default: 2)")
    parser.add_argument("--blind-increase-every", type=int, default=50,
                        help="Increase blinds 1.5x every N hands, 0 to disable (default: 50)")
    parser.add_argument("--max-hands", type=int, default=10000,
                        help="Safety hand limit per tournament (default: 10000)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducibility")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel workers (default: 1, sequential)")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Save per-tournament results to CSV file")
    parser.add_argument("--disable-search", action="store_true",
                        help="Disable DeepCFR real-time search; use raw policy")
    parser.add_argument("--rl_model", type=str, default=None,
                        help="Path to RL model weights (e.g. models/rl_model_run3.pt). "
                             "Rewrites any 'rl' entry in --players to use this model.")
    args = parser.parse_args()

    if args.rl_model:
        import re
        args.players = re.sub(r'(?<![:\w])rl(?![\w:])', f'rl:{args.rl_model}', args.players)

    run_tournament_batch(
        player_spec_str=args.players,
        num_tournaments=args.tournaments,
        chips=args.chips,
        base_sb=args.sb,
        base_bb=args.bb,
        blind_increase_every=args.blind_increase_every,
        max_hands=args.max_hands,
        parallel=args.parallel,
        output_csv=args.output_csv,
        seed=args.seed,
        disable_search=args.disable_search,
    )


if __name__ == "__main__":
    main()
