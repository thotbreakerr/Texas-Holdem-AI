#!/usr/bin/env python3
"""
run_eval.py — Path A vs Path B tournament evaluation harness.

Runs either head-to-head Path A/Path B matches or multiway tournaments against
the strong bot pool, then reports tournament-aware metrics with Wilson 95%
confidence intervals.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import random
import statistics
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any, NamedTuple

from core.engine import Seat
from core.tournament import run_tournament
from bots import create_bot


PATH_A = "PATH_A"
PATH_B = "PATH_B"
DEFAULT_POOL = "smart,mc200,gto,icm,exploitative,opponentmodel"
PILOT_BASELINE = 1.0 / 6.0


@dataclass(frozen=True)
class EvalConfig:
    mode: str
    path_a_profile: str
    path_b_weights: str
    tournaments: int = 1000
    pool: str = DEFAULT_POOL
    chips: int = 500
    sb: int = 5
    bb: int = 10
    blind_increase_every: int = 50
    max_hands: int = 10000
    seed: int | None = None
    output_csv: str | None = None
    parallel: int = 1
    promotion_opponent: str = "gto"
    ante_mode: str = "off"
    ante_fraction_of_bb: float = 0.0


class TournamentTask(NamedTuple):
    player_specs: list[tuple[str, str]]
    chips: int
    base_sb: int
    base_bb: int
    blind_increase_every: int
    max_hands: int
    seed: int | None
    ante_mode: str
    ante_fraction_of_bb: float


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Standard Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2.0 * n)
    margin = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)) ** 0.5)
    low = (center - margin) / denom
    high = (center + margin) / denom
    return (max(0.0, low), min(1.0, high))


def head_to_head_verdict(a_ci: tuple[float, float],
                         b_ci: tuple[float, float]) -> str:
    """Declare a decisive H2H winner only when its lower CI bound exceeds 50%."""
    if a_ci[0] > 0.5:
        return "Production CFR: PATH_A"
    if b_ci[0] > 0.5:
        return "Production CFR: PATH_B"
    return "No decisive winner — within statistical noise"


def production_verdict(summary: dict[str, dict[str, Any]]) -> str:
    """Production CFR decision rule for multiway Path A vs Path B comparison."""
    a = summary.get(PATH_A)
    b = summary.get(PATH_B)
    if not a or not b:
        return "No decisive winner — manual review required"

    a_lo, a_hi = a["win_ci"]
    b_lo, b_hi = b["win_ci"]
    if a["win_rate"] > b["win_rate"] and a_lo > b_hi:
        return "Production CFR: PATH_A"
    if b["win_rate"] > a["win_rate"] and b_lo > a_hi:
        return "Production CFR: PATH_B"

    a_pos = a["avg_position"]
    b_pos = b["avg_position"]
    if a_pos is not None and b_pos is not None:
        if a_pos < b_pos:
            return "Production CFR: PATH_A"
        if b_pos < a_pos:
            return "Production CFR: PATH_B"

    return "Production CFR: PATH_B"


def pilot_verdict(summary: dict[str, dict[str, Any]]) -> str:
    """Pass unless Path B's 95% interval is wholly below random-seat 1/6."""
    deep = summary.get(PATH_B)
    if not deep:
        return "FAIL: PATH_B result missing"
    if deep["win_ci"][1] < PILOT_BASELINE:
        return "FAIL: PATH_B is statistically worse than the 1/6 baseline"
    return "PASS: PATH_B is not statistically worse than the 1/6 baseline"


def promotion_verdict(summary: dict[str, dict[str, Any]]) -> str:
    """Pass unless Path B's 95% interval is wholly below 50% head-to-head."""
    deep = summary.get(PATH_B)
    if not deep:
        return "FAIL: PATH_B result missing"
    if deep["win_ci"][1] < 0.5:
        return "FAIL: PATH_B is statistically worse than the promotion opponent"
    return "PASS: PATH_B confidence interval shows no promotion regression"


def _require_canary_clean_checkpoint(path: str) -> None:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("schema_version") != 2
            or payload.get("canary_status") != "PASS"):
        raise RuntimeError(
            "promotion mode requires a schema-v2 checkpoint stamped "
            "canary_status=PASS")


def _pool_specs(pool: str) -> list[tuple[str, str]]:
    """Return unique (pid, bot_spec) entries for the comma-separated pool."""
    specs = []
    seen = defaultdict(int)
    for raw in [p.strip() for p in pool.split(",") if p.strip()]:
        base = raw.upper().replace(":", "_").replace("/", "_").replace(".", "_")
        seen[base] += 1
        pid = base if seen[base] == 1 else f"{base}_{seen[base]}"
        specs.append((pid, raw))
    return specs


def build_player_specs(config: EvalConfig) -> list[tuple[str, str]]:
    """Build (pid, bot_spec) entries for the selected eval mode."""
    if config.mode == "head_to_head":
        return [
            (PATH_A, f"cfr:{config.path_a_profile}"),
            (PATH_B, f"deep_cfr:{config.path_b_weights}"),
        ]
    if config.mode == "multiway":
        return [
            (PATH_A, f"cfr:{config.path_a_profile}"),
            (PATH_B, f"deep_cfr:{config.path_b_weights}"),
            *_pool_specs(config.pool),
        ]
    if config.mode == "pilot":
        pool = _pool_specs(config.pool)
        if len(pool) < 5:
            raise ValueError("pilot mode requires at least five pool opponents")
        return [
            (PATH_B, f"deep_cfr:{config.path_b_weights}"),
            *pool[:5],
        ]
    if config.mode == "promotion":
        _require_canary_clean_checkpoint(config.path_b_weights)
        return [
            (PATH_B, f"deep_cfr:{config.path_b_weights}"),
            ("PROMOTION_OPPONENT", config.promotion_opponent),
        ]
    if config.mode == "curriculum":
        return [(PATH_B, f"deep_cfr:{config.path_b_weights}")]
    raise ValueError(f"unknown mode: {config.mode}")


def _make_bots(player_specs: list[tuple[str, str]]) -> dict[str, Any]:
    # NOTE: bots are intentionally reconstructed per tournament.  Stateful bots
    # (CFR/Deep CFR opponent-stat trackers) must start each tournament fresh, so
    # we deliberately do not cache bot instances across tournaments.  The cost
    # is re-deserialising model weights from disk each tournament; in parallel
    # mode (Pool) each worker pays this once per task.  If this dominates a very
    # large run, cache the immutable loaded weights (not the bot instances) at
    # the create_bot layer rather than reusing bots here.
    bots = {}
    for pid, bot_spec in player_specs:
        with redirect_stdout(io.StringIO()):
            bots[pid] = create_bot(bot_spec)
    return bots


def _ante_amount(big_blind: int, fraction: float) -> int:
    return max(0, int(big_blind * fraction))


def _ante_label(config: EvalConfig) -> str:
    if config.ante_mode == "off":
        return "off"
    return f"{config.ante_mode}:{config.ante_fraction_of_bb:.6g}bb"


def _ante_kwargs(ante_mode: str, fraction: float, base_bb: int) -> dict[str, Any]:
    if ante_mode == "off":
        return {"ante": 0}
    if ante_mode == "fixed":
        return {"ante": _ante_amount(base_bb, fraction)}
    if ante_mode == "schedule":
        return {
            "ante_schedule": (
                lambda _hand, _sb, bb: _ante_amount(bb, fraction)
            )
        }
    raise ValueError(f"unknown ante_mode: {ante_mode!r}")


def _run_one_tournament(task: TournamentTask) -> dict[str, Any]:
    if task.seed is not None:
        random.seed(task.seed)

    bots = _make_bots(task.player_specs)
    seats = [Seat(player_id=pid, chips=task.chips) for pid, _ in task.player_specs]
    result = run_tournament(
        seats,
        bots,
        small_blind=task.base_sb,
        big_blind=task.base_bb,
        blind_increase_every=task.blind_increase_every,
        max_hands=task.max_hands,
        dealer_index=0,
        dealer_rotation="full_table",
        winner_resolution="chip_count_on_max_hands",
        rng=random.Random(task.seed) if task.seed is not None else random.Random(),
        suppress_output=True,
        log_decisions=False,
        **_ante_kwargs(task.ante_mode, task.ante_fraction_of_bb, task.base_bb),
    )
    result.pop("chip_history", None)
    return result


def _run_all_tournaments(config: EvalConfig,
                         player_specs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    tasks: list[TournamentTask] = []
    for i in range(config.tournaments):
        t_seed = config.seed + i if config.seed is not None else None
        rotation = i % len(player_specs)
        rotated_specs = player_specs[rotation:] + player_specs[:rotation]
        tasks.append(TournamentTask(
            player_specs=rotated_specs,
            chips=config.chips,
            base_sb=config.sb,
            base_bb=config.bb,
            blind_increase_every=config.blind_increase_every,
            max_hands=config.max_hands,
            seed=t_seed,
            ante_mode=config.ante_mode,
            ante_fraction_of_bb=config.ante_fraction_of_bb,
        ))

    if config.parallel > 1:
        with Pool(processes=config.parallel) as pool:
            return list(pool.map(_run_one_tournament, tasks))
    return [_run_one_tournament(task) for task in tasks]


def aggregate_results(results: list[dict[str, Any]],
                      player_specs: list[tuple[str, str]]) -> dict[str, Any]:
    pids = [pid for pid, _ in player_specs]
    bot_types = {pid: spec for pid, spec in player_specs}
    n = len(results)
    itm_cutoff = 3 if len(pids) >= 6 else 1

    wins = defaultdict(int)
    finish_positions = defaultdict(list)
    hands_survived = defaultdict(list)
    itm_counts = defaultdict(int)
    h2h_wins = defaultdict(lambda: defaultdict(int))
    h2h_totals = defaultdict(lambda: defaultdict(int))
    hand_counts = []
    chip_swings = []

    for res in results:
        hand_counts.append(res["hand_count"])
        if res["chip_swing"] is not None:
            chip_swings.append(res["chip_swing"])
        if res["winner"] in pids:
            wins[res["winner"]] += 1

        fo = {pid: (pos, hand, chips) for pid, pos, hand, chips in res["finish_order"]}
        for pid in pids:
            pos, hand, _ = fo.get(pid, (0, res["hand_count"], 0))
            finish_positions[pid].append(pos)
            hands_survived[pid].append(hand)
            if 0 < pos <= itm_cutoff:
                itm_counts[pid] += 1

        for a in pids:
            a_pos = fo.get(a, (0, 0, 0))[0]
            for b in pids:
                if a == b:
                    continue
                b_pos = fo.get(b, (0, 0, 0))[0]
                if a_pos > 0 and b_pos > 0 and a_pos != b_pos:
                    h2h_totals[a][b] += 1
                    if a_pos < b_pos:
                        h2h_wins[a][b] += 1

    summary = {}
    for pid in pids:
        positions = finish_positions[pid]
        positive_positions = [p for p in positions if p > 0]
        hs = hands_survived[pid]
        w = wins[pid]
        summary[pid] = {
            "bot": bot_types[pid],
            "wins": w,
            "win_rate": w / n if n else 0.0,
            "win_ci": wilson_ci(w, n),
            "avg_position": (
                sum(positive_positions) / len(positive_positions)
                if positive_positions else None
            ),
            "itm_rate": itm_counts[pid] / n if n else 0.0,
            "hands_avg": sum(hs) / len(hs) if hs else 0.0,
            "hands_median": statistics.median(hs) if hs else 0.0,
            "hands_max": max(hs) if hs else 0,
        }

    h2h_matrix = {}
    for a in pids:
        h2h_matrix[a] = {}
        for b in pids:
            if a == b:
                h2h_matrix[a][b] = None
                continue
            total = h2h_totals[a][b]
            h2h_matrix[a][b] = (
                h2h_wins[a][b] / total if total else None
            )

    return {
        "players": pids,
        "bot_types": bot_types,
        "summary": summary,
        "h2h_matrix": h2h_matrix,
        "hand_counts": hand_counts,
        "chip_swings": chip_swings,
        "results": results,
    }


def _fmt_ci(ci: tuple[float, float]) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def print_report(config: EvalConfig, aggregated: dict[str, Any]) -> str:
    """Print and return the verdict line."""
    summary = aggregated["summary"]
    pids = aggregated["players"]
    hand_counts = aggregated["hand_counts"]

    print("=" * 80)
    print(f"EVAL REPORT — {config.mode}")
    print("=" * 80)
    print(f"Tournaments: {config.tournaments}")
    print(f"Players: {', '.join(pids)}")
    print(f"Chips: {config.chips} | Blinds: {config.sb}/{config.bb} | "
          f"Blind increase every: {config.blind_increase_every} | "
          f"Ante: {_ante_label(config)}")
    print("Selection metric: tournament first-place rate (win_rate)")
    if hand_counts:
        print(f"Hands/match: avg={sum(hand_counts)/len(hand_counts):.1f} "
              f"min={min(hand_counts)} max={max(hand_counts)}")
    print()

    if config.mode in {"head_to_head", "promotion"}:
        if config.mode == "promotion":
            deep = summary[PATH_B]
            opponent = summary["PROMOTION_OPPONENT"]
            for pid, row in ((PATH_B, deep), ("PROMOTION_OPPONENT", opponent)):
                print(
                    f"  {pid}: wins={row['wins']}/{config.tournaments} "
                    f"win_rate={row['win_rate']:.3f} "
                    f"Wilson95={_fmt_ci(row['win_ci'])}"
                )
            verdict = promotion_verdict(summary)
            print(f"Verdict: {verdict}")
            print("=" * 80)
            return verdict
        a = summary[PATH_A]
        b = summary[PATH_B]
        print("HEAD-TO-HEAD")
        for pid, row in ((PATH_A, a), (PATH_B, b)):
            print(
                f"  {pid}: wins={row['wins']}/{config.tournaments} "
                f"win_rate={row['win_rate']:.3f} "
                f"Wilson95={_fmt_ci(row['win_ci'])}"
            )
        swings = aggregated["chip_swings"]
        if swings:
            print(
                f"  Avg chip swing: {sum(swings)/len(swings):.1f} "
                f"(n={len(swings)})"
            )
        verdict = head_to_head_verdict(a["win_ci"], b["win_ci"])
        print(f"Verdict: {verdict}")
        print("=" * 80)
        return verdict

    print("RANK TABLE")
    print(f"{'Rank':<5} {'Bot':<16} {'Wins':>7} {'WinRate':>9} "
          f"{'Wilson95':>18} {'AvgPos':>8} {'ITM':>8} "
          f"{'HandsAvg':>9} {'HandsMed':>9} {'HandsMax':>9}")
    ranked = sorted(
        pids,
        key=lambda pid: (
            summary[pid]["win_rate"],
            -(summary[pid]["avg_position"] or 999),
        ),
        reverse=True,
    )
    for rank, pid in enumerate(ranked, 1):
        row = summary[pid]
        avg_pos = row["avg_position"]
        print(
            f"{rank:<5} {pid:<16} {row['wins']:>7} "
            f"{row['win_rate']:>9.3f} {_fmt_ci(row['win_ci']):>18} "
            f"{(avg_pos if avg_pos is not None else 0):>8.2f} "
            f"{row['itm_rate']:>8.3f} {row['hands_avg']:>9.1f} "
            f"{row['hands_median']:>9.1f} {row['hands_max']:>9}"
        )

    print()
    print("HEAD-TO-HEAD MATRIX (row finished above column)")
    col_w = 10
    print(f"{'':>{col_w}}", end="")
    for pid in pids:
        print(f"{pid[:col_w-1]:>{col_w}}", end="")
    print()
    for a in pids:
        print(f"{a[:col_w-1]:>{col_w}}", end="")
        for b in pids:
            rate = aggregated["h2h_matrix"][a][b]
            if rate is None:
                cell = "---"
            else:
                cell = f"{rate:.3f}"
            print(f"{cell:>{col_w}}", end="")
        print()

    verdict = (
        pilot_verdict(summary)
        if config.mode == "pilot"
        else production_verdict(summary)
    )
    print()
    print(verdict)
    print("=" * 80)
    return verdict


def write_csv(path: str, aggregated: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pids = aggregated["players"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tournament", "winner", "hands",
            *[f"{pid}_position" for pid in pids],
            *[f"{pid}_hands_survived" for pid in pids],
            *[f"{pid}_final_chips" for pid in pids],
        ])
        for i, res in enumerate(aggregated["results"], 1):
            fo = {pid: (pos, hand) for pid, pos, hand, _ in res["finish_order"]}
            row = [i, res["winner"], res["hand_count"]]
            for pid in pids:
                row.append(fo.get(pid, (0, 0))[0])
            for pid in pids:
                row.append(fo.get(pid, (0, 0))[1])
            for pid in pids:
                row.append(res["final_chips"].get(pid, 0))
            writer.writerow(row)


def run_evaluation(config: EvalConfig, emit: bool = True) -> dict[str, Any]:
    if config.mode == "curriculum":
        return run_curriculum(config, emit=emit)

    player_specs = build_player_specs(config)
    results = _run_all_tournaments(config, player_specs)
    aggregated = aggregate_results(results, player_specs)
    if emit:
        verdict = print_report(config, aggregated)
        if config.output_csv:
            write_csv(config.output_csv, aggregated)
            print(f"CSV saved: {config.output_csv}")
    else:
        verdict = (
            head_to_head_verdict(
                aggregated["summary"][PATH_A]["win_ci"],
                aggregated["summary"][PATH_B]["win_ci"],
            )
            if config.mode == "head_to_head"
            else (
                promotion_verdict(aggregated["summary"])
                if config.mode == "promotion"
                else (
                    pilot_verdict(aggregated["summary"])
                    if config.mode == "pilot"
                    else production_verdict(aggregated["summary"])
                )
            )
        )
    aggregated["verdict"] = verdict
    aggregated["ante"] = {
        "mode": config.ante_mode,
        "fraction_of_bb": config.ante_fraction_of_bb,
        "label": _ante_label(config),
    }
    return aggregated


def _curriculum_tiers(config: EvalConfig):
    """Progressively harder Deep CFR checkpoint gates."""
    deep_spec = (PATH_B, f"deep_cfr:{config.path_b_weights}")
    return [
        {
            "name": "Tier 1 — random-only 5-player",
            "target": 0.40,
            "specs": [deep_spec] + [(f"RANDOM_{i}", "random") for i in range(1, 5)],
        },
        {
            "name": "Tier 2 — smart-only 5-player",
            "target": 0.25,
            "specs": [deep_spec] + [(f"SMART_{i}", "smart") for i in range(1, 5)],
        },
        {
            "name": "Tier 3 — full strong pool",
            # 7-player field (deep_spec + 6 opponents) → uniform break-even 1/7.
            "target": 1 / 7,
            "specs": [
                deep_spec,
                ("SMART", "smart"),
                ("GTO", "gto"),
                ("MC200", "mc200"),
                ("ICM", "icm"),
                ("EXPLOITATIVE", "exploitative"),
                ("OPPONENTMODEL", "opponentmodel"),
            ],
        },
    ]


def run_curriculum(config: EvalConfig, emit: bool = True) -> dict[str, Any]:
    """Run Deep CFR through progressively harder pre-production gates."""
    tier_results = []
    if emit:
        print("=" * 80)
        print("DEEP CFR CURRICULUM EVAL")
        print("=" * 80)
        print(f"Tournaments per tier: {config.tournaments}")
        print(f"Weights: {config.path_b_weights}")
        print(f"Chips: {config.chips} | Blinds: {config.sb}/{config.bb}")
        print(f"Ante: {_ante_label(config)}")
        print("Selection metric: tournament first-place rate (win_rate)")
        print("=" * 80)

    for tier in _curriculum_tiers(config):
        results = _run_all_tournaments(config, tier["specs"])
        aggregated = aggregate_results(results, tier["specs"])
        deep = aggregated["summary"][PATH_B]
        passed = deep["win_rate"] > tier["target"]
        tier_row = {
            "name": tier["name"],
            "target": tier["target"],
            "passed": passed,
            "summary": deep,
            "aggregated": aggregated,
        }
        tier_results.append(tier_row)

        if emit:
            print()
            print(tier["name"])
            print("-" * len(tier["name"]))
            avg_pos = deep["avg_position"]
            avg_pos_str = f"{avg_pos:.2f}" if avg_pos is not None else "n/a"
            print(
                f"Deep CFR wins={deep['wins']}/{config.tournaments} "
                f"win_rate={deep['win_rate']:.3f} "
                f"Wilson95={_fmt_ci(deep['win_ci'])} "
                f"avg_pos={avg_pos_str} "
                f"hands_avg={deep['hands_avg']:.1f}"
            )
            print(
                f"Target: > {tier['target']:.3f} "
                f"({'PASS' if passed else 'FAIL'})"
            )

        if not passed:
            if emit:
                print("Stopping curriculum early; this checkpoint failed the tier.")
                print("=" * 80)
            break

    verdict = "PASS" if tier_results and all(t["passed"] for t in tier_results) else "FAIL"
    return {
        "mode": "curriculum",
        "tiers": tier_results,
        "verdict": verdict,
        "ante": {
            "mode": config.ante_mode,
            "fraction_of_bb": config.ante_fraction_of_bb,
            "label": _ante_label(config),
        },
    }


def parse_args(argv: list[str] | None = None) -> EvalConfig:
    parser = argparse.ArgumentParser(
        description="Evaluate Path A vs Path B with tournament-aware metrics."
    )
    parser.add_argument(
        "--mode",
        choices=("head_to_head", "multiway", "pilot", "promotion", "curriculum"),
                        required=True)
    parser.add_argument("--path_a_profile", default="models/cfr_regret_deep_v2.pkl",
                        help="Path A CFR profile path")
    parser.add_argument("--path_b_weights", default="models/deep_cfr_v2.pt",
                        help="Path B Deep CFR weights path")
    parser.add_argument("--tournaments", type=int, default=1000)
    parser.add_argument("--pool", default=DEFAULT_POOL,
                        help="Comma-separated bot pool for multiway mode")
    parser.add_argument(
        "--promotion-opponent",
        default="gto",
        help="Strongest existing bot spec for promotion mode",
    )
    parser.add_argument("--chips", type=int, default=500)
    parser.add_argument("--sb", type=int, default=5)
    parser.add_argument("--bb", type=int, default=10)
    parser.add_argument("--blind-increase-every", type=int, default=50)
    parser.add_argument("--max-hands", type=int, default=10000)
    parser.add_argument(
        "--ante-mode",
        choices=("off", "fixed", "schedule"),
        default="off",
        help="Tournament ante mode; default off keeps eval in the training family",
    )
    parser.add_argument(
        "--ante-fraction-of-bb",
        type=float,
        default=0.0,
        help="Ante as a fraction of the big blind for fixed/schedule modes",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--parallel", type=int, default=1)
    ns = parser.parse_args(argv)
    seed = ns.seed
    if ns.mode == "pilot" and seed is None:
        seed = 20260613
    if ns.ante_fraction_of_bb < 0:
        parser.error("--ante-fraction-of-bb must be non-negative")
    return EvalConfig(
        mode=ns.mode,
        path_a_profile=ns.path_a_profile,
        path_b_weights=ns.path_b_weights,
        tournaments=ns.tournaments,
        pool=ns.pool,
        chips=ns.chips,
        sb=ns.sb,
        bb=ns.bb,
        blind_increase_every=ns.blind_increase_every,
        max_hands=ns.max_hands,
        seed=seed,
        output_csv=ns.output_csv,
        parallel=max(1, ns.parallel),
        promotion_opponent=ns.promotion_opponent,
        ante_mode=ns.ante_mode,
        ante_fraction_of_bb=ns.ante_fraction_of_bb,
    )


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    run_evaluation(config, emit=True)


if __name__ == "__main__":
    main()
