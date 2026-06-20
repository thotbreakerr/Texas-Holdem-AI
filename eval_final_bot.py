#!/usr/bin/env python3
"""Standalone tournament eval for the final tournament hybrid bot.

Seats ``final_survival`` (or ``final_aggro``) against the baseline pool
(smart, mc200, gto, icm, exploitative) under the locked class-tournament
assumptions, WITHOUT loading any CFR / Deep CFR model files.  This is the
verification harness for Phase 3 (postflop equity & pot odds): run it before
and after a change to confirm the candidate's tournament win rate moves the
right way.

Examples
--------
    .venv/bin/python eval_final_bot.py --profile survival --tournaments 200
    .venv/bin/python eval_final_bot.py --profile both --tournaments 500 \
        --pool smart,mc200,gto,icm,exploitative
    .venv/bin/python eval_final_bot.py --profile survival --p5-arms all \
        --p5-telemetry --tournaments 100
"""
from __future__ import annotations

import argparse
import io
import math
import os
import random
import sys
from contextlib import redirect_stdout

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from bots import create_bot
from core.engine import Seat
from core.tournament import run_tournament


P5_ARM_SPECS = {
    "P4": "p4",
    "P4+telemetry": "telemetry",
    "P4+station-only": "station",
    "P4+strict-R2-only": "r2",
    "P4+station+R2": "p5",
}
P5_ARM_CHOICES = tuple(P5_ARM_SPECS)


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _hero_finish(result: dict, hero: str, n_players: int) -> tuple[int | None, int | None]:
    """Return (finish_position, hand_busted) for hero (pos 1 == winner)."""
    for entry in result.get("finish_order") or []:
        if entry and entry[0] == hero:
            pos = entry[1] if len(entry) > 1 else None
            busted = entry[2] if len(entry) > 2 else None
            return pos, busted
    # Not in finish_order means hero survived to the end (winner).
    if result.get("winner") == hero:
        return 1, None
    return None, None


def _arm_list(raw: str | None) -> list[str]:
    if raw is None:
        return ["P4"]
    if raw.strip().lower() == "all":
        return list(P5_ARM_CHOICES)
    arms = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [arm for arm in arms if arm not in P5_ARM_SPECS]
    if unknown:
        raise ValueError(f"unknown P5 arm(s): {', '.join(unknown)}")
    return arms or ["P4"]


def _candidate_spec(profile: str, arm: str) -> str:
    return f"final_{profile}:{P5_ARM_SPECS[arm]}"


def _empty_p5_eval_telemetry() -> dict:
    return {
        "p5_decisions": 0,
        "p5_station_read_fire_count": 0,
        "p5_r2_read_fire_count": 0,
        "p5_error_count": 0,
        "p5_villain_samples": {},
    }


def _merge_p5_eval_telemetry(total: dict, row: dict | None) -> None:
    if not row:
        return
    for key in (
        "p5_decisions",
        "p5_station_read_fire_count",
        "p5_r2_read_fire_count",
        "p5_error_count",
    ):
        total[key] += int(row.get(key, 0) or 0)
    samples = total["p5_villain_samples"]
    for pid, stats in (row.get("p5_villain_samples") or {}).items():
        bucket = samples.setdefault(pid, {})
        for stat_key, value in (stats or {}).items():
            bucket[stat_key] = int(bucket.get(stat_key, 0) or 0) + int(value or 0)


def _finalize_p5_eval_telemetry(total: dict) -> dict:
    decisions = max(1, int(total.get("p5_decisions", 0) or 0))
    station = int(total.get("p5_station_read_fire_count", 0) or 0)
    r2 = int(total.get("p5_r2_read_fire_count", 0) or 0)
    total["p5_station_read_fire_rate"] = station / decisions
    total["p5_r2_read_fire_rate"] = r2 / decisions
    total["p5_any_active_read_fire_rate"] = (station + r2) / decisions
    return total


def evaluate_profile(profile: str, pool: list[str], args, arm: str) -> dict:
    candidate = _candidate_spec(profile, arm)
    specs: list[tuple[str, str]] = [("HERO", candidate)]
    specs += [(f"P{i}", bot) for i, bot in enumerate(pool)]
    n_players = len(specs)

    wins = 0
    finishes: list[int] = []
    early_busts = 0
    reached_hu = 0
    p5_telemetry = _empty_p5_eval_telemetry()

    for t in range(args.tournaments):
        seats = [Seat(player_id=pid, chips=args.chips) for pid, _ in specs]
        bots = {pid: create_bot(bot) for pid, bot in specs}
        hero_core = getattr(bots["HERO"], "bot", None)
        if args.future_edge_tax_cap is not None:
            if hero_core is not None and hasattr(hero_core, "future_edge_tax_cap_override"):
                hero_core.future_edge_tax_cap_override = args.future_edge_tax_cap
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = run_tournament(
                seats,
                bots,
                small_blind=args.sb,
                big_blind=args.bb,
                blind_increase_every=args.blind_increase_every,
                max_hands=args.max_hands,
                dealer_index=t % n_players,
                dealer_rotation="full_table",
                winner_resolution="chip_count_on_max_hands",
                rng=random.Random(args.seed + t + 1),
                suppress_output=True,
                log_decisions=False,
            )
        if result.get("winner") == "HERO":
            wins += 1
        pos, busted = _hero_finish(result, "HERO", n_players)
        if pos is not None:
            finishes.append(pos)
            if pos <= 2:
                reached_hu += 1
        if busted is not None and busted <= args.early_bust_hands:
            early_busts += 1
        telemetry = getattr(hero_core, "p5_telemetry_summary", None)
        if callable(telemetry):
            _merge_p5_eval_telemetry(p5_telemetry, telemetry())

    n = args.tournaments
    lo, hi = wilson_interval(wins, n)
    avg_finish = sum(finishes) / len(finishes) if finishes else float("nan")
    return {
        "candidate": candidate,
        "p5_arm": arm,
        "tournaments": n,
        "players": n_players,
        "wins": wins,
        "win_rate": wins / n if n else 0.0,
        "win_ci": (lo, hi),
        "expected_win_rate": 1.0 / n_players,
        "avg_finish": avg_finish,
        "reached_hu_rate": reached_hu / n if n else 0.0,
        "early_bust_rate": early_busts / n if n else 0.0,
        "p5_telemetry": _finalize_p5_eval_telemetry(p5_telemetry),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="survival",
                    choices=["survival", "aggro", "both"])
    ap.add_argument("--tournaments", type=int, default=200)
    ap.add_argument("--chips", type=int, default=1000)
    ap.add_argument("--sb", type=int, default=5)
    ap.add_argument("--bb", type=int, default=10)
    ap.add_argument("--blind-increase-every", type=int, default=50)
    ap.add_argument("--pool", default="smart,mc200,gto,icm,exploitative")
    ap.add_argument("--max-hands", type=int, default=1000)
    ap.add_argument("--early-bust-hands", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260619)
    ap.add_argument("--p5-arm", choices=P5_ARM_CHOICES, default="P4",
                    help="Single Phase 5 ablation arm to evaluate.")
    ap.add_argument("--p5-arms", default=None,
                    help="Comma-separated Phase 5 arms, or 'all'. Overrides --p5-arm.")
    ap.add_argument("--p5-telemetry", action="store_true",
                    help="Print Phase 5 pre-gate telemetry summaries.")
    ap.add_argument(
        "--future-edge-tax-cap", type=float, default=None,
        help="A/B override (Phase 4 §12 R7): hard ceiling on the hero's "
             "future-edge tax, in equity fractions. 0 disables the tax; omit "
             "to use the built-in per-player caps.",
    )
    args = ap.parse_args()

    pool = [p.strip() for p in args.pool.split(",") if p.strip()]
    profiles = ["survival", "aggro"] if args.profile == "both" else [args.profile]
    arms = _arm_list(args.p5_arms) if args.p5_arms is not None else [args.p5_arm]

    tax_cap = ("built-in" if args.future_edge_tax_cap is None
               else ("OFF" if args.future_edge_tax_cap == 0 else f"{args.future_edge_tax_cap:.3f}"))
    print(f"Pool: {', '.join(pool)}  |  field size: {len(pool) + 1}  |  "
          f"chips={args.chips} sb={args.sb} bb={args.bb} "
          f"blind_up_every={args.blind_increase_every}  |  future-edge-tax-cap={tax_cap}")
    print("=" * 72)
    print(f"P5 arms: {', '.join(arms)}")
    for profile in profiles:
        for arm in arms:
            s = evaluate_profile(profile, pool, args, arm)
            lo, hi = s["win_ci"]
            print(f"\n{s['candidate']}  [{s['p5_arm']}]  ({s['tournaments']} tournaments, "
                  f"{s['players']}-player)")
            print(f"  win rate        : {s['win_rate']:.3f}  "
                  f"(95% CI {lo:.3f}–{hi:.3f}; chance baseline "
                  f"{s['expected_win_rate']:.3f})")
            print(f"  avg finish pos  : {s['avg_finish']:.2f}  (1 = win)")
            print(f"  reached heads-up: {s['reached_hu_rate']:.3f}")
            print(f"  early-bust rate : {s['early_bust_rate']:.3f}  "
                  f"(busted within first {args.early_bust_hands} hands)")
            if args.p5_telemetry:
                tel = s["p5_telemetry"]
                print(f"  p5 errors       : {tel['p5_error_count']}")
                print(f"  p5 read fires   : any={tel['p5_any_active_read_fire_rate']:.4f} "
                      f"station={tel['p5_station_read_fire_rate']:.4f} "
                      f"r2={tel['p5_r2_read_fire_rate']:.4f} "
                      f"(decisions={tel['p5_decisions']})")
                for pid, stats in sorted(tel["p5_villain_samples"].items()):
                    print(
                        f"    {pid}: station_n={stats.get('station_response_n', 0)} "
                        f"pressure_fold_n={stats.get('pressure_fold_n', 0)} "
                        f"jam_opp_n={stats.get('jam_opportunity_n', 0)} "
                        f"jam_like={stats.get('jam_like_count', 0)} "
                        f"large_bet={stats.get('large_bet_count', 0)}"
                    )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
