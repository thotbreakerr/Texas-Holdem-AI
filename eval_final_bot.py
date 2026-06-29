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
P5_ARM_TOKEN_ALIASES = {
    "p4": "P4",
    "telemetry": "P4+telemetry",
    "station": "P4+station-only",
    "station_only": "P4+station-only",
    "r2": "P4+strict-R2-only",
    "strict_r2": "P4+strict-R2-only",
    "p5": "P4+station+R2",
}

SEAT_GEOMETRY_LAYOUTS: dict[str, tuple[str, ...] | None] = {
    "none": None,
    "target-before-after": ("T", "H", "T", "F", "F", "F"),
    "target-after-hero": ("F", "T", "H", "T", "F", "F"),
    "bracketing": ("T", "F", "F", "T", "H", "F"),
    "pressure-nit-hero": ("F", "F", "F", "PF", "N", "H"),
}
SEAT_GEOMETRY_ALIASES = {
    "": "none",
    "off": "none",
    "default": "none",
    "target-before-hero": "target-before-after",
    "target-adjacent": "target-before-after",
    "target-sandwich": "target-before-after",
    "target-after": "target-after-hero",
    "bracket": "bracketing",
    "pressure-filler-nit-hero": "pressure-nit-hero",
    "filler-nit-hero": "pressure-nit-hero",
}
TARGET_BOT_TYPES = {
    "calling_station",
    "loose_passive",
    "minraise",
    "minraiser",
    "overbet_merchant",
}
TARGET_BOT_PREFIXES = ("maniac",)
NIT_BOT_TYPES = {"nit", "folder"}
STRESS_FUNNEL_READS = (
    "jam_like",
    "short_jam_like",
    "large_bet",
    "station",
    "fold_to_pressure",
    "vpip",
    "pfr",
)


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _canonical_seat_geometry(raw: str | None) -> str:
    token = str(raw or "none").strip().lower().replace("_", "-")
    token = SEAT_GEOMETRY_ALIASES.get(token, token)
    if token not in SEAT_GEOMETRY_LAYOUTS:
        names = ", ".join(sorted(SEAT_GEOMETRY_LAYOUTS))
        aliases = ", ".join(sorted(k for k in SEAT_GEOMETRY_ALIASES if k))
        raise ValueError(f"unknown seat geometry {raw!r}; expected one of: {names}; aliases: {aliases}")
    return token


def _bot_base_type(bot_type: str) -> str:
    return str(bot_type or "").strip().lower().split(":", 1)[0]


def _is_target_bot(bot_type: str) -> bool:
    base = _bot_base_type(bot_type)
    return base in TARGET_BOT_TYPES or any(base.startswith(prefix) for prefix in TARGET_BOT_PREFIXES)


def _is_nit_bot(bot_type: str) -> bool:
    return _bot_base_type(bot_type) in NIT_BOT_TYPES


def _is_pressure_filler_bot(bot_type: str) -> bool:
    return _bot_base_type(bot_type) == "pressure_filler"


def _build_eval_specs(candidate: str, pool: list[str], seat_geometry: str | None = None) -> list[tuple[str, str]]:
    """Return ring-order ``(pid, bot_type)`` specs for an eval table."""
    geometry = _canonical_seat_geometry(seat_geometry)
    default_specs = [("HERO", candidate)] + [(f"P{i}", bot) for i, bot in enumerate(pool)]
    layout = SEAT_GEOMETRY_LAYOUTS[geometry]
    if layout is None:
        return default_specs

    if len(default_specs) != len(layout):
        raise ValueError(
            f"seat geometry {geometry!r} requires {len(layout)} total seats "
            f"(hero + {len(layout) - 1} opponents); got {len(default_specs)}"
        )

    remaining = list(default_specs[1:])

    def take(role: str, predicate) -> tuple[str, str]:
        for idx, entry in enumerate(remaining):
            if predicate(entry[1]):
                return remaining.pop(idx)
        raise ValueError(f"seat geometry {geometry!r} needs a {role} opponent in --stress-pool/--pool")

    targets = [take("target", _is_target_bot) for _ in range(layout.count("T"))]
    pressure_fillers = [
        take("pressure_filler", _is_pressure_filler_bot)
        for _ in range(layout.count("PF"))
    ]
    nits = [take("nit/folder", _is_nit_bot) for _ in range(layout.count("N"))]

    specs: list[tuple[str, str]] = []
    for token in layout:
        if token == "H":
            specs.append(("HERO", candidate))
        elif token == "T":
            specs.append(targets.pop(0))
        elif token == "PF":
            specs.append(pressure_fillers.pop(0))
        elif token == "N":
            specs.append(nits.pop(0))
        elif token == "F":
            if not remaining:
                raise ValueError(f"seat geometry {geometry!r} ran out of filler opponents")
            specs.append(remaining.pop(0))
        else:
            raise ValueError(f"internal error: unknown seat geometry token {token!r}")

    if remaining:
        raise ValueError(f"seat geometry {geometry!r} left unused opponents: {remaining!r}")
    return specs


def _hero_relative_offsets(specs: list[tuple[str, str]], predicate) -> list[int]:
    n = len(specs)
    hero_idx = next(idx for idx, (pid, _) in enumerate(specs) if pid == "HERO")
    return sorted((idx - hero_idx) % n for idx, (_, bot_type) in enumerate(specs) if predicate(bot_type))


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


def _arm_from_token(raw: str) -> str:
    token = str(raw or "").strip()
    if token in P5_ARM_SPECS:
        return token
    key = token.lower().replace("-", "_").replace("+", "_")
    if key in P5_ARM_TOKEN_ALIASES:
        return P5_ARM_TOKEN_ALIASES[key]
    raise ValueError(f"unknown P5 arm token: {raw!r}")


def _candidate_spec(profile: str, arm: str) -> str:
    return f"final_{profile}:{P5_ARM_SPECS[arm]}"


def _empty_p5_eval_telemetry() -> dict:
    return {
        "p5_decisions": 0,
        "p5_station_read_fire_count": 0,
        "p5_r2_read_fire_count": 0,
        "p5_r2_tax_relief_applied_count": 0,
        "p5_r2_tax_relieved_total": 0.0,
        "n_changed_decisions": 0,
        "n_chip_ev_ok": 0,
        "exante_ev_margin_total": 0.0,
        "mean_exante_ev_margin": 0.0,
        "realized_chip_delta_total": 0.0,
        "n_negative_realized": 0,
        "n_realized_decisions": 0,
        "n_pending_outcomes": 0,
        "no_realized_harm": True,
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
        "p5_r2_tax_relief_applied_count",
        "n_changed_decisions",
        "n_chip_ev_ok",
        "n_negative_realized",
        "n_realized_decisions",
        "n_pending_outcomes",
        "p5_error_count",
    ):
        total[key] += int(row.get(key, 0) or 0)
    total["p5_r2_tax_relieved_total"] += float(row.get("p5_r2_tax_relieved_total", 0.0) or 0.0)
    changed = int(row.get("n_changed_decisions", row.get("p5_r2_tax_relief_applied_count", 0)) or 0)
    total["exante_ev_margin_total"] += float(
        row.get(
            "exante_ev_margin_total",
            float(row.get("mean_exante_ev_margin", 0.0) or 0.0) * changed,
        )
        or 0.0
    )
    total["realized_chip_delta_total"] += float(row.get("realized_chip_delta_total", 0.0) or 0.0)
    samples = total["p5_villain_samples"]
    for pid, stats in (row.get("p5_villain_samples") or {}).items():
        bucket = samples.setdefault(pid, {})
        for stat_key, value in (stats or {}).items():
            if str(stat_key).endswith("_hat"):
                bucket[stat_key] = float(value or 0.0)
            else:
                bucket[stat_key] = int(bucket.get(stat_key, 0) or 0) + int(value or 0)


def _finalize_p5_eval_telemetry(total: dict) -> dict:
    decisions = max(1, int(total.get("p5_decisions", 0) or 0))
    station = int(total.get("p5_station_read_fire_count", 0) or 0)
    r2 = int(total.get("p5_r2_read_fire_count", 0) or 0)
    total["p5_station_read_fire_rate"] = station / decisions
    total["p5_r2_read_fire_rate"] = r2 / decisions
    total["p5_any_active_read_fire_rate"] = (station + r2) / decisions
    changed = int(total.get("n_changed_decisions", 0) or 0)
    total["mean_exante_ev_margin"] = (
        float(total.get("exante_ev_margin_total", 0.0) or 0.0) / changed
        if changed else 0.0
    )
    total["no_realized_harm"] = float(total.get("realized_chip_delta_total", 0.0) or 0.0) >= 0.0
    return total


def _empty_stress_eval_telemetry() -> dict:
    return {
        "enabled": False,
        "seat_geometry": "none",
        "by_pid": {},
        "by_archetype": {},
        "totals": {},
        "station_false_positive_count": 0,
        "station_false_positive_sampled_count": 0,
        "station_false_positive_rate": 0.0,
        "short_jam_like_contamination_rate": 0.0,
    }


def _merge_stress_eval_telemetry(total: dict, pid: str, archetype: str, row: dict | None) -> None:
    if not row:
        return
    total["enabled"] = True
    for target in (
        total["by_pid"].setdefault(pid, {"archetype": archetype}),
        total["by_archetype"].setdefault(archetype, {"archetype": archetype}),
    ):
        for key, value in row.items():
            try:
                amount = int(value or 0)
            except (TypeError, ValueError):
                continue
            target[key] = int(target.get(key, 0) or 0) + amount


def _p_hat(k: int, n: int, p0: float = 0.45, w: float = 10.0) -> float:
    n = max(0, int(n or 0))
    k = max(0, int(k or 0))
    denom = n + w
    if denom <= 0:
        return p0
    return (k + p0 * w) / denom


def _finalize_stress_eval_telemetry(total: dict, p5: dict) -> dict:
    if not total.get("enabled"):
        return total
    samples = p5.get("p5_villain_samples") or {}
    observed_keys = {
        "jam_like": "jam_like_count",
        "short_jam_like": "short_jam_like_count",
        "large_bet": "large_bet_count",
        "station": "postflop_pressure_call",
        "fold_to_pressure": "pressure_fold_n",
        "vpip": "vpip_seen",
        "pfr": "preflop_raise_seen",
    }
    stress_totals: dict[str, int] = {}
    fp_count = 0
    fp_sampled = 0
    for pid, bucket in total["by_pid"].items():
        sample = samples.get(pid, {})
        for read, sample_key in observed_keys.items():
            observed = int(sample.get(sample_key, 0) or 0)
            true_key = f"true_{read}_count"
            folded_key = f"missed_due_to_hero_already_folded_{read}_count"
            hand_end_key = f"missed_due_to_hand_end_{read}_count"
            bucket[f"hero_observed_{read}_count"] = observed
            bucket[hand_end_key] = max(
                0,
                int(bucket.get(true_key, 0) or 0)
                - observed
                - int(bucket.get(folded_key, 0) or 0),
            )
        archetype = str(bucket.get("archetype") or "")
        if archetype in ("loose_passive", "nit"):
            # Mirror the bot's O1 station_score: call-rate over ALL postflop
            # pressure responses (folds in the denominator), not call-vs-raise,
            # AND the fold-to-pressure guard (a true station does not fold).
            calls = int(sample.get("postflop_pressure_call", 0) or 0)
            raises = int(sample.get("postflop_pressure_raise", 0) or 0)
            folds = int(sample.get("postflop_pressure_fold", 0) or 0)
            station_n = calls + raises + folds
            bfold = int(sample.get("pressure_fold", 0) or 0)
            bcall = int(sample.get("pressure_call", 0) or 0)
            braise = int(sample.get("pressure_raise", 0) or 0)
            f2p = _p_hat(bfold, bfold + bcall + braise)
            not_folder = f2p <= 0.40  # mirrors _P5_STATION_MAX_FOLD_TO_PRESSURE
            if station_n >= 4:
                fp_sampled += 1
                if not_folder and _p_hat(calls, station_n) >= 0.60:
                    fp_count += 1
                    bucket["station_false_positive"] = 1
                else:
                    bucket["station_false_positive"] = 0
        for key, value in bucket.items():
            if key == "archetype":
                continue
            try:
                stress_totals[key] = int(stress_totals.get(key, 0) or 0) + int(value or 0)
            except (TypeError, ValueError):
                continue

    total["totals"] = stress_totals
    total["station_false_positive_count"] = fp_count
    total["station_false_positive_sampled_count"] = fp_sampled
    total["station_false_positive_rate"] = fp_count / fp_sampled if fp_sampled else 0.0
    true_jams = int(stress_totals.get("true_jam_like_count", 0) or 0)
    short_jams = int(stress_totals.get("true_short_jam_like_count", 0) or 0)
    total["short_jam_like_contamination_rate"] = (
        short_jams / (true_jams + short_jams) if (true_jams + short_jams) else 0.0
    )
    return total


def _stress_funnel_fields(totals: dict, read: str) -> dict[str, int]:
    true_count = int(totals.get(f"true_{read}_count", 0) or 0)
    observed = int(totals.get(f"hero_observed_{read}_count", 0) or 0)
    folded = int(totals.get(f"missed_due_to_hero_already_folded_{read}_count", 0) or 0)
    folded_pre = int(totals.get(f"missed_due_to_hero_already_folded_preflop_{read}_count", 0) or 0)
    folded_post = int(totals.get(f"missed_due_to_hero_already_folded_postflop_{read}_count", 0) or 0)
    folded_unknown = max(0, folded - folded_pre - folded_post)
    hand_end = int(totals.get(f"missed_due_to_hand_end_{read}_count", 0) or 0)
    return {
        "true": true_count,
        "observed": observed,
        "folded_pre": folded_pre,
        "folded_post": folded_post,
        "folded_unknown": folded_unknown,
        "hand_end": hand_end,
    }


def evaluate_profile(profile: str, pool: list[str], args, arm: str) -> dict:
    candidate = _candidate_spec(profile, arm)
    seat_geometry = _canonical_seat_geometry(getattr(args, "seat_geometry", "none"))
    specs = _build_eval_specs(candidate, pool, seat_geometry)
    n_players = len(specs)

    wins = 0
    finishes: list[int] = []
    early_busts = 0
    reached_hu = 0
    p5_telemetry = _empty_p5_eval_telemetry()
    stress_telemetry = _empty_stress_eval_telemetry()
    stress_telemetry["seat_geometry"] = seat_geometry

    for t in range(args.tournaments):
        # Seed the module-global RNG that stochastic opponent bots
        # (gto/smart/exploitative) draw from. Without this the global stream is
        # seeded from os.urandom at process startup, so the same eval seed
        # produces different results every run -- and, in particular, differs
        # across PYTHONHASHSEED values purely by chance. A fixed per-tournament
        # seed makes the whole table reproducible across processes. The offset
        # keeps this stream distinct from the per-tournament deck RNG below so
        # bot randomness is not correlated with the shuffle. (HERO itself never
        # relies on this stream -- it gates on a stable hash and save/restores
        # global RNG state around its Monte Carlo equity.)
        random.seed(args.seed + t + 1 + 7_000_003)
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
            _merge_p5_eval_telemetry(p5_telemetry, telemetry(result))
        for pid, bot_type in specs:
            if pid == "HERO":
                continue
            core = getattr(bots.get(pid), "bot", None)
            stress_summary = getattr(core, "stress_telemetry_summary", None)
            if callable(stress_summary):
                archetype = getattr(getattr(core, "config", None), "name", bot_type)
                _merge_stress_eval_telemetry(
                    stress_telemetry,
                    pid,
                    str(archetype),
                    stress_summary(),
                )

    n = args.tournaments
    lo, hi = wilson_interval(wins, n)
    avg_finish = sum(finishes) / len(finishes) if finishes else float("nan")
    p5_final = _finalize_p5_eval_telemetry(p5_telemetry)
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
        "p5_telemetry": p5_final,
        "seat_geometry": seat_geometry,
        "seat_order": [pid for pid, _ in specs],
        "stress_telemetry": _finalize_stress_eval_telemetry(stress_telemetry, p5_final),
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
    ap.add_argument("--arm", default=None,
                    help="Short alias for --p5-arm: p4, telemetry, station, r2, or p5.")
    ap.add_argument("--p5-arms", default=None,
                    help="Comma-separated Phase 5 arms, or 'all'. Overrides --p5-arm.")
    ap.add_argument("--p5-telemetry", action="store_true",
                    help="Print Phase 5 pre-gate telemetry summaries.")
    ap.add_argument("--stress-pool", default=None,
                    help="Opt-in comma-separated Phase 7 stress-opponent pool. "
                         "When set, overrides --pool for this run only.")
    ap.add_argument(
        "--seat-geometry",
        default="none",
        help="Opt-in deterministic Phase 7 ring layout: none, target-before-after, "
             "target-after-hero, bracketing, or pressure-nit-hero.",
    )
    ap.add_argument(
        "--future-edge-tax-cap", type=float, default=None,
        help="A/B override (Phase 4 §12 R7): hard ceiling on the hero's "
             "future-edge tax, in equity fractions. 0 disables the tax; omit "
             "to use the built-in per-player caps.",
    )
    args = ap.parse_args()
    try:
        args.seat_geometry = _canonical_seat_geometry(args.seat_geometry)
    except ValueError as exc:
        ap.error(str(exc))

    pool_raw = args.stress_pool if args.stress_pool is not None else args.pool
    pool = [p.strip() for p in pool_raw.split(",") if p.strip()]
    profiles = ["survival", "aggro"] if args.profile == "both" else [args.profile]
    if args.p5_arms is not None:
        arms = _arm_list(args.p5_arms)
    elif args.arm is not None:
        arms = [_arm_from_token(args.arm)]
    else:
        arms = [args.p5_arm]

    tax_cap = ("built-in" if args.future_edge_tax_cap is None
               else ("OFF" if args.future_edge_tax_cap == 0 else f"{args.future_edge_tax_cap:.3f}"))
    pool_label = "Stress pool" if args.stress_pool is not None else "Pool"
    geometry_label = "" if args.seat_geometry == "none" else f"  |  seat_geometry={args.seat_geometry}"
    print(f"{pool_label}: {', '.join(pool)}  |  field size: {len(pool) + 1}  |  "
          f"chips={args.chips} sb={args.sb} bb={args.bb} "
          f"blind_up_every={args.blind_increase_every}{geometry_label}  |  "
          f"future-edge-tax-cap={tax_cap}")
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
                print(f"  r2 tax relief   : applied={tel['p5_r2_tax_relief_applied_count']} "
                      f"relieved_total={tel['p5_r2_tax_relieved_total']:.4f}")
                harm_label = "no net realized harm" if tel["no_realized_harm"] else "net realized harm"
                print(f"  r2 value/no-harm: changed={tel['n_changed_decisions']} "
                      f"chipEV_ok={tel['n_chip_ev_ok']}/{tel['n_changed_decisions']} "
                      f"mean_margin={tel['mean_exante_ev_margin']:.4f} "
                      f"realized_delta={tel['realized_chip_delta_total']:.1f} "
                      f"negative_realized={tel['n_negative_realized']} "
                      f"realized={tel['n_realized_decisions']} "
                      f"pending={tel['n_pending_outcomes']} "
                      f"({harm_label}; realized outcome only, not counterfactual EV)")
                for pid, stats in sorted(tel["p5_villain_samples"].items()):
                    # Pooled (cross-tournament) rates -- these match the station
                    # FP measurement. The per-villain station_score_hat in the
                    # sample is a per-tournament average-of-ratios and reads low.
                    _c = int(stats.get("postflop_pressure_call", 0) or 0)
                    _r = int(stats.get("postflop_pressure_raise", 0) or 0)
                    _f = int(stats.get("postflop_pressure_fold", 0) or 0)
                    _bf = int(stats.get("pressure_fold", 0) or 0)
                    _bc = int(stats.get("pressure_call", 0) or 0)
                    _br = int(stats.get("pressure_raise", 0) or 0)
                    _callrate = _p_hat(_c, _c + _r + _f)
                    _f2p = _p_hat(_bf, _bf + _bc + _br)
                    print(
                        f"    {pid}: station_n={stats.get('station_response_n', 0)} "
                        f"pressure_fold_n={stats.get('pressure_fold_n', 0)} "
                        f"jam_opp_n={stats.get('jam_opportunity_n', 0)} "
                        f"jam_like={stats.get('jam_like_count', 0)} "
                        f"large_bet={stats.get('large_bet_count', 0)} "
                        f"pooled_callrate={_callrate:.3f} "
                        f"pooled_fold_to_pressure={_f2p:.3f}"
                    )
            stress = s.get("stress_telemetry") or {}
            if stress.get("enabled"):
                totals = stress.get("totals") or {}
                geometry = stress.get("seat_geometry") or s.get("seat_geometry") or "none"
                print(f"  stress events   : true_jam={totals.get('true_jam_like_count', 0)} "
                      f"obs_jam={totals.get('hero_observed_jam_like_count', 0)} "
                      f"true_large={totals.get('true_large_bet_count', 0)} "
                      f"obs_large={totals.get('hero_observed_large_bet_count', 0)} "
                      f"true_station={totals.get('true_station_count', 0)} "
                      f"obs_station={totals.get('hero_observed_station_count', 0)}")
                print(f"  stress censor   : hero_folded={totals.get('missed_due_to_hero_already_folded_count', 0)} "
                      f"short_jam_rate={stress.get('short_jam_like_contamination_rate', 0.0):.3f} "
                      f"station_fp={stress.get('station_false_positive_count', 0)}/"
                      f"{stress.get('station_false_positive_sampled_count', 0)}")
                print(f"  stress funnel   : geometry={geometry}")
                for read in STRESS_FUNNEL_READS:
                    row = _stress_funnel_fields(totals, read)
                    if not any(row.values()):
                        continue
                    unknown = (
                        f" folded_unknown={row['folded_unknown']}"
                        if row["folded_unknown"]
                        else ""
                    )
                    print(
                        f"    {read}: true={row['true']} obs={row['observed']} "
                        f"folded_pre={row['folded_pre']} folded_post={row['folded_post']}"
                        f"{unknown} hand_end={row['hand_end']}"
                    )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
