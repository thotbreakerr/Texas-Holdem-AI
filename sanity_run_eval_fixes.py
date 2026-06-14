"""
Regression for run_eval fixes:
  - A tournament that hits max_hands with survivors resolves a deterministic
    winner by chip count (instead of leaving winner=None, which biased win_rate
    downward and produced an uncredited no-decision).
  - Tier 3's break-even target matches its 7-player field (1/7).
  - Pilot mode is exactly six-player, rotates every starting seat, and only
    fails when Path B's Wilson interval is wholly below 1/6.
"""
import sys
import tempfile
import os

sys.path.insert(0, "/Users/jaroslavaupart/Desktop/Projects/Texas-Holdem-AI")

import torch

from core.bot_api import Action
import run_eval


class CheckBot:
    """Always checks/calls — never folds, never bets, so nobody busts."""
    def act(self, view):
        legal = {a["type"] for a in view.legal_actions}
        for t in ("check", "call", "fold"):
            if t in legal:
                return Action(t)
        return Action(next(iter(legal)))


def run():
    PASS = True

    # ── max-hands winner resolution ──────────────────────────────────────────
    specs = [("A", "checkbot"), ("B", "checkbot")]
    orig = run_eval._make_bots
    run_eval._make_bots = lambda player_specs: {pid: CheckBot()
                                                for pid, _ in player_specs}
    try:
        # max_hands=1, no blind escalation, check-bots → both survive the cap.
        task = run_eval.TournamentTask(
            player_specs=specs,
            chips=500,
            base_sb=5,
            base_bb=10,
            blind_increase_every=10_000,
            max_hands=1,
            seed=123,
            ante_mode="off",
            ante_fraction_of_bb=0.0,
        )
        result = run_eval._run_one_tournament(task)
    finally:
        run_eval._make_bots = orig

    if result["winner"] is not None and result["winner"] in ("A", "B"):
        lead = max(result["final_chips"], key=lambda p: result["final_chips"][p])
        ok = result["winner"] == lead
        print(f"[CHECK 1] {'PASS' if ok else 'FAIL'} — max-hands winner resolved "
              f"to chip leader ({result['winner']}, chips={result['final_chips']})")
        PASS &= ok
    else:
        PASS = False
        print(f"[CHECK 1] FAIL — winner not resolved: {result['winner']}")

    # ── Tier 3 target matches field size ─────────────────────────────────────
    cfg = run_eval.EvalConfig(mode="curriculum",
                              path_a_profile="x.pkl", path_b_weights="y.pt")
    tiers = run_eval._curriculum_tiers(cfg)
    tier3 = next(t for t in tiers if "Tier 3" in t["name"])
    n_players = len(tier3["specs"])
    if abs(tier3["target"] - 1 / n_players) < 1e-9:
        print(f"[CHECK 2] PASS — Tier 3 target {tier3['target']:.4f} == 1/{n_players}")
    else:
        PASS = False
        print(f"[CHECK 2] FAIL — Tier 3 target {tier3['target']:.4f} != 1/{n_players}")

    pilot_cfg = run_eval.EvalConfig(
        mode="pilot", path_a_profile="x.pkl", path_b_weights="y.pt",
        tournaments=6, seed=100)
    pilot_specs = run_eval.build_player_specs(pilot_cfg)
    six_players = len(pilot_specs) == 6 and pilot_specs[0][0] == run_eval.PATH_B
    print(f"[CHECK 3] {'PASS' if six_players else 'FAIL'} — pilot field has "
          f"{len(pilot_specs)} players")
    PASS &= six_players

    seen_first = []
    original_runner = run_eval._run_one_tournament

    def capture(task):
        seen_first.append(task.player_specs[0][0])
        return {
            "winner": task.player_specs[0][0],
            "hand_count": 1,
            "finish_order": [
                (pid, i + 1, 1, 0)
                for i, (pid, _) in enumerate(task.player_specs)
            ],
            "final_chips": {pid: 0 for pid, _ in task.player_specs},
            "chip_swing": None,
        }

    run_eval._run_one_tournament = capture
    try:
        run_eval._run_all_tournaments(pilot_cfg, pilot_specs)
    finally:
        run_eval._run_one_tournament = original_runner
    rotated = seen_first == [pid for pid, _ in pilot_specs]
    print(f"[CHECK 4] {'PASS' if rotated else 'FAIL'} — starting seats rotate "
          f"across six tournaments: {seen_first}")
    PASS &= rotated

    passing = run_eval.pilot_verdict({
        run_eval.PATH_B: {"win_ci": (0.10, 0.20)}}).startswith("PASS")
    failing = run_eval.pilot_verdict({
        run_eval.PATH_B: {"win_ci": (0.05, 0.16)}}).startswith("FAIL")
    verdict_ok = passing and failing
    print(f"[CHECK 5] {'PASS' if verdict_ok else 'FAIL'} — pilot verdict uses "
          "the 1/6 confidence-interval baseline")
    PASS &= verdict_ok

    promotion_pass = run_eval.promotion_verdict({
        run_eval.PATH_B: {"win_ci": (0.40, 0.55)}}).startswith("PASS")
    promotion_fail = run_eval.promotion_verdict({
        run_eval.PATH_B: {"win_ci": (0.30, 0.49)}}).startswith("FAIL")
    with tempfile.TemporaryDirectory() as tmp:
        clean = os.path.join(tmp, "clean.pt")
        dirty = os.path.join(tmp, "dirty.pt")
        torch.save({"schema_version": 2, "canary_status": "PASS"}, clean)
        torch.save({"schema_version": 2, "canary_status": "WARN"}, dirty)
        run_eval._require_canary_clean_checkpoint(clean)
        try:
            run_eval._require_canary_clean_checkpoint(dirty)
        except RuntimeError:
            rejected_dirty = True
        else:
            rejected_dirty = False
    promotion_ok = promotion_pass and promotion_fail and rejected_dirty
    print(f"[CHECK 6] {'PASS' if promotion_ok else 'FAIL'} — promotion requires "
          "a canary-clean checkpoint and a non-regressing head-to-head CI")
    PASS &= promotion_ok

    print("=" * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED [PASS]' if PASS else 'SOME CHECKS FAILED [FAIL]'}")
    return PASS


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
