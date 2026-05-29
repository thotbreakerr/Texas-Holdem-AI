"""
Regression for run_eval fixes:
  - A tournament that hits max_hands with survivors resolves a deterministic
    winner by chip count (instead of leaving winner=None, which biased win_rate
    downward and produced an uncredited no-decision).
  - Tier 3's break-even target matches its 7-player field (1/7).
"""
import sys

sys.path.insert(0, "/Users/jaroslavaupart/Desktop/Projects/Texas-Holdem-AI")

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
        task = (specs, 500, 5, 10, 10_000, 1, 123)
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

    print("=" * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED [PASS]' if PASS else 'SOME CHECKS FAILED [FAIL]'}")
    return PASS


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
