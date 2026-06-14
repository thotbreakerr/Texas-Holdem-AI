"""
sanity_eval.py — pre-eval gate for Path A vs Path B harness.

This sanity runs small tournament samples with empty/random Path A/B state to
verify the eval harness computes real metrics, Wilson intervals, coherent
head-to-head matrices, and verdict logic before overnight-trained artifacts
exist.
"""
from __future__ import annotations

import io
import math
import os
import pickle
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, ".")

from core.bot_api import PlayerView
from run_eval import (
    DEFAULT_POOL,
    EvalConfig,
    PATH_A,
    PATH_B,
    head_to_head_verdict,
    run_evaluation,
    wilson_ci,
)
from bots import create_bot


PASS = True


def _check(condition: bool, ok: str, fail: str) -> None:
    global PASS
    if condition:
        print(f"  PASS - {ok}")
    else:
        print(f"  FAIL - {fail}")
        PASS = False


def _missing_path(name: str) -> str:
    return os.path.join("/tmp", f"texas_holdem_eval_missing_{name}")


def _make_temp_cfr_profile() -> str:
    """Write a minimal-but-VALID CFR profile and return its path.

    The Path A harness builds its CFR bot via ``cfr:<path>`` in inference mode,
    which (by design) fails loudly on a missing/empty profile.  The end-to-end
    smoke sections need a real, loadable profile — not a missing path — so we
    create one with a couple of well-formed Gate-2A keys (7 fields / 6 colons;
    NUM_ACTIONS == 8).  Unmatched info-sets fall back to search/heuristic at
    inference time, which is all the harness needs to compute real metrics.
    """
    nodes = {
        # street:n_opp:position:spr:opp_stat:bucket:history  (6 colons)
        "preflop:1:late:high:NA:10:": {
            "regret_sum": [0.0] * 8,
            "strategy_sum": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "preflop:2:middle:mid:NA:25:R": {
            "regret_sum": [0.0] * 8,
            "strategy_sum": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    }
    profile = {"nodes": nodes, "hands_played": 100, "total_iterations": 500}
    fd, path = tempfile.mkstemp(suffix="_cfr_path_a.pkl")
    with os.fdopen(fd, "wb") as f:
        pickle.dump(profile, f)
    return path


def _make_temp_deep_cfr_weights() -> str:
    """Write a randomly-initialised Deep CFR checkpoint and return its path.

    Like Path A, the harness builds its Path B bot via ``deep_cfr:<path>`` in
    inference mode, which fails loudly on missing weights.  The smoke sections
    only need a loadable checkpoint exercising the real inference path — the
    section's stated goal is "empty/random Path A/B state" — so we mint a fresh
    untrained network rather than depending on an overnight-trained artifact.
    We use the ``.large()`` config because that is the architecture
    ``DeepCFRBot`` reconstructs when inferring the config from a state dict.
    """
    import torch
    from bots.deep_cfr_bot import (
        DEEP_CFR_SCHEMA_VERSION,
        DeepCFRNetwork,
        DeepCFRConfig,
    )

    net = DeepCFRNetwork(DeepCFRConfig.large())
    fd, path = tempfile.mkstemp(suffix="_deep_cfr_path_b.pt")
    os.close(fd)
    with redirect_stdout(io.StringIO()):
        torch.save({
            "schema_version": DEEP_CFR_SCHEMA_VERSION,
            "config": DeepCFRConfig.large(),
            "network_state_dict": net.state_dict(),
            "iteration": 0,
        }, path)
    return path


TEMP_CFR_PROFILE = _make_temp_cfr_profile()
TEMP_DEEP_CFR_WEIGHTS = _make_temp_deep_cfr_weights()


def _ci_valid(ci) -> bool:
    return 0.0 <= ci[0] <= ci[1] <= 1.0


print("=" * 70)
print("sanity_eval.py - Path A vs Path B eval harness")
print("=" * 70)
print()


# ---------------------------------------------------------------------------
# Section 1 - Harness runs end-to-end with empty Path A/B
# ---------------------------------------------------------------------------

print("=" * 60)
print("Section 1: Head-to-head harness smoke")
print("=" * 60)

try:
    config = EvalConfig(
        mode="head_to_head",
        path_a_profile=TEMP_CFR_PROFILE,
        path_b_weights=TEMP_DEEP_CFR_WEIGHTS,
        tournaments=5,
        chips=200,
        seed=101,
    )
    with redirect_stdout(io.StringIO()) as out:
        h2h = run_evaluation(config, emit=True)
    text = out.getvalue()

    _check(len(h2h["results"]) == 5,
           "all 5 head-to-head tournaments completed",
           "head-to-head tournament count was not 5")
    all_cis_valid = all(_ci_valid(row["win_ci"])
                        for row in h2h["summary"].values())
    _check(all_cis_valid,
           "all Wilson CIs are valid probabilities",
           "at least one Wilson CI was invalid")
    _check("Wilson95=" in text,
           "report includes Wilson95 intervals",
           "report omitted Wilson95 intervals")
    _check("Verdict:" in text,
           "report includes a verdict line",
           "report omitted verdict line")

except Exception as e:
    print(f"  FAIL - exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ---------------------------------------------------------------------------
# Section 2 - Multiway mode runs end-to-end
# ---------------------------------------------------------------------------

print("=" * 60)
print("Section 2: Multiway harness smoke")
print("=" * 60)

try:
    config = EvalConfig(
        mode="multiway",
        path_a_profile=TEMP_CFR_PROFILE,
        path_b_weights=TEMP_DEEP_CFR_WEIGHTS,
        tournaments=5,
        pool=DEFAULT_POOL,
        chips=200,
        seed=202,
    )
    with redirect_stdout(io.StringIO()) as out:
        multi = run_evaluation(config, emit=True)
    text = out.getvalue()

    _check(len(multi["results"]) == 5,
           "all 5 multiway tournaments completed",
           "multiway tournament count was not 5")
    wr_sum = sum(row["win_rate"] for row in multi["summary"].values())
    print(f"  Win-rate sum: {wr_sum:.6f}")
    _check(abs(wr_sum - 1.0) < 1e-9,
           "win rates sum to 1.0",
           "win rates do not sum to 1.0")

    symmetric = True
    matrix = multi["h2h_matrix"]
    for a in multi["players"]:
        for b in multi["players"]:
            if a == b:
                continue
            ab = matrix[a][b]
            ba = matrix[b][a]
            if ab is None or ba is None:
                continue
            if abs((ab + ba) - 1.0) > 0.001:
                symmetric = False
    _check(symmetric,
           "head-to-head matrix is symmetric/coherent",
           "head-to-head matrix symmetry check failed")

    all_cis_valid = all(_ci_valid(row["win_ci"])
                        for row in multi["summary"].values())
    _check(all_cis_valid and "Wilson95" in text,
           "each bot has a Wilson CI reported",
           "missing or invalid Wilson CI in multiway report")

except Exception as e:
    print(f"  FAIL - exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ---------------------------------------------------------------------------
# Section 3 - Wilson CI math correctness
# ---------------------------------------------------------------------------

print("=" * 60)
print("Section 3: Wilson CI math")
print("=" * 60)

try:
    low, high = wilson_ci(100, 200)
    print(f"  Wilson(100/200): [{low:.6f}, {high:.6f}]")
    _check(abs(low - 0.43136) < 0.001 and abs(high - 0.56864) < 0.001,
           "Wilson interval matches known n=200 p=0.5 calculation",
           "Wilson interval differs from known calculation")
except Exception as e:
    print(f"  FAIL - exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ---------------------------------------------------------------------------
# Section 4 - Decision verdict logic
# ---------------------------------------------------------------------------

print("=" * 60)
print("Section 4: Verdict logic")
print("=" * 60)

try:
    scenarios = [
        (
            "Path A clear",
            (0.61, 0.69),
            (0.31, 0.39),
            "Production CFR: PATH_A",
        ),
        (
            "Path B clear",
            (0.34, 0.42),
            (0.58, 0.66),
            "Production CFR: PATH_B",
        ),
        (
            "Tie/noise",
            (0.46, 0.56),
            (0.44, 0.54),
            "No decisive winner",
        ),
    ]
    for name, a_ci, b_ci, expected in scenarios:
        verdict = head_to_head_verdict(a_ci, b_ci)
        print(f"  {name}: {verdict}")
        _check(verdict.startswith(expected),
               f"{name} produced expected verdict",
               f"{name} expected {expected!r}, got {verdict!r}")
except Exception as e:
    print(f"  FAIL - exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ---------------------------------------------------------------------------
# Section 5 - Factory inference-mode load contract
# ---------------------------------------------------------------------------

print("=" * 60)
print("Section 5: Factory inference-mode load contract")
print("=" * 60)

# Factory inference-mode load contract (matches production policy):
#   • bare `cfr` with no default profile present is a CONVENIENT FALLBACK —
#     it degrades to the heuristic and still produces a valid action.
#   • an EXPLICIT `cfr:<path>` to a missing file is LOUD in inference mode:
#     it refuses to silently masquerade a typo'd path as a trained profile.
#   • DeepCFR missing-path is also loud (P0 safeguard).
legal_types = {"fold", "check", "call", "bet", "raise"}
view = PlayerView(
    me="hero",
    street="preflop",
    position="BTN",
    hole_cards=[("A", "h"), ("K", "h")],
    board=[],
    pot=15,
    to_call=10,
    min_raise=20,
    max_raise=200,
    legal_actions=[
        {"type": "fold"},
        {"type": "call"},
        {"type": "raise", "min": 20, "max": 200},
    ],
    stacks={"hero": 200, "villain": 200},
    opponents=["villain"],
    history=[],
)

# 1. Bare `cfr` → convenient heuristic fallback (no raise, valid action).
try:
    cfr = create_bot("cfr")
    cfr_action = cfr.act(view)
    print(f"  bare cfr action: {cfr_action.type}")
    _check(cfr_action.type in legal_types,
           "bare cfr degrades to heuristic and acts",
           "bare cfr did not produce a valid action")
except Exception as e:
    print(f"  FAIL - bare cfr raised unexpectedly: {type(e).__name__}: {e}")
    PASS = False

# 2. Explicit `cfr:<missing>` → loud fail in inference mode.
try:
    create_bot(f"cfr:{_missing_path('path_a.pkl')}")
    print("  FAIL - expected explicit cfr:<missing> to raise (inference safeguard)")
    PASS = False
except (FileNotFoundError, RuntimeError) as e:
    print(f"  OK - explicit cfr:<missing> raised {type(e).__name__}")

# 3. Explicit `deep_cfr:<missing>` → loud fail (P0 safeguard).
try:
    create_bot(f"deep_cfr:{_missing_path('path_b.pt')}")
    print("  FAIL - expected deep_cfr missing-path to raise (P0 safeguard)")
    PASS = False
except (FileNotFoundError, RuntimeError) as e:
    print(f"  OK - deep_cfr missing-path raised {type(e).__name__}")

print()


# ---------------------------------------------------------------------------
# Section 6 - Existing sanity scripts still pass
# ---------------------------------------------------------------------------

print("=" * 60)
print("Section 6: Regression sanity subprocesses")
print("=" * 60)

try:
    checks = [
        ("sanity_test_hand.py", 120),
        ("sanity_cfr_equity.py", 300),
        ("sanity_train_cfr.py", 900),
        ("sanity_train_deep_cfr.py", 900),
    ]
    for script, timeout in checks:
        proc = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode == 0:
            print(f"  PASS - {script} passed")
        else:
            print(f"  FAIL - {script} failed with exit {proc.returncode}")
            print("  STDOUT tail:")
            for line in proc.stdout.strip().splitlines()[-10:]:
                print(f"    {line}")
            print("  STDERR tail:")
            for line in proc.stderr.strip().splitlines()[-10:]:
                print(f"    {line}")
            PASS = False
except subprocess.TimeoutExpired as e:
    print(f"  FAIL - subprocess timed out: {e}")
    PASS = False
except Exception as e:
    print(f"  FAIL - exception: {type(e).__name__}: {e}")
    PASS = False

print()


# Cleanup: remove the temporary Path A profile / Path B weights for Sections 1-2.
for _tmp in (TEMP_CFR_PROFILE, TEMP_DEEP_CFR_WEIGHTS):
    if _tmp and os.path.exists(_tmp):
        os.unlink(_tmp)


print("=" * 70)
if PASS:
    print("ALL CHECKS PASSED")
else:
    print("SOME CHECKS FAILED")
    sys.exit(1)
print("=" * 70)
