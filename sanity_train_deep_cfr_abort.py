"""
Regressions for canary maturity and mid-run abort signaling.

Before 100k, every metric is diagnostic. At/after the configured enforcement
boundary, three consecutive failing checkpoints abort without promotion.

The mature check also pins the older signaling fix: run_training returns the
documented aborted dict rather than propagating RuntimeError.
"""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import training.train_deep_cfr as tdc


def run():
    PASS = True
    orig_probe = tdc.quick_canary_probe
    tdc.quick_canary_probe = lambda *a, **k: {"raw_all_in": 0.99,
                                              "search_all_in": 0.99}

    try:
        with tempfile.TemporaryDirectory() as tmp:
            early_path = os.path.join(tmp, "early.pt")
            early_args = tdc.parse_args([
                "--variant", "small",
                "--iterations", "4",
                "--checkpoint-interval", "2",
                "--update-interval", "2",
                "--fit-steps", "2",
                "--fit-batch-size", "8",
                "--batch-size", "8",
                "--save-path", early_path,
                "--device", "cpu",
            ])
            early = tdc.run_training(early_args)
            early_saved = os.path.exists(early_path)

            mature_path = os.path.join(tmp, "mature.pt")
            mature_args = tdc.parse_args([
                "--variant", "small",
                "--iterations", "6",
                "--checkpoint-interval", "2",
                "--update-interval", "2",
                "--fit-steps", "3",
                "--fit-batch-size", "8",
                "--batch-size", "8",
                "--canary-enforce-iteration", "0",
                "--canary-fail-patience", "3",
                "--save-path", mature_path,
                "--device", "cpu",
            ])
            raised = None
            mature = None
            try:
                mature = tdc.run_training(mature_args)
            except Exception as exc:  # noqa: BLE001
                raised = exc
            mature_saved = os.path.exists(mature_path)
    finally:
        tdc.quick_canary_probe = orig_probe

    if early.get("status") == "complete" and early_saved:
        print("[CHECK 1] PASS — pre-deploy FAIL is diagnostic and checkpoint saved")
    else:
        PASS = False
        print(f"[CHECK 1] FAIL — early status={early.get('status')!r}, "
              f"checkpoint={early_saved}")

    if raised is not None:
        PASS = False
        print(f"[CHECK 2] FAIL — mature run raised instead of returning: "
              f"{type(raised).__name__}: {raised}")
    else:
        print("[CHECK 2] PASS — mature canary abort returned without raising")

    if (mature is not None and mature.get("status") == "aborted"
            and mature.get("abort_reason") == "collapse_canary"
            and mature.get("canary_fail_streak") == 3
            and not mature_saved):
        print(f"[CHECK 3] PASS — third consecutive mature FAIL aborted at iter "
              f"{mature.get('final_iter')} without promotion")
    else:
        PASS = False
        print(f"[CHECK 3] FAIL — mature result={mature}, file={mature_saved}")

    print("=" * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED [PASS]' if PASS else 'SOME CHECKS FAILED [FAIL]'}")
    return PASS


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
