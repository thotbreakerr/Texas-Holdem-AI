"""
Regression for the mid-run canary-abort signaling bug.

Bug (pre-fix): when the all-in collapse canary FAILed at a mid-run checkpoint,
the handler set abort_without_save=True and did a bare `raise`.  The outer try
only caught KeyboardInterrupt, so the RuntimeError propagated uncaught and the
documented `return {"status": "aborted", ...}` path was unreachable for the
mid-run case.

Fix: the handler `break`s instead, so the finally prints the ABORT footer and
run_training returns status="aborted" (which main() maps to a nonzero exit).
This test forces a mid-run canary FAIL and asserts run_training RETURNS the
aborted dict rather than raising.
"""
import os
import sys
import tempfile

sys.path.insert(0, "/Users/jaroslavaupart/Desktop/Projects/Texas-Holdem-AI")

import training.train_deep_cfr as tdc


def run():
    PASS = True
    tmp = tempfile.mkdtemp()
    save_path = os.path.join(tmp, "deep_cfr_probe.pt")

    # Force every canary probe to look like an all-in collapse → classify FAIL.
    orig_probe = tdc.quick_canary_probe
    tdc.quick_canary_probe = lambda *a, **k: {"raw_all_in": 0.99,
                                              "search_all_in": 0.99}

    # checkpoint fires at t=2 (mid-run; iterations=4) and must trip the canary.
    args = tdc.parse_args([
        "--variant", "small",
        "--iterations", "4",
        "--checkpoint-interval", "2",
        "--update-interval", "2",
        "--batch-size", "8",
        "--save-path", save_path,
        "--device", "cpu",
    ])

    raised = None
    result = None
    try:
        result = tdc.run_training(args)
    except Exception as e:  # noqa: BLE001 — we explicitly test for non-raising
        raised = e
    finally:
        tdc.quick_canary_probe = orig_probe

    if raised is not None:
        PASS = False
        print(f"[CHECK 1] FAIL — run_training raised instead of returning: "
              f"{type(raised).__name__}: {raised}")
    else:
        print("[CHECK 1] PASS — run_training returned without raising")

    if result is not None and result.get("status") == "aborted":
        print(f"[CHECK 2] PASS — status='aborted' (final_iter={result.get('final_iter')})")
    else:
        PASS = False
        got = result.get("status") if result else None
        print(f"[CHECK 2] FAIL — status is {got!r}, expected 'aborted'")

    # The aborted result must report the (lack of) last checkpoint so callers
    # can recover; with a FAIL on the very first checkpoint, nothing was saved.
    if result is not None and "checkpoint_saved" in result:
        print(f"[CHECK 3] PASS — reports checkpoint_saved="
              f"{result.get('checkpoint_saved')!r}")
    else:
        PASS = False
        print("[CHECK 3] FAIL — aborted result missing checkpoint_saved")

    print("=" * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED [PASS]' if PASS else 'SOME CHECKS FAILED [FAIL]'}")
    return PASS


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
