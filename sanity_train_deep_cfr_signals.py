"""
sanity_train_deep_cfr_signals.py — interrupt/terminate lifecycle gate (B5/M4 + B4/I9)

Pre-fix behavior this pins against:
  * SIGINT set a flag, broke the loop, and the run returned status="complete"
    with exit code 0 — an orchestrator could not tell an interrupted run from
    a finished one.
  * SIGTERM was untrapped, so an OS kill (sleep/shutdown/timeout) lost the
    in-flight checkpoint entirely.

Checks 1-3 run training/train_deep_cfr.py as a real subprocess; checks 5-6
run run_training in-process so they can stub the traversal/probe and count
canary calls.
NO check uses --disable-collapse-canary: the canary stays enabled everywhere,
because emergency (interrupt/abort) saves are canary-free by design (4.1) and
checks 1-5 never reach a periodic checkpoint (check 6 reaches one on
purpose — and forces it to FAIL).
  1. SIGTERM mid-run → checkpoint file exists, exit code 143 (128+15), and
     the summary reports "interrupted".
  2. SIGINT mid-run → same, exit code 130 (128+2).
  3. The interrupted checkpoint is a FINAL artifact below the all-in deploy
     gate, so it must carry the "shadow_only": true stamp (B4/I9) and the
     persisted nonfinite_skips counter (B2/I5 metadata).
  4. --iterations defaults to 1,000,000 (B4/I9 — the old 100k default sat
     below the 150k deploy gate).
  5. Exact iteration accounting + emergency save is canary-free (4.1): a real
     SIGINT raised DURING iteration 7 must report/checkpoint exactly 7
     completed iterations (pre-4.1 the loop variable leaked 8), and the
     checkpoint must be written WITHOUT consulting the collapse canary — the
     probe is stubbed to a guaranteed-FAIL verdict plus a call counter, so a
     blocked save OR any probe call fails the check.
  6. Emergency exits outrank a canary abort (4.2): a real SIGINT raised
     WHILE a periodic canary probe runs — where that probe then FAILs — must
     still save the emergency checkpoint, keep status="interrupted" with the
     true signal and completed-iteration count, and must NOT write or promote
     the .safe / .warn artifacts.  Pre-4.2, abort_without_save outranked the
     interrupt: the run reported "interrupted" yet saved nothing.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import threading
import time

# Repo root = this file's directory (gates live at the repo root), so the
# gate runs in any clone — never hard-code an absolute machine path here.
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import torch  # noqa: E402  (needs repo on sys.path first for checkpoint unpickling)

TRAINER = os.path.join(REPO, "training", "train_deep_cfr.py")
BANNER = "TRAINING DEEP CFR PLUS"


def _launch(save_path: str) -> tuple[subprocess.Popen, list[str]]:
    """Start a real training subprocess and wait until its banner appears.

    The banner prints AFTER the signal handlers are installed, so signaling
    any time after seeing it cannot race handler installation.  A reader
    thread drains stdout so the pipe can never fill and block the child.
    """
    cmd = [
        sys.executable, "-u", TRAINER,  # -u: unbuffered stdout through the pipe
        "--variant", "small",
        "--iterations", "500000",            # far more than we let it run
        "--update-interval", "1000000",      # no gradient steps needed
        "--checkpoint-interval", "1000000",  # no periodic checkpoints
        "--batch-size", "8",
        "--aivat-sims", "1",
        "--save-path", save_path,
        "--device", "cpu",
        # Collapse canary deliberately ENABLED: periodic checkpoints never
        # fire (interval above) and the emergency save is canary-free by
        # design (4.1), so the run stays deterministic without the
        # smoke-test-only --disable-collapse-canary flag.
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=REPO,
    )
    lines: list[str] = []

    def _drain():
        for line in proc.stdout:
            lines.append(line)

    threading.Thread(target=_drain, daemon=True).start()

    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if any(BANNER in line for line in lines):
            return proc, lines
        if proc.poll() is not None:
            raise AssertionError(
                "trainer exited before printing its banner:\n" + "".join(lines))
        time.sleep(0.2)
    proc.kill()
    raise AssertionError("trainer never printed its banner within 120s")


def interrupt_run(sig: signal.Signals, expected_rc: int, label: str):
    """Run, signal mid-run, and return (ok, checkpoint_path)."""
    ok = True
    tmpdir = tempfile.mkdtemp()
    save_path = os.path.join(tmpdir, f"sig_{label}.pt")

    proc, lines = _launch(save_path)
    time.sleep(3.0)  # let a few traversals run so we are genuinely mid-run
    proc.send_signal(sig)
    try:
        rc = proc.wait(timeout=180)
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"  [FAIL] — trainer did not exit within 180s of {sig.name}")
        return False, save_path
    output = "".join(lines)

    print(f"  exit code: {rc} (expected {expected_rc})")
    if rc == expected_rc:
        print(f"  [PASS] — {sig.name} exits {expected_rc}, not 0")
    else:
        ok = False
        print(f"  [FAIL] — wrong exit code for {sig.name}")
        print("  --- last output lines ---")
        for line in output.strip().splitlines()[-12:]:
            print(f"    {line}")

    if os.path.exists(save_path):
        print(f"  [PASS] — checkpoint saved on {sig.name}: {save_path}")
    else:
        ok = False
        print(f"  [FAIL] — no checkpoint written after {sig.name}")

    if "Training interrupted" in output and f"{sig.name}" in output:
        print(f"  [PASS] — summary reports status interrupted ({sig.name})")
    else:
        ok = False
        print("  [FAIL] — summary does not report an interrupted run")

    if "Training complete." in output:
        ok = False
        print("  [FAIL] — interrupted run still claimed 'complete' (pre-fix bug)")
    else:
        print("  [PASS] — interrupted run does not claim 'complete'")

    return ok, save_path


def exact_accounting_check() -> bool:
    """In-process: SIGINT during iteration 7 → exactly 7 iterations counted,
    emergency checkpoint saved without ever consulting the collapse canary.

    The traversal stub raises a REAL signal (os.kill to our own pid) while
    iteration 7 is running; the trainer's handler sets its flag, iteration 7
    finishes, and the loop must break at the top of iteration 8.  Pre-4.1
    the report and checkpoint metadata used the loop variable (8 — one more
    than actually ran), and the final save went through
    checkpoint_with_canary, whose FAIL verdict silently dropped the
    checkpoint.  The probe stub returns a guaranteed-FAIL verdict AND counts
    calls, so this check proves both fixes at once with the canary ENABLED.
    """
    import io
    from contextlib import redirect_stdout

    import training.train_deep_cfr as tdc
    from bots.deep_cfr_bot import DeepCFRBot

    ok = True
    tmpdir = tempfile.mkdtemp()
    save_path = os.path.join(tmpdir, "sig_accounting.pt")
    SIGNAL_AT = 7
    probe_calls = {"n": 0}

    def fail_probe(*_a, **_k):
        probe_calls["n"] += 1
        return {"raw_all_in": 1.0, "search_all_in": 1.0}  # guaranteed FAIL

    def killing_recurse(self, *_a, iteration=0, **_k):
        if iteration == SIGNAL_AT:
            os.kill(os.getpid(), signal.SIGINT)  # real signal, mid-iteration
        return 0.0  # skip the (slow, irrelevant) tree traversal

    orig_probe = tdc.quick_canary_probe
    orig_recurse = DeepCFRBot._cfr_recurse
    tdc.quick_canary_probe = fail_probe
    DeepCFRBot._cfr_recurse = killing_recurse
    try:
        args = tdc.parse_args([
            "--variant", "small",
            "--iterations", "1000",
            "--update-interval", "1000000",      # no gradient steps
            "--checkpoint-interval", "1000000",  # no periodic checkpoints
            "--batch-size", "8",
            "--aivat-sims", "1",
            "--save-path", save_path,
            "--device", "cpu",
            # collapse canary deliberately ENABLED
        ])
        with redirect_stdout(io.StringIO()):
            result = tdc.run_training(args)
    finally:
        tdc.quick_canary_probe = orig_probe
        DeepCFRBot._cfr_recurse = orig_recurse

    print(f"  status={result.get('status')!r}, signal={result.get('signal')}, "
          f"final_iter={result.get('final_iter')}, "
          f"canary_probe_calls={probe_calls['n']}")
    if (result.get("status") == "interrupted"
            and result.get("signal") == int(signal.SIGINT)):
        print("  [PASS] — in-process SIGINT reported as interrupted")
    else:
        ok = False
        print("  [FAIL] — in-process SIGINT not reported as interrupted")
    if result.get("final_iter") == SIGNAL_AT:
        print(f"  [PASS] — exactly {SIGNAL_AT} completed iterations reported")
    else:
        ok = False
        print(f"  [FAIL] — reported {result.get('final_iter')} iterations; "
              f"only {SIGNAL_AT} completed (off-by-one)")
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        if ckpt.get("iteration") == SIGNAL_AT:
            print(f"  [PASS] — checkpoint metadata iteration == {SIGNAL_AT}")
        else:
            ok = False
            print(f"  [FAIL] — checkpoint iteration "
                  f"{ckpt.get('iteration')} != {SIGNAL_AT}")
        if ckpt.get("shadow_only") is True:
            print("  [PASS] — emergency final save kept the shadow_only stamp")
        else:
            ok = False
            print("  [FAIL] — emergency save lost the shadow_only stamp")
    else:
        ok = False
        print("  [FAIL] — emergency checkpoint missing (canary blocked it?)")
    if probe_calls["n"] == 0:
        print("  [PASS] — collapse canary never consulted on the emergency path")
    else:
        ok = False
        print(f"  [FAIL] — canary probe ran {probe_calls['n']}x during an "
              f"emergency save")
    return ok


def interrupt_during_failing_canary_check() -> bool:
    """In-process: SIGINT lands WHILE a periodic canary probe is running and
    that probe then FAILs → the emergency checkpoint must still be saved (4.2).

    Pre-4.2, the FAIL verdict set abort_without_save and that flag outranked
    the interrupt in the final-save logic: the run reported
    status="interrupted" with a valid final_iter, yet saved NOTHING.  The
    probe stub raises the real signal mid-probe and then returns a
    guaranteed-FAIL verdict, reproducing the race deterministically.  The
    FAILing canary must still do its real job — no promoted .safe / side
    .warn artifact may appear; only the canary-free emergency save to
    --save-path.
    """
    import io
    from contextlib import redirect_stdout

    import training.train_deep_cfr as tdc
    from bots.deep_cfr_bot import DeepCFRBot

    ok = True
    tmpdir = tempfile.mkdtemp()
    save_path = os.path.join(tmpdir, "sig_during_canary.pt")
    CKPT_AT = 5  # first periodic checkpoint (--checkpoint-interval below)
    probe_calls = {"n": 0}

    def fail_probe_with_signal(*_a, **_k):
        probe_calls["n"] += 1
        # The operator's Ctrl-C arrives while the probe is mid-flight …
        os.kill(os.getpid(), signal.SIGINT)
        # … and the probe then reports a collapsed policy (FAIL verdict).
        return {"raw_all_in": 1.0, "search_all_in": 1.0}

    def stub_recurse(self, *_a, **_k):
        return 0.0  # skip the (slow, irrelevant) tree traversal

    orig_probe = tdc.quick_canary_probe
    orig_recurse = DeepCFRBot._cfr_recurse
    tdc.quick_canary_probe = fail_probe_with_signal
    DeepCFRBot._cfr_recurse = stub_recurse
    try:
        args = tdc.parse_args([
            "--variant", "small",
            "--iterations", "1000",
            "--update-interval", "1000000",        # no gradient steps
            "--checkpoint-interval", str(CKPT_AT),  # periodic canary fires
            "--batch-size", "8",
            "--aivat-sims", "1",
            "--all-in-deploy-iteration", str(CKPT_AT),
            "--save-path", save_path,
            "--device", "cpu",
            # Canary deliberately ENABLED and mature — it must actually FAIL.
        ])
        with redirect_stdout(io.StringIO()):
            result = tdc.run_training(args)
    finally:
        tdc.quick_canary_probe = orig_probe
        DeepCFRBot._cfr_recurse = orig_recurse

    print(f"  status={result.get('status')!r}, signal={result.get('signal')}, "
          f"final_iter={result.get('final_iter')}, "
          f"canary_probe_calls={probe_calls['n']}")
    if (result.get("status") == "interrupted"
            and result.get("signal") == int(signal.SIGINT)):
        print("  [PASS] — reported as a SIGINT interrupt, not a canary abort")
    else:
        ok = False
        print(f"  [FAIL] — interrupt status/reason lost "
              f"(status={result.get('status')!r})")
    if result.get("final_iter") == CKPT_AT:
        print(f"  [PASS] — exactly {CKPT_AT} completed iterations reported")
    else:
        ok = False
        print(f"  [FAIL] — reported {result.get('final_iter')} iterations; "
              f"{CKPT_AT} completed")
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        if ckpt.get("iteration") == CKPT_AT and not ckpt.get("shadow_only", False):
            print("  [PASS] — emergency checkpoint saved despite the FAILing "
                  "canary (correct mature iteration)")
        else:
            ok = False
            print(f"  [FAIL] — emergency checkpoint metadata wrong "
                  f"(iteration={ckpt.get('iteration')}, "
                  f"shadow_only={ckpt.get('shadow_only')})")
        if result.get("checkpoint_saved") == save_path:
            print("  [PASS] — result reports the emergency checkpoint path")
        else:
            ok = False
            print(f"  [FAIL] — result.checkpoint_saved is "
                  f"{result.get('checkpoint_saved')!r}, not the save path")
    else:
        ok = False
        print("  [FAIL] — no checkpoint saved: the FAILing canary blocked "
              "the emergency save (pre-4.2 bug)")
    safe_path = tdc.safe_checkpoint_path(save_path)
    warn_path = tdc.warn_checkpoint_path(save_path, CKPT_AT)
    if not os.path.exists(safe_path) and not os.path.exists(warn_path):
        print("  [PASS] — FAILing canary still blocked .safe/.warn promotion")
    else:
        ok = False
        print(f"  [FAIL] — a promoted/side checkpoint appeared despite FAIL "
              f"(safe={os.path.exists(safe_path)}, "
              f"warn={os.path.exists(warn_path)})")
    if probe_calls["n"] == 1:
        print("  [PASS] — canary probe ran exactly once (the periodic one); "
              "the emergency save added none")
    else:
        ok = False
        print(f"  [FAIL] — canary probe ran {probe_calls['n']}x (expected 1)")
    return ok


def run() -> bool:
    PASS = True

    print("=" * 60)
    print("Check 1: SIGTERM mid-run → checkpoint + exit 143")
    print("=" * 60)
    ok, term_ckpt = interrupt_run(signal.SIGTERM, 143, "term")
    PASS &= ok
    print()

    print("=" * 60)
    print("Check 2: SIGINT mid-run → checkpoint + exit 130")
    print("=" * 60)
    ok, _ = interrupt_run(signal.SIGINT, 130, "int")
    PASS &= ok
    print()

    print("=" * 60)
    print("Check 3: interrupted final checkpoint carries metadata stamps")
    print("=" * 60)
    if os.path.exists(term_ckpt):
        ckpt = torch.load(term_ckpt, map_location="cpu", weights_only=False)
        shadow = ckpt.get("shadow_only")
        skips = ckpt.get("nonfinite_skips")
        iteration = ckpt.get("iteration")
        print(f"  iteration={iteration}, shadow_only={shadow}, "
              f"nonfinite_skips={skips}")
        # The run was stopped long before the 150k deploy gate, so the FINAL
        # checkpoint must be stamped shadow-only (B4/I9).
        if shadow is True:
            print("  [PASS] — under-gate final checkpoint stamped shadow_only")
        else:
            PASS = False
            print("  [FAIL] — shadow_only stamp missing on under-gate final save")
        if skips == 0:
            print("  [PASS] — nonfinite_skips persisted (0 for a clean run)")
        else:
            PASS = False
            print("  [FAIL] — nonfinite_skips metadata missing/wrong")
    else:
        PASS = False
        print("  [FAIL] — no SIGTERM checkpoint to inspect")
    print()

    print("=" * 60)
    print("Check 4: --iterations defaults to 1,000,000")
    print("=" * 60)
    from training.train_deep_cfr import parse_args, ALL_IN_DEPLOY_ITERATION
    default_iters = parse_args(["--variant", "small"]).iterations
    print(f"  default --iterations = {default_iters:,} "
          f"(deploy gate {ALL_IN_DEPLOY_ITERATION:,})")
    if default_iters == 1_000_000 and default_iters >= ALL_IN_DEPLOY_ITERATION:
        print("  [PASS] — default run trains past the all-in deploy gate")
    else:
        PASS = False
        print("  [FAIL] — default --iterations is below the deploy gate")
    print()

    print("=" * 60)
    print("Check 5: exact iteration accounting + canary-free emergency save")
    print("=" * 60)
    PASS &= exact_accounting_check()
    print()

    print("=" * 60)
    print("Check 6: SIGINT during a FAILING periodic canary still saves (4.2)")
    print("=" * 60)
    PASS &= interrupt_during_failing_canary_check()
    print()

    print("=" * 60)
    if PASS:
        print("ALL CHECKS PASSED [PASS]")
    else:
        print("SOME CHECKS FAILED [FAIL]")
    print("=" * 60)
    return PASS


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
