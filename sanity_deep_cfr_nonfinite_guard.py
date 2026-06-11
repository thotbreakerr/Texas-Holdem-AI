"""
sanity_deep_cfr_nonfinite_guard.py — finiteness guard regressions (B2/I5)

Pre-fix, one NaN loss propagated through clip_grad_norm_ (which does NOT
protect against NaN) into every parameter, and the run kept saving silently
NaN-corrupted checkpoints while reporting "complete".

Checks:
  1. NaN injected into a regret target → train_step skips the optimizer step,
     parameters are bit-identical afterwards, and the skip is reported.
  2. Control: the same setup with finite targets DOES step and changes params.
  3. Threshold abort: a run whose every train_step reports a non-finite skip
     aborts with status="aborted" / abort_reason="nonfinite_loss" after
     NONFINITE_SKIPS_ABORT_THRESHOLD consecutive skips, exits nonzero via
     main()'s status mapping, still saves a final checkpoint (params are
     finite — the bad steps were skipped), and persists the skip counter in
     that checkpoint's metadata.
  4. Recovery: alternating skip / finite-step sequences never abort because a
     successful step resets the consecutive counter.
  5. Emergency-save guarantee (4.1): the same threshold abort with the
     collapse canary ENABLED and stubbed to a guaranteed-FAIL verdict still
     writes the final checkpoint, and the probe is never even consulted —
     pre-4.1 the finally block routed the abort save through
     checkpoint_with_canary, whose FAIL verdict silently dropped the
     checkpoint the abort message had just promised.
"""
from __future__ import annotations

import io
import math
import os
import sys
import tempfile
from contextlib import redirect_stdout

# Repo root = this file's directory (gates live at the repo root), so the
# gate runs in any clone — never hard-code an absolute machine path here.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import training.train_deep_cfr as tdc
from bots.deep_cfr_bot import (
    DeepCFRBot, DeepCFRConfig, ReservoirBuffer, NUM_ACTIONS,
)


def _snapshot(network) -> dict:
    return {k: v.detach().clone() for k, v in network.state_dict().items()}


def _params_equal(a: dict, b: dict) -> bool:
    return all(torch.equal(a[k], b[k]) for k in a)


def _make_regret_sample(bot, poison: bool):
    """One (input, target, mask, weight) regret-buffer entry; optionally NaN."""
    state = tdc.build_initial_state(n_seats=6, hero_seat=0)
    input_dict = state.to_network_input(0)
    target = torch.zeros(NUM_ACTIONS)
    mask = torch.ones(NUM_ACTIONS)
    if poison:
        target[0] = float("nan")  # masked-in NaN → NaN loss
    return (input_dict, target, mask, 1.0)


def run() -> bool:
    PASS = True
    batch_size = 8

    with redirect_stdout(io.StringIO()):
        bot = DeepCFRBot(config=DeepCFRConfig.small(), inference_mode=False,
                         aivat_sims=1)
    bot.network.to(torch.device("cpu"))
    optimizer = torch.optim.Adam(bot.network.parameters(), lr=1e-3)

    # ── Check 1: NaN target → step skipped, params untouched ────────────────
    print("=" * 60)
    print("Check 1: NaN loss skips the optimizer step")
    print("=" * 60)
    regret_buf = ReservoirBuffer()
    for i in range(batch_size):
        regret_buf.add(_make_regret_sample(bot, poison=True))
    empty_v, empty_s = ReservoirBuffer(), ReservoirBuffer()

    before = _snapshot(bot.network)
    r_loss, v_loss, s_loss, info = tdc.train_step(
        bot.network, optimizer, regret_buf, empty_v, empty_s,
        batch_size, torch.device("cpu"),
    )
    after = _snapshot(bot.network)

    print(f"  losses: r={r_loss}, v={v_loss}, s={s_loss}")
    print(f"  info: did_step={info['did_step']}, "
          f"nonfinite_skip={info.get('nonfinite_skip')}")
    if info.get("nonfinite_skip") is True and info["did_step"] is False:
        print("  [PASS] — non-finite loss reported as a skip, not a step")
    else:
        print("  [FAIL] — skip not reported (guard missing?)")
        PASS = False
    if math.isnan(r_loss):
        print("  [PASS] — component losses still reported (r is NaN, loggable)")
    else:
        print(f"  [FAIL] — expected NaN regret loss, got {r_loss}")
        PASS = False
    if _params_equal(before, after):
        print("  [PASS] — parameters bit-identical after the skipped step")
    else:
        print("  [FAIL] — parameters changed despite non-finite loss")
        PASS = False
    finite_params = all(torch.isfinite(v).all() for v in after.values())
    if finite_params:
        print("  [PASS] — parameters finite")
    else:
        print("  [FAIL] — NaN leaked into parameters")
        PASS = False
    print()

    # ── Check 2: control — finite targets do step ───────────────────────────
    print("=" * 60)
    print("Check 2: finite loss still steps (control)")
    print("=" * 60)
    finite_buf = ReservoirBuffer()
    for i in range(batch_size):
        finite_buf.add(_make_regret_sample(bot, poison=False))
    before = _snapshot(bot.network)
    _, _, _, info = tdc.train_step(
        bot.network, optimizer, finite_buf, empty_v, empty_s,
        batch_size, torch.device("cpu"),
    )
    after = _snapshot(bot.network)
    if info["did_step"] and not info.get("nonfinite_skip"):
        print("  [PASS] — finite batch performed an optimizer step")
    else:
        print("  [FAIL] — finite batch did not step")
        PASS = False
    if not _params_equal(before, after):
        print("  [PASS] — parameters changed on the finite step")
    else:
        print("  [FAIL] — parameters unchanged on a finite step")
        PASS = False
    print()

    # ── Check 3: threshold abort via run_training ───────────────────────────
    print("=" * 60)
    print(f"Check 3: {tdc.NONFINITE_SKIPS_ABORT_THRESHOLD} consecutive skips "
          f"abort the run")
    print("=" * 60)

    def skipping_train_step(*_a, **_k):
        nan = float("nan")
        return nan, 0.0, 0.0, {
            "heads_trained": {"regret": False, "value": False, "sizing": False},
            "did_step": False,
            "nonfinite_skip": True,
        }

    def stub_recurse(self, *_a, **_k):
        return 0.0  # skip the (slow, irrelevant) tree traversal entirely

    tmpdir = tempfile.mkdtemp()
    save_path = os.path.join(tmpdir, "nonfinite_abort.pt")
    original_train_step = tdc.train_step
    original_recurse = DeepCFRBot._cfr_recurse
    tdc.train_step = skipping_train_step
    DeepCFRBot._cfr_recurse = stub_recurse
    threshold = tdc.NONFINITE_SKIPS_ABORT_THRESHOLD
    try:
        args = tdc.parse_args([
            "--variant", "small",
            "--iterations", str(threshold * 4),  # plenty past the threshold
            "--update-interval", "1",            # one train_step per iteration
            "--checkpoint-interval", "1000000",  # no periodic checkpoints
            "--batch-size", "8",
            "--aivat-sims", "1",
            "--save-path", save_path,
            "--device", "cpu",
            "--disable-collapse-canary",
        ])
        with redirect_stdout(io.StringIO()) as out:
            result = tdc.run_training(args)
        text = out.getvalue()
    finally:
        tdc.train_step = original_train_step
        DeepCFRBot._cfr_recurse = original_recurse

    print(f"  status={result.get('status')!r}, "
          f"abort_reason={result.get('abort_reason')!r}, "
          f"final_iter={result.get('final_iter')}, "
          f"nonfinite_skips={result.get('nonfinite_skips')}")
    if (result.get("status") == "aborted"
            and result.get("abort_reason") == "nonfinite_loss"):
        print("  [PASS] — run aborted with abort_reason='nonfinite_loss'")
    else:
        print("  [FAIL] — run did not abort on persistent non-finite losses")
        PASS = False
    if result.get("final_iter") == threshold:
        print(f"  [PASS] — aborted exactly at iteration {threshold}")
    else:
        print(f"  [FAIL] — expected abort at {threshold}, "
              f"got {result.get('final_iter')}")
        PASS = False
    if result.get("nonfinite_skips") == threshold:
        print("  [PASS] — result reports the skip count")
    else:
        print("  [FAIL] — skip count missing/wrong in result")
        PASS = False
    if "[ABORT]" in text and "non-finite" in text:
        print("  [PASS] — abort reason printed")
    else:
        print("  [FAIL] — abort message missing from output")
        PASS = False
    # main() maps any non-"complete" status to a nonzero exit; assert the
    # mapping holds for this status (no subprocess needed).
    if result.get("status") != "complete":
        print("  [PASS] — status maps to nonzero exit via main()")
    else:
        print("  [FAIL] — status would exit 0")
        PASS = False
    # The final checkpoint is still written (params are finite) and carries
    # the persisted counter.
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        skips_meta = ckpt.get("nonfinite_skips")
        print(f"  checkpoint nonfinite_skips metadata: {skips_meta}")
        if skips_meta == threshold:
            print("  [PASS] — counter persisted in checkpoint metadata")
        else:
            print("  [FAIL] — checkpoint metadata missing the skip counter")
            PASS = False
    else:
        print("  [FAIL] — final checkpoint missing after nonfinite abort")
        PASS = False
    print()

    # ── Check 4: a finite step resets the consecutive counter ───────────────
    print("=" * 60)
    print("Check 4: alternating skip/step never aborts (consecutive reset)")
    print("=" * 60)

    call_count = {"n": 0}

    def alternating_train_step(*_a, **_k):
        call_count["n"] += 1
        if call_count["n"] % 2 == 1:
            return float("nan"), 0.0, 0.0, {
                "heads_trained": {"regret": False, "value": False,
                                  "sizing": False},
                "did_step": False,
                "nonfinite_skip": True,
            }
        return 0.5, 0.1, 0.0, {
            "heads_trained": {"regret": True, "value": True, "sizing": False},
            "did_step": True,
            "nonfinite_skip": False,
        }

    save_path2 = os.path.join(tmpdir, "nonfinite_recovery.pt")
    tdc.train_step = alternating_train_step
    DeepCFRBot._cfr_recurse = stub_recurse
    try:
        # 3x threshold total skips, but never 2 consecutive → must complete.
        args = tdc.parse_args([
            "--variant", "small",
            "--iterations", str(threshold * 6),
            "--update-interval", "1",
            "--checkpoint-interval", "1000000",
            "--batch-size", "8",
            "--aivat-sims", "1",
            "--save-path", save_path2,
            "--device", "cpu",
            "--disable-collapse-canary",
        ])
        with redirect_stdout(io.StringIO()):
            result2 = tdc.run_training(args)
    finally:
        tdc.train_step = original_train_step
        DeepCFRBot._cfr_recurse = original_recurse

    print(f"  status={result2.get('status')!r}, "
          f"nonfinite_skips={result2.get('nonfinite_skips')}")
    if result2.get("status") == "complete":
        print("  [PASS] — interleaved finite steps prevent the abort")
    else:
        print("  [FAIL] — run aborted despite recovering between skips")
        PASS = False
    if result2.get("nonfinite_skips") == threshold * 3:
        print("  [PASS] — total (non-consecutive) skips still counted")
    else:
        print(f"  [FAIL] — expected {threshold * 3} total skips, "
              f"got {result2.get('nonfinite_skips')}")
        PASS = False
    print()

    # ── Check 5: emergency save survives a FAILing canary (4.1) ─────────────
    print("=" * 60)
    print("Check 5: abort checkpoint survives a FAILing collapse canary")
    print("=" * 60)

    probe_calls = {"n": 0}

    def fail_probe(*_a, **_k):
        probe_calls["n"] += 1
        return {"raw_all_in": 1.0, "search_all_in": 1.0}  # guaranteed FAIL

    save_path3 = os.path.join(tmpdir, "nonfinite_abort_canary_fail.pt")
    original_probe = tdc.quick_canary_probe
    tdc.train_step = skipping_train_step
    DeepCFRBot._cfr_recurse = stub_recurse
    tdc.quick_canary_probe = fail_probe
    try:
        args = tdc.parse_args([
            "--variant", "small",
            "--iterations", str(threshold * 4),
            "--update-interval", "1",            # one train_step per iteration
            "--checkpoint-interval", "1000000",  # no periodic checkpoints
            "--batch-size", "8",
            "--aivat-sims", "1",
            "--save-path", save_path3,
            "--device", "cpu",
            # Collapse canary deliberately ENABLED (no --disable flag): the
            # emergency save after the nonfinite abort must neither consult
            # nor be blocked by it.
        ])
        with redirect_stdout(io.StringIO()):
            result3 = tdc.run_training(args)
    finally:
        tdc.train_step = original_train_step
        DeepCFRBot._cfr_recurse = original_recurse
        tdc.quick_canary_probe = original_probe

    print(f"  status={result3.get('status')!r}, "
          f"abort_reason={result3.get('abort_reason')!r}, "
          f"final_iter={result3.get('final_iter')}, "
          f"canary_probe_calls={probe_calls['n']}")
    if (result3.get("status") == "aborted"
            and result3.get("abort_reason") == "nonfinite_loss"):
        print("  [PASS] — abort semantics unchanged with the canary enabled")
    else:
        print("  [FAIL] — abort semantics changed under an enabled canary")
        PASS = False
    if os.path.exists(save_path3):
        ckpt3 = torch.load(save_path3, map_location="cpu", weights_only=False)
        if (ckpt3.get("iteration") == threshold
                and ckpt3.get("nonfinite_skips") == threshold):
            print(f"  [PASS] — checkpoint written at iteration {threshold} "
                  f"with the skip counter, despite the FAILing canary")
        else:
            print(f"  [FAIL] — checkpoint metadata wrong: "
                  f"iteration={ckpt3.get('iteration')}, "
                  f"nonfinite_skips={ckpt3.get('nonfinite_skips')}")
            PASS = False
    else:
        print("  [FAIL] — FAILing canary blocked the emergency checkpoint "
              "(pre-4.1 bug)")
        PASS = False
    if probe_calls["n"] == 0:
        print("  [PASS] — collapse canary never consulted on the abort path")
    else:
        print(f"  [FAIL] — canary probe ran {probe_calls['n']}x during the "
              f"emergency save")
        PASS = False
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
