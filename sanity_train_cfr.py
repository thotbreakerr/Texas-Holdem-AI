"""
sanity_train_cfr.py — Path A training pipeline pre-overnight gate
------------------------------------------------------------------
This is a SLOW gate (~5–8 minutes). It exercises the full Path A
training loop including AIVAT leaf evaluation and recursive tree CFR.
Run this before kicking off overnight training, not as part of a fast
pre-commit / pre-PR sanity sweep.

Sections:
  1. Short training run completes on a small budget
  2. Info-set count grows over the training run
  3. All saved info-set keys are 7-field format
  4. Checkpoint round-trip
  5. Resume training from checkpoint
  6. Anti-substitution: training invokes _cfr_recurse, NOT _estimate_action_value
  7. Existing sanity_cfr_equity.py still passes
  8. Multi-iteration regret accumulation
"""
import io
import os
import pickle
import random
import subprocess
import sys
import tempfile
import time as _time
from contextlib import redirect_stdout
from copy import deepcopy

sys.path.insert(0, ".")

from bots.cfr_bot import CFRBot, _CFRNode, _info_set_key
from training.train_cfr_bot_multiway import train_cfr_bot_multiway
from core.bot_api import PlayerView, Action

PASS = True

print("=" * 70)
print("sanity_train_cfr.py — Path A training pipeline pre-overnight gate")
print("=" * 70)
print("This is a SLOW gate (~5–8 minutes). It exercises the full Path A")
print("training loop including AIVAT leaf evaluation and tree CFR. Run this")
print("before kicking off any overnight training, NOT as part of a fast")
print("pre-commit / pre-PR sanity sweep.")
print("=" * 70)
print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 1 — Short training run completes on a small budget
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 1: Short training run (3 tournaments, iterations=1)")
print("=" * 60)

tmp_profile = None
train_bot = None
try:
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmp_profile = f.name

    # Delete the temp file so CFRBot doesn't try to load it
    os.unlink(tmp_profile)

    t0 = _time.monotonic()
    with redirect_stdout(io.StringIO()) as train_out:
        train_bot = train_cfr_bot_multiway(
            num_tournaments=3,
            chips_per_player=200,   # small stacks → shorter tournaments
            iterations=1,           # 1 traversal/decision (tree CFR is expensive)
            save_every=1,
            profile_path=tmp_profile,
        )
    elapsed = _time.monotonic() - t0
    train_text = train_out.getvalue()

    print(f"  Wall-clock: {elapsed:.1f}s")

    if elapsed <= 600:
        print("  [PASS] — completed under 600s")
    else:
        print(f"  [FAIL] — took {elapsed:.1f}s (>600s)")
        PASS = False

    if os.path.exists(tmp_profile) and os.path.getsize(tmp_profile) > 0:
        print(f"  [PASS] — profile saved ({os.path.getsize(tmp_profile)} bytes)")
    else:
        print("  [FAIL] — profile file missing or empty")
        PASS = False

    s = train_bot.stats()
    print(f"  Info sets:        {s['info_sets']}")
    print(f"  Total iterations: {s['total_iterations']}")
    print(f"  Recursion calls:  {s['recursion_calls']}")
    print(f"  Hands played:     {s['hands_played']}")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 2 — Info-set count grows over the training run
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 2: Info-set count grows during training")
print("=" * 60)

try:
    if train_bot is None:
        raise RuntimeError("Section 1 failed — no bot available")

    pre_count = 0   # fresh bot starts with 0 info sets
    post_count = len(train_bot._nodes)

    print(f"  Before training: {pre_count}")
    print(f"  After training:  {post_count}")

    if post_count > pre_count:
        print("  [PASS] — info sets grew from 0")
    else:
        print("  [FAIL] — info sets did not grow")
        PASS = False

except Exception as e:
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 3 — All saved info-set keys are 7-field format
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 3: All info-set keys are 7-field format (6 colons)")
print("=" * 60)

saved_data = None
try:
    if tmp_profile is None or not os.path.exists(tmp_profile):
        raise RuntimeError("No profile from Section 1")

    with open(tmp_profile, "rb") as f:
        saved_data = pickle.load(f)

    all_keys = list(saved_data["nodes"].keys())
    print(f"  Total keys in profile: {len(all_keys)}")

    bad_keys = []
    for k in all_keys:
        if k.count(":") != 6:
            bad_keys.append(k)

    if not bad_keys:
        print("  [PASS] — all keys have exactly 6 colons (7 fields)")
    else:
        print(f"  [FAIL] — {len(bad_keys)} keys have wrong format:")
        for bk in bad_keys[:5]:
            print(f"    {bk}  (colons={bk.count(':')})")
        PASS = False

    # Spot-check key structure
    if all_keys:
        sample = all_keys[0]
        fields = sample.split(":")
        print(f"  Sample key: {sample}")
        print(f"    Fields: street={fields[0]}, n_opp={fields[1]}, "
              f"position={fields[2]}, spr={fields[3]}, "
              f"opp_stat={fields[4]}, bucket={fields[5]}, "
              f"history={fields[6]}")

except Exception as e:
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 4 — Checkpoint round-trip
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 4: Checkpoint round-trip")
print("=" * 60)

try:
    if tmp_profile is None or not os.path.exists(tmp_profile):
        raise RuntimeError("No profile from Section 1")
    if train_bot is None:
        raise RuntimeError("No bot from Section 1")
    if saved_data is None:
        raise RuntimeError("No saved data from Section 3")

    # Load into a fresh bot
    fresh_bot = CFRBot(inference_mode=True, profile_path=tmp_profile)

    # Check node count matches
    orig_count = len(saved_data["nodes"])
    fresh_count = len(fresh_bot._nodes)
    print(f"  Saved keys:  {orig_count}")
    print(f"  Loaded keys: {fresh_count}")

    if fresh_count == orig_count:
        print("  [PASS] — node count matches")
    else:
        print(f"  [FAIL] — expected {orig_count}, got {fresh_count}")
        PASS = False

    # Profile loaded flag
    if fresh_bot._profile_loaded:
        print("  [PASS] — _profile_loaded is True")
    else:
        print("  [FAIL] — _profile_loaded is False")
        PASS = False

    # Spot-check one key's regret/strategy tensors
    if train_bot._nodes and fresh_bot._nodes:
        check_key = next(iter(train_bot._nodes))
        if check_key in fresh_bot._nodes:
            orig_node = train_bot._nodes[check_key]
            loaded_node = fresh_bot._nodes[check_key]
            regret_match = (orig_node.regret_sum == loaded_node.regret_sum)
            strat_match = (orig_node.strategy_sum == loaded_node.strategy_sum)
            if regret_match and strat_match:
                print(f"  [PASS] — regret/strategy match for key '{check_key[:40]}...'")
            else:
                print(f"  [FAIL] — tensor mismatch for key '{check_key[:40]}...'")
                PASS = False
        else:
            print(f"  [FAIL] — key '{check_key[:40]}...' not found in loaded bot")
            PASS = False

except Exception as e:
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 5 — Resume training from checkpoint
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 5: Resume training from checkpoint")
print("=" * 60)

try:
    if tmp_profile is None or not os.path.exists(tmp_profile):
        raise RuntimeError("No profile from Section 1")

    # Load the checkpoint into a fresh training bot
    resume_bot = CFRBot(
        iterations=1,
        profile_path=tmp_profile,
        use_average=True,
        inference_mode=False,
    )

    pre_iters = resume_bot._total_iterations
    pre_nodes = len(resume_bot._nodes)
    print(f"  Before resume: iters={pre_iters}, nodes={pre_nodes}")

    # Run 1 more tournament
    with redirect_stdout(io.StringIO()):
        train_cfr_bot_multiway(
            num_tournaments=1,
            chips_per_player=200,
            iterations=1,
            save_every=1,
            profile_path=tmp_profile,
        )

    # Re-load to verify
    resume_bot2 = CFRBot(
        iterations=1,
        profile_path=tmp_profile,
        inference_mode=True,
    )
    post_iters = resume_bot2._total_iterations
    post_nodes = len(resume_bot2._nodes)
    print(f"  After resume:  iters={post_iters}, nodes={post_nodes}")

    if post_iters > pre_iters:
        print("  [PASS] — total_iterations increased")
    else:
        print(f"  [FAIL] — total_iterations did not increase ({pre_iters} → {post_iters})")
        PASS = False

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 6 — Anti-substitution: training invokes _cfr_recurse, NOT
#              _estimate_action_value
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 6: Anti-substitution (_cfr_recurse vs _estimate_action_value)")
print("=" * 60)

try:
    bot6 = CFRBot(iterations=1, inference_mode=False)
    bot6._recursion_calls = 0

    # Monkey-patch _estimate_action_value to count calls
    _original_eav = bot6._estimate_action_value
    _eav_calls = [0]  # mutable counter (module scope can't use nonlocal)

    def _counting_eav(*args, **kwargs):
        _eav_calls[0] += 1
        return _original_eav(*args, **kwargs)

    bot6._estimate_action_value = _counting_eav

    # Build a synthetic PlayerView that forces act() through training path
    view6 = PlayerView(
        me="hero",
        street="preflop",
        position="BTN",
        hole_cards=[("A", "h"), ("A", "s")],
        board=[],
        pot=15,
        to_call=10,
        min_raise=20,
        max_raise=500,
        legal_actions=[
            {"type": "fold"},
            {"type": "call"},
            {"type": "raise", "min": 20, "max": 500},
        ],
        stacks={"hero": 500, "opp1": 500, "opp2": 500},
        opponents=["opp1", "opp2"],
        history=[],
    )

    # Call act() — training mode → _run_iterations → _cfr_recurse
    with redirect_stdout(io.StringIO()):
        action6 = bot6.act(view6)

    print(f"  _recursion_calls: {bot6._recursion_calls}")
    print(f"  _estimate_action_value calls: {_eav_calls[0]}")
    print(f"  Returned action: {action6.type}")

    if bot6._recursion_calls > 0:
        print("  [PASS] — _cfr_recurse was invoked during training")
    else:
        print("  [FAIL] — _cfr_recurse was NOT invoked during training.")
        print("    Training path is NOT using recursive tree CFR.")
        print("    This is the same class of substitution bug from Round 3.")
        PASS = False

    if _eav_calls[0] == 0:
        print("  [PASS] — _estimate_action_value was NOT called during training")
    else:
        print(f"  [FAIL] — _estimate_action_value was called {_eav_calls[0]} times")
        print("    Training is silently using the DEPRECATED heuristic value")
        print("    function instead of recursive tree CFR with AIVAT leaf evaluation.")
        print("    This is the substitution bug from Round 3.")
        PASS = False

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 7 — Existing sanity_cfr_equity.py still passes
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 7: sanity_cfr_equity.py regression check")
print("=" * 60)

try:
    result_proc = subprocess.run(
        [sys.executable, "sanity_cfr_equity.py"],
        capture_output=True, text=True, timeout=300,
    )
    if result_proc.returncode == 0:
        print("  [PASS] — sanity_cfr_equity.py passed")
    else:
        print("  [FAIL] — sanity_cfr_equity.py regressed")
        print("  STDOUT (last 10 lines):")
        for line in result_proc.stdout.strip().split("\n")[-10:]:
            print(f"    {line}")
        if result_proc.stderr:
            print("  STDERR (last 5 lines):")
            for line in result_proc.stderr.strip().split("\n")[-5:]:
                print(f"    {line}")
        PASS = False

except subprocess.TimeoutExpired:
    print("  [FAIL] — sanity_cfr_equity.py timed out (>300s)")
    PASS = False
except Exception as e:
    print(f"  [FAIL] — could not run: {e}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 8 — Multi-iteration regret accumulation
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 8: Multi-iteration regret accumulation")
print("=" * 60)

try:
    bot8 = CFRBot(
        iterations=5,
        profile_path=None,
        use_average=True,
        inference_mode=False,
    )

    view8 = PlayerView(
        me="hero",
        street="preflop",
        position="BTN",
        hole_cards=[("A", "h"), ("A", "s")],
        board=[],
        pot=15,
        to_call=10,
        min_raise=20,
        max_raise=500,
        legal_actions=[
            {"type": "fold"},
            {"type": "call"},
            {"type": "raise", "min": 20, "max": 500},
        ],
        stacks={"hero": 500, "opp1": 500, "opp2": 500},
        opponents=["opp1", "opp2"],
        history=[],
    )

    before_stats = bot8.stats()
    before_keys = set(bot8._nodes)
    before_nodes = len(bot8._nodes)

    with redirect_stdout(io.StringIO()):
        action8 = bot8.act(view8)

    after_stats = bot8.stats()
    after_keys = set(bot8._nodes)
    new_keys = list(after_keys - before_keys)

    iter_delta = after_stats["total_iterations"] - before_stats["total_iterations"]
    recursion_delta = after_stats["recursion_calls"] - before_stats["recursion_calls"]
    node_delta = len(bot8._nodes) - before_nodes

    print(f"  Returned action: {action8.type}")
    print(f"  total_iterations delta: {iter_delta}")
    print(f"  recursion_calls delta:  {recursion_delta}")
    print(f"  info-set delta:         {node_delta}")

    if iter_delta == 5:
        print("  [PASS] — total_iterations grew by exactly 5")
    else:
        print(f"  [FAIL] — expected total_iterations +5, got +{iter_delta}")
        PASS = False

    if recursion_delta >= 5:
        print("  [PASS] — recursion_calls grew by at least one call per traversal")
    else:
        print(f"  [FAIL] — expected recursion_calls +>=5, got +{recursion_delta}")
        PASS = False

    if node_delta > 0:
        print("  [PASS] — at least one info-set was created")
    else:
        print("  [FAIL] — no info-sets were created")
        PASS = False

    # Phase 3 (2026-06-10): textbook external sampling accumulates regret
    # at hero (traverser) nodes and strategy_sum at OPPONENT nodes — a
    # single info-set only gets both if it is visited in both roles across
    # traversals. Check each accumulator on its own node population.
    candidate_keys = new_keys if new_keys else list(bot8._nodes)
    regret_key = None
    strategy_key = None
    for key in candidate_keys:
        node = bot8._nodes[key]
        if regret_key is None and any(abs(x) > 1e-12 for x in node.regret_sum):
            regret_key = key
        if strategy_key is None and any(abs(x) > 1e-12 for x in node.strategy_sum):
            strategy_key = key
        if regret_key and strategy_key:
            break

    if regret_key is not None:
        print(f"  [PASS] — regret accumulated (hero node) "
              f"'{regret_key[:50]}...'")
    else:
        print("  [FAIL] — no info-set accumulated regret_sum")
        PASS = False
    if strategy_key is not None:
        print(f"  [PASS] — strategy_sum accumulated (opponent node) "
              f"'{strategy_key[:50]}...'")
    else:
        print("  [FAIL] — no info-set accumulated strategy_sum")
        PASS = False

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Cleanup
# ═══════════════════════════════════════════════════════════════════════════════

if tmp_profile and os.path.exists(tmp_profile):
    os.unlink(tmp_profile)


# ═══════════════════════════════════════════════════════════════════════════════
#  OVERALL
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
if PASS:
    print("ALL CHECKS PASSED [PASS]")
else:
    print("SOME CHECKS FAILED [FAIL]")
    sys.exit(1)
print("=" * 60)
