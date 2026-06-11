"""
sanity_cfr_root_coverage.py — decision-root deployability + profile format gate.

Phase 3.1 (2026-06-11). Guards two retrain blockers:

ROOT COVERAGE / DEPLOYABILITY
  Training traversals are decision-rooted with the hero acting first, and
  external sampling accumulates strategy_sum at opponent nodes. An infoset
  that only ever occurs as a traversal root — the hand's FIRST voluntary
  action, whose token history is empty — therefore collected regret_sum but
  ZERO strategy_sum, and act() discarded the training (search/heuristic
  fallback). Fixed by (a) accumulating strategy_sum at the depth-0 hero
  root (hero reach there is exactly 1, so the unweighted increment is the
  textbook average-strategy update) and (b) act() deploying the
  regret-matched current strategy when an infoset has positive regret but
  no average mass.

PROFILE FORMAT GATE
  save() stamps format_version = PROFILE_FORMAT_VERSION; load() refuses to
  resume TRAINING from missing/lower versions unless allow_stale_profile.
  Inference-mode loads stay permitted.

Sections:
  1. First-to-act root infoset accumulates BOTH regret and strategy mass
  2. act() (inference) deploys the trained root node — no fallback path
  3. Regret-only node (zero strategy_sum) deploys via regret matching
  4. save() stamps format_version; fresh v3 profile resumes in training mode
  5. Unversioned/lower-version profile: training load rejects, override and
     inference-mode loads still work
"""
import io
import os
import pickle
import random
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bots.cfr_bot import (
    CFRBot, _CFRNode, PROFILE_FORMAT_VERSION,
    _info_set_key, _preflop_bucket, _position_bucket, _spr_bucket,
)
from core.bot_api import PlayerView

PASS = True
random.seed(31)

HOLE = [("A", "h"), ("K", "h")]


def make_first_to_act_view() -> PlayerView:
    """6-max UTG preflop opening decision: EMPTY history (blinds are not
    history entries), hero seat index 3 == UTG in the engine ring."""
    return PlayerView(
        me="P4",
        street="preflop",
        position="UTG",
        hole_cards=list(HOLE),
        board=[],
        pot=15,                  # SB 5 + BB 10
        to_call=10,
        min_raise=20,
        max_raise=1000,
        legal_actions=[
            {"type": "fold"},
            {"type": "call"},
            {"type": "raise", "min": 20, "max": 1000},
        ],
        stacks={f"P{i}": 1000 for i in range(1, 7)},
        opponents=["P1", "P2", "P3", "P5", "P6"],
        history=[],
    )


def attach_fallback_recorders(bot: CFRBot) -> dict:
    """Wrap the two fallback paths so we can tell which path act() took."""
    paths = {"search_fallback": False, "heuristic": False}
    orig_search = bot._search_fallback
    orig_heur = bot._heuristic_action

    def rec_search(*a, **kw):
        paths["search_fallback"] = True
        return orig_search(*a, **kw)

    def rec_heur(*a, **kw):
        paths["heuristic"] = True
        return orig_heur(*a, **kw)

    bot._search_fallback = rec_search
    bot._heuristic_action = rec_heur
    return paths


print("=" * 70)
print("sanity_cfr_root_coverage.py — root deployability + profile format gate")
print("=" * 70)
print()

# ═══════════════════════════════════════════════════════════════════════════════
#  Section 1 — first-to-act root infoset accumulates regret AND strategy mass
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 1: root infoset gets regret AND strategy mass")
print("=" * 60)

train_bot = None
try:
    train_bot = CFRBot(iterations=40, profile_path=None, use_average=True,
                       inference_mode=False)
    view = make_first_to_act_view()
    with redirect_stdout(io.StringIO()):
        train_bot.act(view)

    root_keys = [k for k in train_bot._nodes if k.split(":", 6)[6] == ""]
    deep_keys = [k for k in train_bot._nodes if k.split(":", 6)[6] != ""]
    root_regret = sum(sum(abs(x) for x in train_bot._nodes[k].regret_sum)
                      for k in root_keys)
    root_strat = sum(sum(train_bot._nodes[k].strategy_sum) for k in root_keys)
    deep_strat = sum(sum(train_bot._nodes[k].strategy_sum) for k in deep_keys)

    print(f"  empty-history root keys: {len(root_keys)}")
    print(f"  root regret mass:        {root_regret:.3f}")
    print(f"  root strategy mass:      {root_strat:.3f}")
    print(f"  deeper strategy mass:    {deep_strat:.3f}")

    if root_keys:
        print("  [PASS] — first-to-act root infoset exists")
    else:
        print("  [FAIL] — no empty-history root infoset created")
        PASS = False
    if root_regret > 0:
        print("  [PASS] — root accumulated regret_sum")
    else:
        print("  [FAIL] — root has zero regret mass")
        PASS = False
    if root_strat > 0:
        print("  [PASS] — root accumulated strategy_sum (deployable average)")
    else:
        print("  [FAIL] — root strategy_sum is ZERO: first-to-act training "
              "is not deployable (act() would fall back)")
        PASS = False
    if deep_strat > 0:
        print("  [PASS] — opponent-node averaging still accumulates too")
    else:
        print("  [FAIL] — no deeper node accumulated strategy_sum")
        PASS = False

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()

# ═══════════════════════════════════════════════════════════════════════════════
#  Section 2 — act() deploys the trained root node (no fallback path)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 2: inference act() uses the trained node, not a fallback")
print("=" * 60)

try:
    if train_bot is None or not train_bot._nodes:
        raise RuntimeError("Section 1 produced no trained table")

    inf = CFRBot(iterations=0, profile_path=None, use_average=True,
                 inference_mode=True)
    inf._nodes = train_bot._nodes
    # _profile_loaded stays False on purpose: the fallback branch under test
    # sits BEFORE the subgame search, so this keeps the check deterministic.
    paths = attach_fallback_recorders(inf)
    with redirect_stdout(io.StringIO()):
        action = inf.act(make_first_to_act_view())

    print(f"  chosen action: {action.type}"
          f"{' ' + str(action.amount) if action.amount is not None else ''}")
    print(f"  search_fallback={paths['search_fallback']} "
          f"heuristic={paths['heuristic']}")
    if not paths["search_fallback"] and not paths["heuristic"]:
        print("  [PASS] — trained first-to-act infoset is deployable")
    else:
        print("  [FAIL] — act() fell back; trained root mass was discarded")
        PASS = False

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()

# ═══════════════════════════════════════════════════════════════════════════════
#  Section 3 — regret-only node (zero strategy_sum) deploys via regret matching
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 3: regret-only node deploys regret-matched strategy")
print("=" * 60)

try:
    view = make_first_to_act_view()
    # Reconstruct the exact live info key act() will build for this view.
    key = _info_set_key(
        "preflop",
        _preflop_bucket(HOLE),
        "",                              # empty token history
        n_opponents=5,
        position_bucket=_position_bucket("UTG"),
        spr_bucket=_spr_bucket(1000, 15, [1000] * 5),
        opp_stat_bucket="TA-TA-TA-TA-TA",
    )

    node = _CFRNode()
    node.regret_sum[1] = 100.0           # all positive regret on check_call
    # strategy_sum stays all-zero: the pre-Phase-3.1 profile shape.

    inf3 = CFRBot(iterations=0, profile_path=None, use_average=True,
                  inference_mode=True)
    inf3._nodes = {key: node}
    paths3 = attach_fallback_recorders(inf3)
    with redirect_stdout(io.StringIO()):
        action3 = inf3.act(view)

    print(f"  planted key: {key}")
    print(f"  chosen action: {action3.type}")
    print(f"  search_fallback={paths3['search_fallback']} "
          f"heuristic={paths3['heuristic']}")

    if not paths3["search_fallback"] and not paths3["heuristic"]:
        print("  [PASS] — no fallback despite strategy_sum == 0")
    else:
        print("  [FAIL] — regret-only node was discarded by act()")
        PASS = False
    if action3.type == "call":
        print("  [PASS] — action follows the regret-matched strategy "
              "(100% check_call)")
    else:
        print(f"  [FAIL] — expected 'call' from regret matching, got "
              f"{action3.type!r}")
        PASS = False

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()

# ═══════════════════════════════════════════════════════════════════════════════
#  Section 4 — save() stamps format_version; fresh profile resumes training
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print(f"Section 4: save() stamps format_version={PROFILE_FORMAT_VERSION}; "
      "fresh profile resumes")
print("=" * 60)

tmp_v3 = None
try:
    if train_bot is None or not train_bot._nodes:
        raise RuntimeError("Section 1 produced no trained table")

    fd, tmp_v3 = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    train_bot.save(tmp_v3)

    with open(tmp_v3, "rb") as f:
        raw = pickle.load(f)
    stamped = raw.get("format_version")
    print(f"  saved format_version: {stamped}")
    if stamped == PROFILE_FORMAT_VERSION:
        print("  [PASS] — save() stamps the current format version")
    else:
        print(f"  [FAIL] — expected {PROFILE_FORMAT_VERSION}, got {stamped}")
        PASS = False

    # The critical resume property: a checkpoint written DURING the retrain
    # must load back in training mode without tripping the gate.
    with redirect_stdout(io.StringIO()):
        resume = CFRBot(iterations=1, profile_path=tmp_v3,
                        inference_mode=False)
    if len(resume._nodes) == len(train_bot._nodes):
        print(f"  [PASS] — fresh v{PROFILE_FORMAT_VERSION} checkpoint resumes "
              f"in training mode ({len(resume._nodes)} nodes)")
    else:
        print(f"  [FAIL] — resume loaded {len(resume._nodes)} nodes, "
              f"expected {len(train_bot._nodes)}")
        PASS = False

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()

# ═══════════════════════════════════════════════════════════════════════════════
#  Section 5 — stale profiles: training load rejects; override + inference load
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 5: stale-profile gate (reject / override / inference)")
print("=" * 60)

tmp_stale = None
try:
    if tmp_v3 is None or not os.path.exists(tmp_v3):
        raise RuntimeError("Section 4 produced no profile")

    with open(tmp_v3, "rb") as f:
        raw = pickle.load(f)

    for label, version in (("missing", None), ("lower (2)", 2)):
        data = dict(raw)
        if version is None:
            data.pop("format_version", None)
        else:
            data["format_version"] = version
        fd, tmp_stale = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        with open(tmp_stale, "wb") as f:
            pickle.dump(data, f)

        # 5a. Training-mode load must REJECT.
        try:
            with redirect_stdout(io.StringIO()):
                CFRBot(iterations=1, profile_path=tmp_stale,
                       inference_mode=False)
            print(f"  [FAIL] — training load accepted {label} format_version")
            PASS = False
        except RuntimeError as e:
            print(f"  [PASS] — training load rejected {label} "
                  f"format_version ({str(e)[:60]}...)")

        # 5b. Explicit override must load.
        try:
            with redirect_stdout(io.StringIO()):
                over = CFRBot(iterations=1, profile_path=tmp_stale,
                              inference_mode=False, allow_stale_profile=True)
            if over._nodes:
                print(f"  [PASS] — allow_stale_profile=True loads {label} "
                      f"profile ({len(over._nodes)} nodes)")
            else:
                print(f"  [FAIL] — override load produced 0 nodes ({label})")
                PASS = False
        except RuntimeError as e:
            print(f"  [FAIL] — override still rejected ({label}): {e}")
            PASS = False

        # 5c. Inference-mode load stays permitted.
        try:
            with redirect_stdout(io.StringIO()):
                inf5 = CFRBot(profile_path=tmp_stale, inference_mode=True)
            if inf5._nodes:
                print(f"  [PASS] — inference mode loads {label} profile")
            else:
                print(f"  [FAIL] — inference load produced 0 nodes ({label})")
                PASS = False
        except RuntimeError as e:
            print(f"  [FAIL] — inference load rejected ({label}): {e}")
            PASS = False

        os.unlink(tmp_stale)
        tmp_stale = None

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False
finally:
    for p in (tmp_v3, tmp_stale):
        if p and os.path.exists(p):
            os.unlink(p)

print()

# ═══════════════════════════════════════════════════════════════════════════════
#  OVERALL
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
if PASS:
    print("OVERALL: ALL CHECKS PASSED")
else:
    print("OVERALL: SOME CHECKS FAILED")
print("=" * 60)
sys.exit(0 if PASS else 1)
