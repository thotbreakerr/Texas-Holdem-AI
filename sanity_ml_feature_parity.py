"""
sanity_ml_feature_parity.py — Phase 2: ML feature train/inference parity
-------------------------------------------------------------------------
Proves that supervised training features (training/train_ml_bot.py /
PokerDataset) mean exactly the same thing as MLBot inference features
(bots/ml_bot.py), via the shared builder in core/ml_features.py.

Covers the three historical mismatches — the equality assertions here FAIL
on the old code:
  1. Preflop hand strength: training divided by EVAL_HAND_MAX (collapsing
     AA and 72o both to ~4e-7); inference used the preflop heuristic max.
  2. Position: engine logs omitted position, so training defaulted every
     sample to MP (0.3) regardless of the true seat.
  3. Memory/VPIP: inference counted checks as VPIP and used cumulative
     per-opponent stats; training used a pooled last-10-decisions window.

Phase 2.1 wiring gates:
  5. Session safety: legacy/per-hand logs (no session header) are rejected
     for cumulative-memory training unless explicitly allowed; memory dedup
     does not swallow a new tournament whose Table reuses hand ids.
  6. Checkpoint schema versioning: MLBot refuses raw legacy state dicts and
     wrong-version checkpoints; loads only current-version checkpoints.

Phase 2.1b strictness gates:
  6e-6h. Malformed v2 checkpoints (missing/non-dict/garbage state_dict,
     corrupted weights) are refused loudly without crashing.
  7. Strict session headers: the header must be the FIRST row, appear
     exactly once, and carry a nonempty session_id + log_format_version=2;
     late/duplicate/empty/wrong-version/malformed headers raise even with
     allow_unmarked_sessions=True.
  8. UI restart safety: bot adapters forward reset_memory() to MLBot;
     run_tournament.py resets bot memory at tournament start so reused
     instances survive restarted hand ids.

Phase 2.1c fail-closed gates:
  6i-6j. Unreadable checkpoint files (empty -> EOFError, random bytes ->
     pickle.UnpicklingError) fall back loudly instead of crashing.
  7h-7k. Malformed log rows (invalid JSON, top-level non-objects) are hard
     errors regardless of allow_unmarked_sessions — a damaged second
     session_start must not be skipped as noise and merge two tournaments
     into one cumulative-memory replay.

Feature layout indices (core/ml_features.py): hand_strength=20, pot_odds=21,
position=22, memory=[23 aggression, 24 tightness, 25 vpip].
"""
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from core.bot_api import Action, PlayerView
from core.engine import Table, Seat
from core.logger import DecisionLogger
from core.ml_features import POSITION_ORDER, OpponentMemory
from bots.ml_bot import MLBot
from training.train_ml_bot import PokerDataset

IDX_STRENGTH, IDX_POT_ODDS, IDX_POSITION = 20, 21, 22
IDX_MEM_AGGR, IDX_MEM_TIGHT, IDX_MEM_VPIP = 23, 24, 25

AA = [["A", "s"], ["A", "h"]]
O72 = [["7", "d"], ["2", "c"]]

PASS = True


def check(cond, label):
    global PASS
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    if not cond:
        PASS = False


def fresh_bot():
    return MLBot(model_path="/nonexistent.pt", use_fallback=True,
                 starting_chips=500)


def live_features(bot, view):
    """Exactly the act() feature path: update memory, then build features."""
    bot._update_memory(view)
    return bot._make_features(view).squeeze(0).tolist()


def write_rows(rows, tmpdir, session_header=True,
               fname="session_synthetic_0.jsonl"):
    path = os.path.join(tmpdir, fname)
    with open(path, "w") as f:
        if session_header:
            f.write(json.dumps({"session_start": {
                "session_id": "synthetic", "log_format_version": 2}}) + "\n")
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def dataset_from_rows(rows, tmpdir, **kwargs):
    write_rows(rows, tmpdir)
    return PokerDataset(tmpdir, **kwargs)


def make_row(player, hole, position, *, street="preflop", board=None, pot=30,
             to_call=20, stacks=None, opponents=None, acting=None,
             chosen=None, hand_id=0):
    stacks = stacks or {"HERO": 500, "OPP_A": 500, "OPP_B": 500}
    opponents = opponents if opponents is not None else [p for p in stacks if p != player]
    return {
        "player": player,
        "position": position,
        "street": street,
        "hole": hole,
        "board": board or [],
        "pot": pot,
        "to_call": to_call,
        "legal": [{"type": "fold"}, {"type": "call"},
                  {"type": "raise", "min": 40, "max": 500}],
        "chosen_action": chosen or {"type": "call", "amount": None},
        "stacks": stacks,
        "opponents": opponents,
        "acting_opponents": acting if acting is not None else opponents,
        "all_in_opponents": [],
        "folded": False,
        "hand_id": hand_id,
    }


def make_view(row, history=None):
    """Live PlayerView equivalent of a logged decision row."""
    return PlayerView(
        me=row["player"],
        street=row["street"],
        position=row["position"],
        hole_cards=[tuple(c) for c in row["hole"]],
        board=[tuple(c) for c in row["board"]],
        pot=row["pot"],
        to_call=row["to_call"],
        min_raise=40,
        max_raise=500,
        legal_actions=row["legal"],
        stacks=row["stacks"],
        opponents=row["opponents"],
        history=history or [],
        hand_id=row["hand_id"],
        acting_opponents=row["acting_opponents"],
        all_in_opponents=row["all_in_opponents"],
    )


def compare(train_x, live_x, label):
    same = torch.allclose(torch.tensor(train_x, dtype=torch.float32),
                          torch.tensor(live_x, dtype=torch.float32),
                          atol=1e-6)
    if not same:
        diffs = [(i, t, l) for i, (t, l) in enumerate(zip(train_x, live_x))
                 if abs(t - l) > 1e-6]
        print(f"      diff at indices: {diffs}")
    check(same, label)


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 1: Preflop hand-strength normalization parity (AA and 72o)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 1: Preflop hand-strength normalization (AA, 72o)")
print("=" * 70)

with tempfile.TemporaryDirectory() as tmp:
    rows = [
        make_row("HERO", AA, "BTN", hand_id=0),
        make_row("HERO", O72, "BTN", hand_id=1),
    ]
    ds = dataset_from_rows(rows, tmp)
    (x_aa, _), (x_72, _) = ds[0], ds[1]

    bot = fresh_bot()
    live_aa = live_features(bot, make_view(rows[0]))
    live_72 = live_features(fresh_bot(), make_view(rows[1]))

    compare(x_aa.tolist(), live_aa, "AA: training vector == inference vector")
    compare(x_72.tolist(), live_72, "72o: training vector == inference vector")

    s_aa, s_72 = x_aa[IDX_STRENGTH].item(), x_72[IDX_STRENGTH].item()
    print(f"      training strengths: AA={s_aa:.4f}, 72o={s_72:.4f}")
    # Old bug: training divided preflop scores by EVAL_HAND_MAX -> both ~4e-7.
    check(s_aa > 0.6, f"AA training strength {s_aa:.4f} > 0.6 (not collapsed)")
    check(s_72 > 0.01, f"72o training strength {s_72:.4f} > 0.01 (not collapsed)")
    check(s_aa > s_72, "AA stronger than 72o in training features")
print()

# ═══════════════════════════════════════════════════════════════════════════
#  TEST 2: Position survives logging and feature construction
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 2: Non-MP positions survive logging -> training features")
print("=" * 70)

with tempfile.TemporaryDirectory() as tmp:
    positions = ["BTN", "UTG", "BB", "CO"]
    rows = [make_row("HERO", AA, pos, hand_id=i)
            for i, pos in enumerate(positions)]
    ds = dataset_from_rows(rows, tmp)

    for i, pos in enumerate(positions):
        x, _ = ds[i]
        got = x[IDX_POSITION].item()
        want = POSITION_ORDER[pos]
        check(abs(got - want) < 1e-6,
              f"{pos}: training position feature {got:.2f} == {want:.2f}")
        live_x = live_features(fresh_bot(), make_view(rows[i]))
        compare(x.tolist(), live_x, f"{pos}: full vector parity")

    # Old bug: missing position in logs -> every sample defaulted to MP (0.3).
    x_btn, _ = ds[0]
    check(abs(x_btn[IDX_POSITION].item() - POSITION_ORDER["MP"]) > 1e-6,
          "BTN sample is NOT the old MP default")
print()

# ═══════════════════════════════════════════════════════════════════════════
#  TEST 3: Memory / VPIP / aggression parity (scripted session)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 3: Memory/VPIP/aggression — training replay == live cumulative")
print("=" * 70)

with tempfile.TemporaryDirectory() as tmp:
    stacks = {"HERO": 500, "OPP_A": 500, "OPP_B": 500, "OPP_C": 500}
    opps = ["OPP_A", "OPP_B", "OPP_C"]

    # Hand 0: A raises (aggro), B calls (passive vpip), C checks (NOT vpip),
    # then HERO acts. Hand 1: A folds, then HERO acts — memory must carry
    # over hand 0 cumulatively.
    rows = [
        make_row("OPP_A", O72, "UTG", stacks=stacks, hand_id=0,
                 chosen={"type": "raise", "amount": 40}),
        make_row("OPP_B", O72, "MP", stacks=stacks, hand_id=0,
                 chosen={"type": "call", "amount": None}),
        make_row("OPP_C", O72, "CO", stacks=stacks, hand_id=0, to_call=0,
                 chosen={"type": "check", "amount": None}),
        make_row("HERO", AA, "BTN", stacks=stacks, hand_id=0, acting=opps,
                 chosen={"type": "call", "amount": None}),
        make_row("OPP_A", O72, "MP", stacks=stacks, hand_id=1,
                 chosen={"type": "fold", "amount": None}),
        make_row("HERO", AA, "CO", stacks=stacks, hand_id=1, acting=opps,
                 chosen={"type": "call", "amount": None}),
    ]
    ds = dataset_from_rows(rows, tmp, filter_players=["HERO"])
    check(len(ds) == 2, f"dataset kept 2 HERO samples (got {len(ds)})")

    # --- Live side: one MLBot accumulating across both hands ---
    bot = fresh_bot()
    hist_h0 = [
        {"street": "preflop", "pid": "OPP_A", "type": "raise", "amount": 40},
        {"street": "preflop", "pid": "OPP_B", "type": "call", "amount": None},
        {"street": "preflop", "pid": "OPP_C", "type": "check", "amount": None},
    ]
    live_h0 = live_features(bot, make_view(rows[3], history=hist_h0))
    hist_h1 = [
        {"street": "preflop", "pid": "OPP_A", "type": "fold", "amount": None},
    ]
    live_h1 = live_features(bot, make_view(rows[5], history=hist_h1))

    x0, _ = ds[0]
    x1, _ = ds[1]
    compare(x0.tolist(), live_h0, "hand 0 decision: full vector parity")
    compare(x1.tolist(), live_h1, "hand 1 decision: full vector parity")

    # Expected hand-0 memory: A(raise): aggr=1,vpip=1; B(call): vpip=1;
    # C(check): vpip=0 (checks are NOT VPIP). avg = [1/3, 0, 2/3]
    mem0 = x0[IDX_MEM_AGGR:].tolist()
    print(f"      hand 0 memory features: {[round(v, 4) for v in mem0]}")
    check(abs(mem0[0] - 1 / 3) < 1e-6, "aggression = 1/3 (only A raised)")
    check(abs(mem0[1] - 0.0) < 1e-6, "tightness = 0 (no folds yet)")
    check(abs(mem0[2] - 2 / 3) < 1e-6,
          "VPIP = 2/3 — OPP_C's check NOT counted as VPIP (old bug: 1.0)")

    # Expected hand-1 memory (cumulative): A: 2 actions (raise+fold) ->
    # aggr .5, fold .5, vpip .5; B: vpip 1; C: vpip 0.
    mem1 = x1[IDX_MEM_AGGR:].tolist()
    print(f"      hand 1 memory features: {[round(v, 4) for v in mem1]}")
    check(abs(mem1[0] - 0.5 / 3) < 1e-6, "cumulative aggression carries hand 0")
    check(abs(mem1[1] - 0.5 / 3) < 1e-6, "tightness reflects A's fold")
    check(abs(mem1[2] - 1.5 / 3) < 1e-6, "cumulative VPIP across hands")

    # Direct unit check of the shared definition.
    m = OpponentMemory()
    for _ in range(4):
        m.observe("X", "check")
    check(m.features_for(["X"])[2] == 0.0,
          "OpponentMemory: 4 checks -> VPIP 0.0 (old inference counted 1.0)")
    m2 = OpponentMemory()
    m2.observe("Y", "call"); m2.observe("Y", "raise")
    m2.observe("Y", "bet"); m2.observe("Y", "fold")
    aggr, tight, vpip = m2.features_for(["Y"])
    check((aggr, tight, vpip) == (0.5, 0.25, 0.75),
          "OpponentMemory: call/raise/bet/fold -> (0.5, 0.25, 0.75)")
print()

# ═══════════════════════════════════════════════════════════════════════════
#  TEST 4: End-to-end — engine-logged session vs live MLBot features
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 4: End-to-end engine logs -> PokerDataset == live MLBot features")
print("=" * 70)


class RecordingHero:
    """Captures exactly what MLBot would feed its network at each act()."""

    def __init__(self):
        self.ml = fresh_bot()
        self.recorded = []
        self.positions = []

    def act(self, view):
        self.ml._update_memory(view)
        self.recorded.append(self.ml._make_features(view).squeeze(0).tolist())
        self.positions.append(view.position)
        types = {a["type"] for a in view.legal_actions}
        if "call" in types:
            return Action("call")
        if "check" in types:
            return Action("check")
        return Action("fold")


class Scripted:
    def __init__(self, style):
        self.style = style  # "raiser" | "caller" | "checkfolder"

    def act(self, view):
        types = {a["type"] for a in view.legal_actions}
        if self.style == "raiser":
            for t in ("raise", "bet"):
                if t in types:
                    spec = next(a for a in view.legal_actions if a["type"] == t)
                    return Action(t, spec["min"])
            if "call" in types:
                return Action("call")
        if self.style == "caller":
            if "call" in types:
                return Action("call")
        if "check" in types:
            return Action("check")
        return Action("fold")


with tempfile.TemporaryDirectory() as tmp:
    hero = RecordingHero()
    bots = {
        "HERO": hero,
        "RAISER": Scripted("raiser"),
        "CALLER": Scripted("caller"),
        "CFOLD": Scripted("checkfolder"),
    }
    seats = [Seat(pid, 500) for pid in bots]
    table = Table(rng=random.Random(42))
    session_logger = DecisionLogger(enabled=True, directory=tmp,
                                    session_scoped=True)

    n_hands = 4
    for h in range(n_hands):
        table.play_hand(
            seats, small_blind=5, big_blind=10,
            dealer_index=h % len(seats), bot_for=bots,
            logger=session_logger,
        )
    session_logger.close()

    ds = PokerDataset(tmp, filter_players=["HERO"])
    check(len(ds) == len(hero.recorded),
          f"sample count: dataset {len(ds)} == live decisions {len(hero.recorded)}")
    check(ds.missing_position_rows == 0,
          "engine logs carry position (no MP fallback rows)")

    all_match = len(ds) == len(hero.recorded)
    for i in range(min(len(ds), len(hero.recorded))):
        x, _ = ds[i]
        if not torch.allclose(x, torch.tensor(hero.recorded[i],
                                              dtype=torch.float32), atol=1e-6):
            diffs = [(j, a, b) for j, (a, b) in
                     enumerate(zip(x.tolist(), hero.recorded[i]))
                     if abs(a - b) > 1e-6]
            print(f"      decision {i} diff at: {diffs}")
            all_match = False
    check(all_match, f"all {len(hero.recorded)} decisions: training == inference, "
                     f"element-for-element")

    non_mp = [p for p in hero.positions if p != "MP"]
    check(len(set(hero.positions)) > 1 and non_mp,
          f"hero saw multiple real positions: {sorted(set(hero.positions))}")

    used_memory = any(abs(f - 0.5) > 1e-9
                      for x in [ds[i][0][IDX_MEM_AGGR:] for i in range(len(ds))]
                      for f in x.tolist())
    check(used_memory, "memory features moved off neutral 0.5 during the session")
print()

# ═══════════════════════════════════════════════════════════════════════════
#  TEST 5: Session safety — legacy logs rejected; dedup across sessions
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 5: Legacy-log rejection and cross-session memory identity")
print("=" * 70)

# 5a) Per-hand/legacy logs (no session header) are rejected by default.
with tempfile.TemporaryDirectory() as tmp:
    write_rows([make_row("HERO", AA, "BTN")], tmp, session_header=False)
    try:
        PokerDataset(tmp)
        check(False, "unmarked logs rejected by default (no ValueError raised)")
    except ValueError as e:
        check("session" in str(e).lower(),
              "unmarked logs rejected by default with actionable error")

    # 5b) Explicit opt-in loads them but flags the files.
    ds = PokerDataset(tmp, allow_unmarked_sessions=True)
    check(len(ds) == 1 and len(ds.unmarked_session_files) == 1,
          "allow_unmarked_sessions=True loads legacy file and flags it")

# 5c) Session-scoped logger output is accepted without flags.
with tempfile.TemporaryDirectory() as tmp:
    logger = DecisionLogger(enabled=True, directory=tmp, session_scoped=True)
    logger.start_hand(0)
    logger.log_decision(make_row("HERO", AA, "BTN"))
    logger.close()
    ds = PokerDataset(tmp)
    check(len(ds) == 1 and not ds.unmarked_session_files,
          "session-scoped logger output accepted (header present)")

# 5d) OpponentMemory: a new tournament reusing hand_id 0 must still count.
hist_t1 = [{"pid": "OPP", "type": "raise"}]
hist_t2 = [{"pid": "OPP", "type": "fold"}]

m = OpponentMemory()
m.observe_history(hist_t1, hand_id=0, session_id="T1")
m.observe_history(hist_t2, hand_id=0, session_id="T2")  # same hand_id+idx!
check(m.stats["OPP"]["actions"] == 2,
      "distinct session_id: reused (hand_id=0, idx=0) keys both observed")

m_bad = OpponentMemory()
m_bad.observe_history(hist_t1, hand_id=0)
m_bad.observe_history(hist_t2, hand_id=0)  # no session identity -> swallowed
check(m_bad.stats["OPP"]["actions"] == 1,
      "hazard confirmed: without session identity the new tournament is "
      "silently deduped (why reset/session_id is required)")

# 5e) MLBot.reset_memory() at the tournament boundary fixes 5d's hazard
#     and prevents stat leakage between tournaments.
bot = fresh_bot()
row_t1 = make_row("HERO", AA, "BTN", hand_id=0)
bot._update_memory(make_view(row_t1, history=hist_t1))
check(bot.memory.stats["OPP"]["aggressive"] == 1, "tournament 1 raise observed")

bot.reset_memory()
bot._update_memory(make_view(row_t1, history=hist_t2))  # T2 reuses hand_id 0
s = bot.memory.stats.get("OPP", {})
check(s.get("actions") == 1 and s.get("fold") == 1 and s.get("aggressive", 0) == 0,
      "after reset_memory(): new tournament's reused hand_id 0 is observed, "
      "old tournament's stats do not leak")
print()

# ═══════════════════════════════════════════════════════════════════════════
#  TEST 6: ML checkpoint feature-schema versioning
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 6: MLBot refuses legacy / wrong-schema checkpoints")
print("=" * 70)

from core.ml_features import FEATURE_DIM, FEATURE_SCHEMA_VERSION
from bots.poker_mlp import PokerMLP
from training.train_ml_bot import make_checkpoint

with tempfile.TemporaryDirectory() as tmp:
    sd = PokerMLP(input_dim=FEATURE_DIM, hidden=128, num_classes=6).state_dict()

    # 6a) Legacy raw state dict (the pre-Phase-2 save format) -> refused.
    legacy_path = os.path.join(tmp, "legacy.pt")
    torch.save(sd, legacy_path)
    b = MLBot(model_path=legacy_path, use_fallback=True)
    check(b.model_trained is False,
          "legacy raw state-dict checkpoint refused (model_trained=False)")

    # 6b) Wrong schema version -> refused.
    old_path = os.path.join(tmp, "old_schema.pt")
    torch.save({"state_dict": sd, "feature_schema_version": 1}, old_path)
    b = MLBot(model_path=old_path, use_fallback=True)
    check(b.model_trained is False,
          "feature_schema_version=1 checkpoint refused")

    # 6c) Current trainer format (make_checkpoint) -> loads.
    good_path = os.path.join(tmp, "current.pt")
    torch.save(make_checkpoint(sd), good_path)
    b = MLBot(model_path=good_path, use_fallback=True)
    check(b.model_trained is True,
          f"feature_schema_version={FEATURE_SCHEMA_VERSION} checkpoint loads")

    # 6d) Right version, wrong input dim -> refused.
    dim_path = os.path.join(tmp, "wrong_dim.pt")
    torch.save(make_checkpoint(
        PokerMLP(input_dim=23, hidden=128, num_classes=6).state_dict()), dim_path)
    b = MLBot(model_path=dim_path, use_fallback=True)
    check(b.model_trained is False, "versioned checkpoint with 23 inputs refused")

    # ---- 6e-6h) Malformed v2-marked checkpoints: refuse, never crash ----
    # 6e) Schema marker present but no state_dict at all.
    no_sd_path = os.path.join(tmp, "no_state_dict.pt")
    torch.save({"feature_schema_version": FEATURE_SCHEMA_VERSION}, no_sd_path)
    b = MLBot(model_path=no_sd_path, use_fallback=True)
    check(b.model_trained is False,
          "v2 checkpoint with missing state_dict refused (no crash)")

    # 6f) state_dict is not a dict.
    bad_type_path = os.path.join(tmp, "bad_type.pt")
    torch.save({"state_dict": "garbage",
                "feature_schema_version": FEATURE_SCHEMA_VERSION}, bad_type_path)
    b = MLBot(model_path=bad_type_path, use_fallback=True)
    check(b.model_trained is False,
          "v2 checkpoint with non-dict state_dict refused (no crash)")

    # 6g) state_dict is a dict but net.0.weight is not a tensor.
    junk_path = os.path.join(tmp, "junk_weights.pt")
    torch.save({"state_dict": {"net.0.weight": "junk"},
                "feature_schema_version": FEATURE_SCHEMA_VERSION}, junk_path)
    b = MLBot(model_path=junk_path, use_fallback=True)
    check(b.model_trained is False,
          "v2 checkpoint with non-tensor weights refused (no crash)")

    # 6h) Passes the first-layer check but is corrupt beyond it (missing
    #     key) -> load_state_dict fails -> refused, model rebuilt, and the
    #     bot still acts via fallback.
    corrupt_sd = {k: v for k, v in sd.items() if k != "net.4.weight"}
    corrupt_path = os.path.join(tmp, "corrupt.pt")
    torch.save(make_checkpoint(corrupt_sd), corrupt_path)
    b = MLBot(model_path=corrupt_path, use_fallback=True)
    check(b.model_trained is False,
          "v2 checkpoint with missing layer key refused (load failure caught)")
    action = b.act(make_view(make_row("HERO", AA, "BTN")))
    check(action is not None and action.type in ("fold", "check", "call",
                                                 "raise", "bet"),
          "bot still acts via fallback after refusing corrupt checkpoint")

    # ---- 6i-6j) Phase 2.1c: unreadable files fall back, never crash ----
    # 6i) Empty checkpoint file (EOFError from torch.load).
    empty_path = os.path.join(tmp, "empty.pt")
    open(empty_path, "wb").close()
    b = MLBot(model_path=empty_path, use_fallback=True)
    check(b.model_trained is False,
          "empty checkpoint file falls back (no crash)")

    # 6j) Random bytes / bad pickle (pickle.UnpicklingError).
    bytes_path = os.path.join(tmp, "random_bytes.pt")
    with open(bytes_path, "wb") as f:
        f.write(b"\x00\xffnot a checkpoint\x13\x37" * 64)
    b = MLBot(model_path=bytes_path, use_fallback=True)
    check(b.model_trained is False,
          "random-byte / bad-pickle checkpoint falls back (no crash)")
print()

# ═══════════════════════════════════════════════════════════════════════════
#  TEST 7: Strict session-header validation (Phase 2.1b)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 7: Strict session-header validation")
print("=" * 70)


def write_raw(lines, tmpdir, fname="session_synthetic_0.jsonl"):
    path = os.path.join(tmpdir, fname)
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")
    return path


def header_line(session_id="synthetic", version=2, payload=None):
    if payload is None:
        payload = {"session_id": session_id, "log_format_version": version}
    return json.dumps({"session_start": payload})


DECISION = json.dumps(make_row("HERO", AA, "BTN"))


def expect_invalid(lines, label):
    """Invalid headers must raise — even with allow_unmarked_sessions=True."""
    for allow in (False, True):
        with tempfile.TemporaryDirectory() as tmp:
            write_raw(lines, tmp)
            try:
                PokerDataset(tmp, allow_unmarked_sessions=allow)
                check(False, f"{label} [allow={allow}] (no ValueError raised)")
            except ValueError as e:
                check("header" in str(e).lower(),
                      f"{label} [allow_unmarked_sessions={allow}]")


# 7a) Control: well-formed header as the first row -> loads cleanly.
with tempfile.TemporaryDirectory() as tmp:
    write_raw([header_line(), DECISION], tmp)
    ds = PokerDataset(tmp)
    check(len(ds) == 1 and not ds.invalid_session_files,
          "valid header (first row, session_id, version 2) loads")

# 7b) Late header: a decision row precedes session_start.
expect_invalid([DECISION, header_line()], "late header rejected")

# 7c) Duplicate header: a second session_start later in the file.
expect_invalid([header_line(), DECISION, header_line(session_id="other")],
               "duplicate session_start rejected")

# 7d) Empty / missing session_id.
expect_invalid([header_line(session_id=""), DECISION],
               "empty session_id rejected")
expect_invalid([header_line(payload={"log_format_version": 2}), DECISION],
               "missing session_id rejected")

# 7e) Wrong / missing log_format_version.
expect_invalid([header_line(version=1), DECISION],
               "log_format_version=1 rejected")
expect_invalid([header_line(payload={"session_id": "s"}), DECISION],
               "missing log_format_version rejected")

# 7f) Malformed payload: session_start is not an object.
expect_invalid([json.dumps({"session_start": "yes"}), DECISION],
               "non-object session_start payload rejected")

# 7g) One valid file does NOT mask a corrupt sibling.
with tempfile.TemporaryDirectory() as tmp:
    write_raw([header_line(), DECISION], tmp, fname="good.jsonl")
    write_raw([DECISION, header_line()], tmp, fname="bad.jsonl")
    try:
        PokerDataset(tmp)
        check(False, "corrupt file beside a valid one still raises "
                     "(no ValueError raised)")
    except ValueError as e:
        check("bad.jsonl" in str(e),
              "corrupt file beside a valid one still raises, names the file")

# ---- 7h-7k) Phase 2.1c: malformed rows are hard errors, never noise ----

# 7h) Invalid JSON row inside an otherwise-valid session file.
expect_invalid([header_line(), DECISION, '{"chosen_action": '],
               "invalid JSON row rejected")

# 7i) Top-level non-object rows (null / list / string / number).
expect_invalid([header_line(), "null", DECISION],
               "top-level null row rejected")
expect_invalid([header_line(), DECISION, json.dumps([1, 2])],
               "top-level list row rejected")
expect_invalid([header_line(), '"a string"', DECISION],
               "top-level string row rejected")
expect_invalid([header_line(), "42", DECISION],
               "top-level number row rejected")

# 7j) THE merge hazard this closes: two concatenated sessions where the
#     second header line got damaged. If corrupt rows were skipped as
#     noise, both tournaments would silently merge into one
#     cumulative-memory replay.
truncated_second_header = '{"session_start": {"session_id": "T2", '
expect_invalid([header_line(session_id="T1"), DECISION,
                truncated_second_header, DECISION],
               "corrupted second header cannot merge two sessions")

# 7k) Corrupt rows are NOT 'legacy': a header-less file with a damaged
#     row is a hard error, not opt-in loadable via allow_unmarked_sessions.
expect_invalid([DECISION, "not json at all"],
               "header-less file with corrupt row is a hard error, not legacy")
print()

# ═══════════════════════════════════════════════════════════════════════════
#  TEST 8: UI restart — adapters forward reset_memory(); runner is wired
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 8: UI restart memory reset (adapter forwarding + runner wiring)")
print("=" * 70)

from bots import _wrap, create_bot

# 8a) _PlayerViewAdapter forwards reset_memory to the wrapped MLBot.
adapter = _wrap(fresh_bot())
adapter.bot._update_memory(make_view(make_row("HERO", AA, "BTN", hand_id=0),
                                     history=hist_t1))
check(adapter.bot.memory.stats["OPP"]["aggressive"] == 1,
      "tournament 1: adapter-wrapped MLBot observed the raise")

# 8b) The exact duck-typed loop run_tournament.py uses — must tolerate
#     adapters without reset_memory (e.g. InProcessBot for 'random').
bots_map = {"P1": adapter, "P2": create_bot("random")}
for a in bots_map.values():
    reset = getattr(a, "reset_memory", None)
    if callable(reset):
        reset()
check(adapter.bot.memory.stats == {},
      "reset loop cleared MLBot memory through the adapter")
# Reaching this line at all means the duck-typed loop tolerated the
# 'random' adapter (InProcessBot), whether or not it has reset_memory.
check(True, "reset loop tolerated all adapter types (no crash)")

# 8c) After the reset, a new tournament reusing hand_id 0 is observed.
adapter.bot._update_memory(make_view(make_row("HERO", AA, "BTN", hand_id=0),
                                     history=hist_t2))
s = adapter.bot.memory.stats.get("OPP", {})
check(s.get("actions") == 1 and s.get("fold") == 1
      and s.get("aggressive", 0) == 0,
      "post-reset: restarted hand_id 0 observed, no stat leakage")

# 8d) Wiring smoke check: run_tournament._run_tournament calls reset_memory
#     at tournament start. (Source-level check — importing run_tournament
#     would pull in an interactive matplotlib backend.)
_rt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "run_tournament.py")
with open(_rt_path) as f:
    _rt_src = f.read()
_body = _rt_src.split("def _run_tournament", 1)[1].split("\n    def ", 1)[0]
check("reset_memory" in _body,
      "run_tournament._run_tournament resets bot memory at tournament start")
print()

# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
if PASS:
    print("OVERALL: ALL CHECKS PASSED [PASS]")
else:
    print("OVERALL: SOME CHECKS FAILED [FAIL]")
print("=" * 70)
sys.exit(0 if PASS else 1)
