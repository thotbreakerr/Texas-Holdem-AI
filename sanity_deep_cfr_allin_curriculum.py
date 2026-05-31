"""
sanity_deep_cfr_allin_curriculum.py — Key Change #2 regressions.

Proves the all-in regret-row poisoning fix in `_cfr_recurse`: during training,
the hero now expands and collects regret targets over EXACTLY the policy legal
mask, never the raw target mask.  Concretely:

  CHECK 1  Shadow phase (all_in_policy_probability=0.0): `all_in` is excluded
           from BOTH the policy mask AND the regret target mask whenever a
           non-all-in action is legal, and the four masks the traversal uses —
           policy_legal_mask, hero expansion, EV sum, collected regret mask —
           are identical.  (This is the discriminating proof.  Pre-fix, the
           regret target mask included `all_in` with EV taken over a mask that
           excluded it, which is precisely what poisoned the all_in row.)

  CHECK 2  Full phase (all_in_policy_probability=1.0): `all_in` IS included in
           the policy mask and the regret target mask when legal; masks aligned.

  CHECK 3  No all-in sizing target is ever written to the sizing buffer — proven
           on a short-stack node whose ONLY aggressive action is `all_in` (no
           bet bucket legal): even with `all_in` expanded in the full phase, the
           sizing buffer stays empty.  A normal node confirms bucket sizing is
           still collected (fracs ∈ {0.33,0.50,0.67,0.75,1.00}).

  CHECK 4  Inference is preserved: act_with_trace() on the OLD collapsed
           checkpoint still surfaces `all_in`-row decisions (the training-target
           change does not touch the inference path).  Skipped (non-fatal) if
           the checkpoint is absent.

  CHECK 5  Micro-training smoke guard (coarse, NOT a strength eval): a tiny new
           run through shadow→staged→full, probed RAW (guardrails disabled), and
           asserted not to explode into an all-in collapse on the metrics that
           discriminate at smoke scale — preflop all-in %, avg raise size, and
           strong-hand all-in %.  Preflop PFR is reported but NOT hard-gated here
           (a healthy micro net saturates PFR at ~85-100%, indistinguishable from
           a collapse); PFR is enforced in the live training canary instead.  A
           few-minute run cannot distinguish a well-trained net from a poorly-
           trained one — the 410k collapse took 410k iterations — so this only
           catches gross regressions; CHECK 1 carries the real proof.
"""
import io
import os
import random
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, "/Users/jaroslavaupart/Desktop/Projects/Texas-Holdem-AI")

import torch

from bots.deep_cfr_bot import (
    DeepCFRBot, DeepCFRConfig, _DeepCFRGameState,
    ReservoirBuffer, ABSTRACT_ACTIONS, NUM_ACTIONS,
)
from training.train_deep_cfr import run_training, parse_args
from probe_deep_cfr import (
    _view, _random_hole, _new_stats, _record, _pct, _avg_size, POSITIONS,
)

ALL_IN = ABSTRACT_ACTIONS.index("all_in")
FOLD = ABSTRACT_ACTIONS.index("fold")
CHECK_CALL = ABSTRACT_ACTIONS.index("check_call")
OLD_COLLAPSED_CKPT = "models/deep_cfr_large_clean_1m.warn_410000.pt"


def _make_bot(aivat_sims=1):
    with redirect_stdout(io.StringIO()):
        bot = DeepCFRBot(config=DeepCFRConfig.small(), inference_mode=False,
                         aivat_sims=aivat_sims)
    # Deterministic, fast leaves — we are testing mask plumbing, not values.
    bot._aivat_leaf_value = lambda _state, _hero_seat: 0.0
    return bot


def _preflop_state():
    """HU preflop, hero (seat 0) to act facing the BB — all_in is legal."""
    return _DeepCFRGameState(
        pot=15,
        stacks=[1000, 1000],
        committed_per_seat=[0, 10],
        alive=[True, True],
        street="preflop",
        board=[],
        hole_cards={0: (("A", "s"), ("K", "s")), 1: (("2", "c"), ("7", "d"))},
        seat_order=[0],
        action_idx=0,
        history_events=[],
        deck_remaining=[],
        big_blind=10,
        ring_order=[0, 1],
    )


def _allin_only_state():
    """Short-stack hero facing a bet where the ONLY legal raise is a shove.

    current_bet=50, hero committed 10 (to_call=40), hero stack 60 → max_total=70
    which is < min_total (50+last_raise_size=90) but > current_bet, so
    legal_actions() emits a {raise, all_in=True, min==max==70} spec and
    _legal_abstract_actions() returns {fold, check_call, all_in} with NO bet
    bucket.
    """
    return _DeepCFRGameState(
        pot=60,
        stacks=[60, 940],
        committed_per_seat=[10, 50],
        alive=[True, True],
        street="preflop",
        board=[],
        hole_cards={0: (("A", "s"), ("K", "s")), 1: (("2", "c"), ("7", "d"))},
        seat_order=[0],
        action_idx=0,
        history_events=[],
        deck_remaining=[],
        big_blind=10,
        ring_order=[0, 1],
        last_raise_size=40,
    )


def _collect_root(bot, state, all_in_policy_probability):
    """Run one depth-1 hero traversal; return root policy mask + buffers.

    Spies _regret_match to capture the policy_legal_mask actually used at the
    root hero node (the same mask the strategy is normalized over and the hero
    expands).  The regret buffer's legal_mask_vec is the collected regret mask.
    """
    seen_masks = []
    original = bot._regret_match

    def spy(logits, legal_mask):
        seen_masks.append(list(legal_mask))
        return original(logits, legal_mask)

    bot._regret_match = spy
    r_buf, v_buf, s_buf = ReservoirBuffer(), ReservoirBuffer(), ReservoirBuffer()
    bot._cfr_recurse(
        state, hero_seat=0, depth=1, iteration=1,
        regret_buf=r_buf, value_buf=v_buf, sizing_buf=s_buf,
        all_in_policy_probability=all_in_policy_probability,
    )
    bot._regret_match = original
    policy_mask = set(seen_masks[0]) if seen_masks else set()
    regret_mask = {i for i in range(NUM_ACTIONS)
                   if float(r_buf.buffer[0][2][i].item()) == 1.0} if r_buf.buffer else set()
    return policy_mask, regret_mask, s_buf


# ── CHECK 1: shadow excludes all_in from regret targets; masks aligned ───────

def check_shadow(pass_state):
    bot = _make_bot()
    policy_mask, regret_mask, _ = _collect_root(bot, _preflop_state(), 0.0)
    excluded = ALL_IN not in policy_mask and ALL_IN not in regret_mask
    keeps_normal = {FOLD, CHECK_CALL} <= regret_mask
    aligned = policy_mask == regret_mask and len(policy_mask) > 0
    ok = excluded and keeps_normal and aligned
    pass_state[0] &= ok
    labels = lambda s: [ABSTRACT_ACTIONS[i] for i in sorted(s)]
    print(f"[CHECK 1] {'PASS' if ok else 'FAIL'} — shadow phase: "
          f"policy={labels(policy_mask)} regret={labels(regret_mask)}; "
          f"all_in excluded={excluded}, normal kept={keeps_normal}, "
          f"masks aligned={aligned}")


# ── CHECK 2: full phase includes all_in; masks aligned ───────────────────────

def check_full(pass_state):
    bot = _make_bot()
    policy_mask, regret_mask, _ = _collect_root(bot, _preflop_state(), 1.0)
    included = ALL_IN in policy_mask and ALL_IN in regret_mask
    aligned = policy_mask == regret_mask and len(policy_mask) > 0
    ok = included and aligned
    pass_state[0] &= ok
    labels = lambda s: [ABSTRACT_ACTIONS[i] for i in sorted(s)]
    print(f"[CHECK 2] {'PASS' if ok else 'FAIL'} — full phase: "
          f"policy={labels(policy_mask)} regret={labels(regret_mask)}; "
          f"all_in included={included}, masks aligned={aligned}")


# ── CHECK 3: no all_in sizing target; bucket sizing still collected ──────────

def check_sizing(pass_state):
    _BUCKET_FRACS = {0.33, 0.50, 0.67, 0.75, 1.00}

    # Decisive: all_in expanded (full phase) but no bet bucket legal → no sizing.
    bot = _make_bot()
    policy_mask, _, s_buf = _collect_root(bot, _allin_only_state(), 1.0)
    allin_expanded = ALL_IN in policy_mask
    no_bucket_legal = not ({2, 3, 4, 5, 6} & policy_mask)
    no_sizing = len(s_buf) == 0
    decisive_ok = allin_expanded and no_bucket_legal and no_sizing

    # Positive control: a normal node still collects bucket sizing.
    bot2 = _make_bot()
    _, _, s_buf2 = _collect_root(bot2, _preflop_state(), 1.0)
    fracs = [round(float(f), 2) for _inp, f in s_buf2.buffer]
    bucket_only = bool(fracs) and all(f in _BUCKET_FRACS for f in fracs)

    ok = decisive_ok and bucket_only
    pass_state[0] &= ok
    print(f"[CHECK 3] {'PASS' if ok else 'FAIL'} — all_in expanded but no bet "
          f"bucket → sizing entries={len(s_buf)} (expect 0); normal node "
          f"sizing fracs={fracs} all bucket-valued={bucket_only}")


# ── CHECK 4: inference preserved on the old collapsed checkpoint ──────────────

def check_inference_preserved(pass_state):
    if not os.path.exists(OLD_COLLAPSED_CKPT):
        print(f"[CHECK 4] SKIP — {OLD_COLLAPSED_CKPT} absent (non-fatal); "
              "inference-path preservation not exercised here")
        return
    with redirect_stdout(io.StringIO()):
        bot = DeepCFRBot(weights_path=OLD_COLLAPSED_CKPT, inference_mode=True)
    bot.search_depth = 0
    strong = [[("A", "h"), ("A", "s")], [("K", "h"), ("K", "s")],
              [("A", "h"), ("K", "h")]]
    saw_all_in_row = False
    saw_effective_all_in = False
    for i, hole in enumerate(strong * 4):
        view = _view(hole, position="BTN", pot=15, to_call=10, stack=1000)
        _action, trace = bot.act_with_trace(view)
        if trace.selected_abstract_label == "all_in":
            saw_all_in_row = True
        if trace.final_is_all_in:
            saw_effective_all_in = True
    ok = saw_all_in_row and saw_effective_all_in
    pass_state[0] &= ok
    print(f"[CHECK 4] {'PASS' if ok else 'FAIL'} — old collapsed checkpoint via "
          f"act_with_trace(): all_in-row selected={saw_all_in_row}, "
          f"effective all-in={saw_effective_all_in} (inference path unchanged)")


# ── CHECK 5: micro-training smoke guard (coarse) ─────────────────────────────

def check_micro_training(pass_state):
    # Coarse "doesn't explode" bounds on the metrics that ACTUALLY discriminate a
    # healthy micro run from a collapsed one at smoke scale.  Set with generous
    # margin OVER a healthy seed-42 micro net (preflop all-in ~0-5%, avg raise
    # ~4-6x, strong all-in ~0%) and UNDER the known collapse signature (preflop
    # all-in ~28%, avg raise ~37x, strong all-in ~54%).  Probed RAW (guardrails
    # off), so occasional all-in shoves (~66x pot) dominate avg-raise — it tracks
    # shove rate and is largely redundant with all-in%, so a loose bound hides
    # nothing real (CHECK 3 proved all-in writes no sizing target; Key Change #1
    # ruled out bucket->max preflop).
    #
    # PFR is REPORTED but intentionally NOT a hard micro bound.  Empirically a
    # HEALTHY micro net's preflop PFR saturates at ~85-100% (measured across
    # iters 80-600, aivat_sims 1-50, multiple seeds), and a genuinely collapsed
    # micro run shows PFR=100% too — so PFR cannot separate healthy from collapsed
    # at this scale (a tiny net "raises everything" because betting wins pots
    # cheaply in self-play long before it learns to fold).  A 70% micro FAIL bound
    # would therefore fail healthy runs — a meaningless gate.  PFR IS enforced
    # where it becomes meaningful: the LIVE training canary, and only once the net
    # is mature (iteration >= all_in_deploy_iteration), via
    # train_deep_cfr.decide_canary_status / classify_extra_canary_metrics
    # (WARN >= 40%, FAIL >= 55%), regression-tested in sanity_train_deep_cfr.py
    # Section 13.
    BOUND_PREFLOP_ALL_IN = 22.0
    BOUND_PREFLOP_AVG_RAISE = 25.0
    BOUND_STRONG_ALL_IN = 40.0
    SAMPLES = 120
    SEED = 42

    # Seed module random + torch BEFORE run_training: the bot's per-instance
    # _rng is seeded from random.getrandbits(64) at construction, so module
    # seeding propagates into training, making this run reproducible.  Generous
    # bounds remain the primary robustness mechanism (cross-machine torch).
    random.seed(SEED)
    torch.manual_seed(SEED)
    try:
        import numpy as _np
        _np.random.seed(SEED)
    except Exception:
        pass

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "micro.pt")
        with redirect_stdout(io.StringIO()):
            margs = parse_args([
                "--variant", "small",
                "--iterations", "80",
                "--update-interval", "10",
                "--checkpoint-interval", "80",
                "--batch-size", "32",
                "--aivat-sims", "1",
                "--all-in-warmup-iterations", "20",
                "--all-in-deploy-iteration", "40",
                "--all-in-full-release-iteration", "60",
                "--save-path", ckpt,
                "--device", "cpu",
                "--disable-collapse-canary",
            ])
            run_training(margs)

        if not os.path.exists(ckpt):
            pass_state[0] = False
            print("[CHECK 5] FAIL — micro-training produced no checkpoint")
            return

        with redirect_stdout(io.StringIO()):
            bot = DeepCFRBot(weights_path=ckpt, inference_mode=True)
        bot.search_depth = 0
        # Probe the RAW network: disable guardrails so this tests the training
        # curriculum, not the inference mask (Section 15 of sanity_train_deep_cfr
        # already covers the mask).
        bot._all_in_guardrails_disabled = True

        random.seed(SEED)
        pre = _new_stats()
        for _ in range(SAMPLES):
            hole = _random_hole()
            view = _view(hole, position=random.choice(POSITIONS),
                         pot=15, to_call=10, stack=1000)
            _record(pre, bot.act(view), view)

        strong_hands = [[("A", "h"), ("A", "s")], [("K", "h"), ("K", "s")],
                        [("A", "h"), ("K", "h")]]
        strong = _new_stats()
        for i in range(30):
            view = _view(strong_hands[i % 3], position=random.choice(["CO", "BTN"]),
                         pot=15, to_call=10, stack=1000)
            _record(strong, bot.act(view), view)

    pre_all_in = _pct(pre["all_in"], pre["total"])
    pre_pfr = _pct(pre["pfr"], pre["total"])
    pre_avg = _avg_size(pre)
    strong_all_in = _pct(strong["all_in"], strong["total"])

    not_exploded = (
        pre_all_in < BOUND_PREFLOP_ALL_IN
        and pre_avg < BOUND_PREFLOP_AVG_RAISE
        and strong_all_in < BOUND_STRONG_ALL_IN
    )
    pass_state[0] &= not_exploded
    print(f"[CHECK 5] {'PASS' if not_exploded else 'FAIL'} — micro-train smoke "
          f"(coarse): preflop all-in={pre_all_in:.1f}% (<{BOUND_PREFLOP_ALL_IN}), "
          f"PFR={pre_pfr:.1f}% (reported only; saturates at smoke scale, gated in "
          f"the live canary), avg raise={pre_avg:.2f}x "
          f"(<{BOUND_PREFLOP_AVG_RAISE}), strong all-in={strong_all_in:.1f}% "
          f"(<{BOUND_STRONG_ALL_IN})")


def run(mode="all"):
    """Run the Key Change #2 regressions.

    mode="all"   — every check (default; the standalone invocation).
    mode="fast"  — CHECK 1-4 only: mask plumbing + inference preservation, no
                   training.  This is the fast Tier 3 ladder gate.
    mode="micro" — CHECK 5 only: the micro-training smoke guard.  This is the
                   slow Tier 5 (--full) ladder gate.
    """
    pass_state = [True]
    if mode in ("all", "fast"):
        check_shadow(pass_state)
        check_full(pass_state)
        check_sizing(pass_state)
        check_inference_preserved(pass_state)
    if mode in ("all", "micro"):
        check_micro_training(pass_state)
    print("=" * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED [PASS]' if pass_state[0] else 'SOME CHECKS FAILED [FAIL]'}")
    return pass_state[0]


if __name__ == "__main__":
    # NB: a local argparse (not a parse_args() function) — the module imports
    # train_deep_cfr.parse_args for the micro-training run, so we must not
    # shadow that name here.
    import argparse
    _parser = argparse.ArgumentParser(
        description="Key Change #2 (all-in regret-mask) regressions.")
    _parser.add_argument(
        "--mode", choices=["all", "fast", "micro"], default="all",
        help="all: every check (default). fast: mask/inference checks only "
             "(CHECK 1-4, no training). micro: micro-training smoke only "
             "(CHECK 5, slow).")
    _args = _parser.parse_args()
    sys.exit(0 if run(_args.mode) else 1)
