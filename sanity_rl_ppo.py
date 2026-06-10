"""
sanity_rl_ppo.py -- regression probes for RLBot PPO correctness (Phase 1/1.1).

These checks encode the PPO data-collection contracts:

Sections:
  1.  Legal-action mask matches the legal action set.
  2&3. Illegal actions never sampled; stored action == executed action.
  4.  A hand's reward is assigned exactly once (no duplication across steps).
  5.  Terminal bonus reaches the episode's final transition.
  6.  The final (un-ended) episode is included before the buffer flush.
  7.  Epsilon-greedy exploration is gone from act().
  8.  PPO ratio is exactly ~1.0 for stored actions with unchanged weights
      (networks are deterministic — would fail if dropout returned).
  9.  Fail-closed: empty legal mask raises during training collection;
      missing stored mask raises during the PPO update; eval stays safe.
  10. Fallback conversion cannot silently change the stored action.
  11. No-decision hand chip delta attaches to the most recent transition;
      with no transitions at all in the episode it is explicitly dropped.
  12. Raise-bucket aliasing contract: indices 3/4/5 share type bet/raise
      and differ only in amount (min / midpoint / max).
  13. Masked PPO update runs end-to-end and keeps parameters finite.
"""
from __future__ import annotations

import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from bots.rl_bot import RLBot
from core.bot_api import Action, PlayerView


FAILURES: list[str] = []


def check(name: str, condition: bool, details: str = "") -> None:
    if condition:
        print(f"  PASS - {name}")
    else:
        print(f"  FAIL - {name}: {details}")
        FAILURES.append(f"{name}: {details}")


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_view(legal, street="flop", pot=20, to_call=0):
    """Minimal PlayerView with a controlled legal-action set."""
    return PlayerView(
        me="P2",
        street=street,
        position="BTN",
        hole_cards=[("A", "s"), ("K", "s")],
        board=[("2", "h"), ("7", "d"), ("J", "c")] if street != "preflop" else [],
        pot=pot,
        to_call=to_call,
        min_raise=4,
        max_raise=100,
        legal_actions=legal,
        stacks={"P1": 100, "P2": 100},
        opponents=["P1"],
        history=[],
    )


def make_training_bot():
    return RLBot(model_path="", training_mode=True, use_fallback=False,
                 starting_chips=100)


# Engine-realistic legal-action scenarios.
SCENARIOS = {
    # to_call == 0, no bet yet → check or open-bet
    "open_bet": [
        {"type": "check"},
        {"type": "bet", "min": 2, "max": 100},
    ],
    # facing a bet → fold / call / raise
    "facing_bet": [
        {"type": "fold"},
        {"type": "call"},
        {"type": "raise", "min": 8, "max": 100},
    ],
    # facing an all-in we can only call → fold / call (no raise buckets)
    "fold_call_only": [
        {"type": "fold"},
        {"type": "call"},
    ],
    # degenerate: check only
    "check_only": [
        {"type": "check"},
    ],
}

# Action-head indices that are legal in each scenario
# (0=fold, 1=check, 2=call, 3/4/5=raise buckets).
EXPECTED_LEGAL_IDX = {
    "open_bet":       {1, 3, 4, 5},
    "facing_bet":     {0, 2, 3, 4, 5},
    "fold_call_only": {0, 2},
    "check_only":     {1},
}

# Which executed action types each stored index may map to.
IDX_TO_TYPES = {
    0: {"fold"}, 1: {"check"}, 2: {"call"},
    3: {"bet", "raise"}, 4: {"bet", "raise"}, 5: {"bet", "raise"},
}

N_SAMPLES = 250  # per scenario; a fresh random policy would hit illegal
                 # actions many times in 250 unmasked samples


def view_for(scenario_name):
    legal = SCENARIOS[scenario_name]
    legal_types = {a["type"] for a in legal}
    return make_view(legal, to_call=0 if "check" in legal_types else 10)


def main() -> int:
    torch.manual_seed(0)

    # ── 1. Mask construction ──────────────────────────────────────────────
    section("1. Legal-action mask matches the legal action set")
    bot = make_training_bot()
    for name, legal in SCENARIOS.items():
        mask = bot._legal_action_mask(legal)
        got = {i for i in range(6) if bool(mask[0, i])}
        check(f"mask[{name}]", got == EXPECTED_LEGAL_IDX[name],
              f"expected {sorted(EXPECTED_LEGAL_IDX[name])}, got {sorted(got)}")

    # ── 2 & 3. Sampling: never illegal, stored == executed ────────────────
    section("2&3. Illegal actions never sampled; stored action == executed")
    for name, legal in SCENARIOS.items():
        bot = make_training_bot()
        legal_types = {a["type"] for a in legal}
        view = view_for(name)

        illegal_sampled = 0
        mismatches = 0
        missing_steps = 0
        for _ in range(N_SAMPLES):
            n_before = len(bot.current_episode)
            action = bot.act(view)
            if len(bot.current_episode) != n_before + 1:
                missing_steps += 1
                continue
            step = bot.current_episode[-1]
            idx = step["action"]
            if idx not in EXPECTED_LEGAL_IDX[name]:
                illegal_sampled += 1
            if action.type not in IDX_TO_TYPES[idx]:
                mismatches += 1
            if action.type not in legal_types:
                illegal_sampled += 1

        check(f"step stored per act() [{name}]", missing_steps == 0,
              f"{missing_steps}/{N_SAMPLES} act() calls stored no step")
        check(f"no illegal samples [{name}]", illegal_sampled == 0,
              f"{illegal_sampled}/{N_SAMPLES} illegal actions sampled/executed")
        check(f"stored == executed [{name}]", mismatches == 0,
              f"{mismatches}/{N_SAMPLES} stored actions differ from executed")

    # ── 4. Hand reward assigned exactly once ──────────────────────────────
    section("4. Hand reward is not duplicated across decisions")
    bot = make_training_bot()
    view = view_for("facing_bet")

    for _ in range(3):
        bot.act(view)
    bot.record_reward(0.5)

    rewards = [s.get("reward") for s in bot.current_episode]
    check("hand 1 reward on final step only", rewards == [0.0, 0.0, 0.5],
          f"got {rewards}")
    check("hand 1 total counted once", abs(sum(rewards) - 0.5) < 1e-9,
          f"sum={sum(rewards)} (duplicated reward would be 1.5)")

    for _ in range(2):
        bot.act(view)
    bot.record_reward(-0.25)

    rewards = [s.get("reward") for s in bot.current_episode]
    check("hand 2 doesn't overwrite hand 1",
          rewards == [0.0, 0.0, 0.5, 0.0, -0.25], f"got {rewards}")

    # ── 5. Terminal bonus reaches the final transition ────────────────────
    section("5. Terminal bonus changes the final transition")
    before = [s.get("reward") for s in bot.current_episode]
    bot.record_terminal_bonus(1.0)
    after = [s.get("reward") for s in bot.current_episode]

    changed = [i for i, (b, a) in enumerate(zip(before, after)) if b != a]
    check("exactly one transition changed", changed == [len(before) - 1],
          f"changed indices: {changed}")
    check("bonus added to final reward",
          abs(after[-1] - (before[-1] + 1.0)) < 1e-9,
          f"before={before[-1]}, after={after[-1]}")

    # ── 6. Final episode included before flush ────────────────────────────
    section("6. Final (un-ended) episode is included before buffer flush")
    bot = make_training_bot()
    bot.act(view)
    bot.record_reward(0.1)

    captured: list[list[int]] = []
    bot._ppo_update = lambda episodes: captured.append(
        [len(ep) for ep in episodes]
    )
    bot.flush_buffer()   # note: end_episode() was NOT called by the caller

    check("flush ran exactly one update", len(captured) == 1,
          f"updates run: {len(captured)}")
    check("final episode reached the update",
          bool(captured) and captured[0] == [1],
          f"episode step-counts seen by update: {captured}")
    check("episode state reset after flush",
          bot.current_episode == [] and bot.episode_buffer == [],
          f"current={len(bot.current_episode)} buffer={len(bot.episode_buffer)}")

    # ── 7. Epsilon-greedy removed ──────────────────────────────────────────
    section("7. Epsilon-greedy exploration removed from act()")
    src = inspect.getsource(RLBot.act)
    check("act() has no epsilon-greedy branch",
          "random.randint" not in src and "random.random" not in src,
          "act() still references random exploration")

    # ── 8. PPO ratio == 1.0 with unchanged weights ─────────────────────────
    section("8. PPO ratio for stored actions is ~1.0 with unchanged weights")
    bot = make_training_bot()
    for scenario in ("open_bet", "facing_bet", "fold_call_only", "check_only"):
        v = view_for(scenario)
        for _ in range(5):
            bot.act(v)

    steps = bot.current_episode
    states  = torch.cat([s["state"] for s in steps], dim=0)
    masks   = torch.cat([s["mask"] for s in steps], dim=0)
    actions = torch.tensor([s["action"] for s in steps], dtype=torch.long)
    old_lp  = torch.stack([s["log_prob"] for s in steps]).reshape(-1)

    # Recompute through the exact path _ppo_update uses, weights unchanged.
    with torch.no_grad():
        new_lp = bot._masked_policy_dist(states, masks).log_prob(actions)
    ratio = torch.exp(new_lp - old_lp)
    max_dev = (ratio - 1.0).abs().max().item()
    check("recomputed ratio == 1.0", max_dev < 1e-4,
          f"max |ratio - 1| = {max_dev:.3e} — rollout and update "
          f"distributions disagree (dropout/nondeterminism?)")

    # Networks must contain no dropout layers at all.
    has_dropout = any(
        isinstance(m, torch.nn.Dropout)
        for net in (bot.policy_net, bot.value_net)
        for m in net.modules()
    )
    check("no dropout layers in policy/value nets", not has_dropout,
          "nn.Dropout found — PPO ratios will not reproduce")

    # ── 9. Fail-closed on bad masks ────────────────────────────────────────
    section("9. Fail-closed: empty mask / missing stored mask raise in training")
    bot = make_training_bot()

    raised = False
    try:
        bot.act(make_view([]))   # no legal actions at all
    except ValueError:
        raised = True
    check("empty legal mask raises during training collection", raised,
          "act() accepted an empty legal-action set in training mode")
    check("no step stored for the rejected state",
          len(bot.current_episode) == 0,
          f"{len(bot.current_episode)} step(s) stored from a failed act()")

    bot = make_training_bot()
    bot.act(view_for("facing_bet"))
    bot.record_reward(0.1)
    del bot.current_episode[0]["mask"]    # simulate stale/corrupt trajectory
    raised = False
    try:
        bot.flush_buffer()
    except ValueError:
        raised = True
    check("missing stored mask raises during PPO update", raised,
          "update silently assumed all actions were legal")

    # Eval mode stays permissive: unknown-only legal set must not crash.
    eval_bot = RLBot(model_path="", training_mode=False, use_fallback=False,
                     starting_chips=100)
    try:
        act = eval_bot.act(make_view([{"type": "weird", "min": 5, "max": 10}]))
        check("eval mode stays safe on unknown legal types",
              isinstance(act, Action), f"got {act!r}")
    except Exception as e:  # noqa: BLE001
        check("eval mode stays safe on unknown legal types", False, repr(e))

    # ── 10. Fallback conversion cannot silently change the action ─────────
    section("10. Fallback conversion cannot silently change the stored action")
    bot = make_training_bot()
    # Force every conversion to produce "call" while call is illegal
    # (open_bet scenario) — act() must refuse to store the mismatch.
    bot._action_idx_to_action = lambda idx, legal: Action("call")
    raised = False
    try:
        bot.act(view_for("open_bet"))
    except RuntimeError:
        raised = True
    check("mismatched conversion raises in training mode", raised,
          "act() silently stored an action that differs from the executed one")
    check("no step stored for the mismatch",
          len(bot.current_episode) == 0,
          f"{len(bot.current_episode)} step(s) stored despite the mismatch")

    # ── 11. No-decision hand reward behaviour is explicit ─────────────────
    section("11. No-decision hand chip delta attaches to latest transition")
    bot = make_training_bot()
    view = view_for("facing_bet")
    bot.act(view)
    bot.act(view)
    bot.record_reward(0.5)            # normal hand → [0.0, 0.5]
    bot.record_reward(0.3)            # no-decision hand → folded into last step

    rewards = [s.get("reward") for s in bot.current_episode]
    check("delta added to most recent transition", rewards == [0.0, 0.8],
          f"got {rewards}")
    check("episode return preserved", abs(sum(rewards) - 0.8) < 1e-9,
          f"sum={sum(rewards)}")

    bot2 = make_training_bot()
    bot2.record_reward(1.0)           # nothing to credit: explicitly dropped
    check("episode with zero transitions drops the reward (documented)",
          bot2.current_episode == [] and bot2.episode_buffer == [],
          "reward created phantom state")

    # ── 12. Raise-bucket aliasing contract ────────────────────────────────
    section("12. Raise buckets 3/4/5 share type, differ only in amount")
    bot = make_training_bot()
    legal = [{"type": "fold"}, {"type": "call"},
             {"type": "raise", "min": 10, "max": 50}]
    acts = [bot._action_idx_to_action(i, legal) for i in (3, 4, 5)]
    check("buckets keep aggressive type",
          all(a.type == "raise" for a in acts),
          f"types: {[a.type for a in acts]}")
    check("buckets map to min / midpoint / max",
          [a.amount for a in acts] == [10, 30, 50],
          f"amounts: {[a.amount for a in acts]}")
    check("aggressive executed action keeps the sampled bucket index",
          all(bot._executed_action_idx(a, i) == i
              for i, a in zip((3, 4, 5), acts)),
          "bucket index not preserved through _executed_action_idx")

    # ── 13. Masked PPO update end-to-end ───────────────────────────────────
    section("13. Masked PPO update runs and keeps parameters finite")
    bot = make_training_bot()
    for _ in range(2):                      # two short episodes
        for scenario in ("open_bet", "facing_bet", "fold_call_only"):
            v = view_for(scenario)
            bot.act(v)
            bot.act(v)
            bot.record_reward(0.2)
        bot.record_terminal_bonus(-0.5)
        bot.end_episode()
    try:
        bot.flush_buffer()
        finite = all(
            torch.isfinite(p).all().item()
            for net in (bot.policy_net, bot.value_net)
            for p in net.parameters()
        )
        check("update completed with finite parameters", finite,
              "non-finite parameter after masked PPO update")
    except Exception as e:  # noqa: BLE001
        check("update completed with finite parameters", False, repr(e))

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    if FAILURES:
        print(f"OVERALL: SOME CHECKS FAILED [FAIL]  ({len(FAILURES)} failure(s))")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("OVERALL: ALL CHECKS PASSED [PASS]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
