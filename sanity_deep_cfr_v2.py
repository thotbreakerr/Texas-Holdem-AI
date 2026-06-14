"""Deterministic schema-v2 Deep CFR regression gate."""
from __future__ import annotations

import io
import os
import random
import tempfile
from contextlib import redirect_stdout

import torch

from bots.deep_cfr_bot import (
    ABSTRACT_ACTIONS,
    DEEP_CFR_SCHEMA_VERSION,
    NUM_ACTIONS,
    DeepCFRBot,
    DeepCFRConfig,
    ReservoirBuffer,
    _DeepCFRGameState,
    build_network_input,
)
from core.bot_api import PlayerView
from training.train_deep_cfr import (
    CURRICULUM_PROFILES,
    classify_canary,
    classify_extra_canary_metrics,
    load_checkpoint,
    parse_args,
    pilot_health_failures,
    quick_canary_probe,
    save_checkpoint,
    train_step,
)


def _quiet_bot() -> DeepCFRBot:
    with redirect_stdout(io.StringIO()):
        bot = DeepCFRBot(
            config=DeepCFRConfig.small(),
            inference_mode=False,
            aivat_sims=1,
        )
    return bot


def _state(actor: int = 0) -> _DeepCFRGameState:
    return _DeepCFRGameState(
        pot=15,
        stacks=[1000, 990],
        committed_per_seat=[0, 10],
        total_committed_per_seat=[0, 10],
        alive=[True, True],
        street="preflop",
        board=[],
        hole_cards={
            0: (("A", "s"), ("K", "s")),
            1: (("2", "c"), ("7", "d")),
        },
        seat_order=[actor],
        action_idx=0,
        history_events=[],
        deck_remaining=[],
        big_blind=10,
        ring_order=[0, 1],
    )


def _view(player_count: int = 6) -> PlayerView:
    pids = ["hero"] + [f"opp{i}" for i in range(1, player_count)]
    return PlayerView(
        me="hero",
        street="preflop",
        position="BTN",
        hole_cards=[("A", "s"), ("K", "s")],
        board=[],
        pot=15,
        to_call=10,
        min_raise=20,
        max_raise=1000,
        stacks={pid: 1000 for pid in pids},
        opponents=pids[1:],
        legal_actions=[
            {"type": "fold"},
            {"type": "call"},
            {"type": "raise", "min": 20, "max": 1000},
        ],
        history=[],
    )


def _optimizers(bot: DeepCFRBot) -> dict:
    return {
        "advantage": torch.optim.Adam(bot.network.advantage.parameters(), 1e-3),
        "strategy": torch.optim.Adam(bot.network.strategy.parameters(), 1e-3),
        "value": torch.optim.Adam(bot.network.value.parameters(), 1e-3),
        "sizing": torch.optim.Adam(bot.network.sizing.parameters(), 1e-3),
    }


def _final_linear(module: torch.nn.Module) -> torch.nn.Linear:
    layers = [item for item in module.modules() if isinstance(item, torch.nn.Linear)]
    return layers[-1]


def check_traversal_contracts() -> None:
    bot = _quiet_bot()
    leaf_calls = {"count": 0}

    def constant_leaf(_state, _hero):
        leaf_calls["count"] += 1
        return 10.0

    bot._aivat_leaf_value = constant_leaf
    state = _state(actor=0)
    legal = state.legal_abstract_actions()
    regret = ReservoirBuffer()
    strategy = ReservoirBuffer()
    value = ReservoirBuffer()
    sizing = ReservoirBuffer()
    result = bot._cfr_recurse(
        state,
        hero_seat=0,
        depth=1,
        iteration=11,
        regret_buf=regret,
        strategy_buf=strategy,
        value_buf=value,
        sizing_buf=sizing,
        exploration_epsilon=0.0,
    )

    assert len(regret) == 1
    _, target, mask, weight = regret.buffer[0]
    collected = {idx for idx in range(NUM_ACTIONS) if mask[idx].item() == 1.0}
    assert collected == set(legal)
    assert ABSTRACT_ACTIONS.index("all_in") in collected
    assert leaf_calls["count"] == len(legal)
    assert weight == 11.0

    action_values = {}
    for action_idx in legal:
        next_state = state.apply_action(0, action_idx)
        added = (
            next_state.committed_per_seat[0]
            - state.committed_per_seat[0]
        ) / state.big_blind
        action_values[action_idx] = 10.0 - added
    expected_ev = sum(action_values.values()) / len(action_values)
    assert abs(result - expected_ev) < 1e-6
    for action_idx in legal:
        assert abs(target[action_idx].item()
                   - (action_values[action_idx] - expected_ev)) < 1e-5

    opponent_bot = _quiet_bot()
    opponent_calls = {"count": 0}

    def one_leaf(_state, _hero):
        opponent_calls["count"] += 1
        return 0.0

    opponent_bot._aivat_leaf_value = one_leaf
    opponent_strategy = ReservoirBuffer()
    opponent_regret = ReservoirBuffer()
    opponent_bot._cfr_recurse(
        _state(actor=1),
        hero_seat=0,
        depth=1,
        iteration=7,
        regret_buf=opponent_regret,
        strategy_buf=opponent_strategy,
        value_buf=ReservoirBuffer(),
        sizing_buf=ReservoirBuffer(),
        exploration_epsilon=0.0,
    )
    assert opponent_calls["count"] == 1
    assert len(opponent_regret) == 0
    assert len(opponent_strategy) == 1
    _, policy, policy_mask, policy_weight = opponent_strategy.buffer[0]
    legal_opponent = set(_state(actor=1).legal_abstract_actions())
    assert policy_weight == 7.0
    assert abs(policy.sum().item() - 1.0) < 1e-6
    for idx in range(NUM_ACTIONS):
        assert policy_mask[idx].item() == float(idx in legal_opponent)
        if idx not in legal_opponent:
            assert policy[idx].item() == 0.0


def check_network_isolation_and_counts() -> None:
    bot = _quiet_bot()
    device = next(bot.network.parameters()).device
    batch = _state().to_network_input(0)
    assert torch.equal(
        bot.network.advantage_forward(batch),
        torch.zeros(1, NUM_ACTIONS, device=device),
    )

    advantage_before = {
        name: value.detach().clone()
        for name, value in bot.network.advantage.state_dict().items()
    }
    value_buf = ReservoirBuffer()
    sizing_buf = ReservoirBuffer()
    for _ in range(2):
        value_buf.add((batch, 3.0))
        sizing_buf.add((batch, 0.5))
    _, _, _, info = train_step(
        bot.network,
        _optimizers(bot),
        ReservoirBuffer(),
        value_buf,
        sizing_buf,
        batch_size=2,
        device=device,
        strategy_buf=ReservoirBuffer(),
    )
    assert info["heads_trained"]["value"]
    assert info["heads_trained"]["sizing"]
    for name, value in bot.network.advantage.state_dict().items():
        assert torch.equal(value, advantage_before[name])

    two = build_network_input(_view(2), None)
    six = build_network_input(_view(6), None)
    assert not torch.equal(two["scalars"], six["scalars"])
    seated_slice = slice(14, 19)
    active_slice = slice(19, 24)
    assert two["scalars"][0, seated_slice].tolist() == [1, 0, 0, 0, 0]
    assert six["scalars"][0, seated_slice].tolist() == [0, 0, 0, 0, 1]
    assert two["scalars"][0, active_slice].tolist() == [1, 0, 0, 0, 0]
    assert six["scalars"][0, active_slice].tolist() == [0, 0, 0, 0, 1]


def check_deployment_policy_and_checkpoint() -> None:
    bot = _quiet_bot()
    with torch.no_grad():
        advantage_out = _final_linear(bot.network.advantage.head)
        strategy_out = _final_linear(bot.network.strategy.head)
        advantage_out.weight.zero_()
        advantage_out.bias.zero_()
        advantage_out.bias[ABSTRACT_ACTIONS.index("all_in")] = 10.0
        strategy_out.weight.zero_()
        strategy_out.bias.zero_()
        strategy_out.bias[ABSTRACT_ACTIONS.index("fold")] = 10.0

    deployed, legal, _ = bot.policy_probabilities(_view(), use_advantage=False)
    diagnostic, _, _ = bot.policy_probabilities(_view(), use_advantage=True)
    assert max(legal, key=lambda idx: deployed[idx]) == ABSTRACT_ACTIONS.index("fold")
    assert max(legal, key=lambda idx: diagnostic[idx]) == ABSTRACT_ACTIONS.index("all_in")
    assert all(deployed[idx] == 0.0 for idx in range(NUM_ACTIONS) if idx not in legal)

    buffers = {
        "regret": ReservoirBuffer(10),
        "strategy": ReservoirBuffer(10),
        "value": ReservoirBuffer(10),
        "sizing": ReservoirBuffer(10),
    }
    buffers["regret"].add(("regret", 1))
    buffers["strategy"].add(("strategy", 2))
    buffers["value"].add(("value", 3))
    buffers["sizing"].add(("sizing", 4))
    optimizers = _optimizers(bot)
    losses = {"regret": [], "strategy": [], "value": [], "sizing": []}

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "v2.pt")
        save_checkpoint(
            path,
            123,
            bot,
            optimizers,
            losses,
            buffers=buffers,
            canary_fail_streak=2,
            round_index=4,
            last_fit_iteration=100,
            curriculum_profile="sixmax",
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["schema_version"] == DEEP_CFR_SCHEMA_VERSION

        restored = _quiet_bot()
        restored_buffers = {
            name: ReservoirBuffer(1) for name in buffers
        }
        restored_optimizers = _optimizers(restored)
        meta = {}
        iteration = load_checkpoint(
            path,
            restored,
            restored_optimizers,
            meta_out=meta,
            buffers=restored_buffers,
        )
        assert iteration == 123
        assert meta["canary_fail_streak"] == 2
        assert meta["round_index"] == 4
        assert meta["last_fit_iteration"] == 100
        for name, value in bot.network.state_dict().items():
            assert torch.equal(value.cpu(), restored.network.state_dict()[name].cpu())
        for name in buffers:
            assert restored_buffers[name]._count == buffers[name]._count
            assert restored_buffers[name].buffer == buffers[name].buffer

        v1_path = os.path.join(tmp, "v1.pt")
        torch.save({"iteration": 137381, "network_state_dict": {}}, v1_path)
        try:
            load_checkpoint(
                v1_path,
                restored,
                restored_optimizers,
                buffers=restored_buffers,
            )
        except RuntimeError as exc:
            assert "postmortem-only" in str(exc)
        else:
            raise AssertionError("schema-v1 resume was accepted")


def check_probability_canary_and_cli() -> None:
    class CollapsedV1Fixture:
        search_depth = 1

        def policy_probabilities(self, _view, *, search_depth=0,
                                 use_advantage=False):
            _ = search_depth, use_advantage
            probs = [0.0] * NUM_ACTIONS
            probs[ABSTRACT_ACTIONS.index("fold")] = 0.449
            probs[ABSTRACT_ACTIONS.index("all_in")] = 0.551
            return probs, list(range(NUM_ACTIONS)), 1.0

    metrics = quick_canary_probe(
        CollapsedV1Fixture(), torch.device("cpu"), n=8, seed=19)
    assert abs(metrics["raw_all_in"] - 0.551) < 1e-9
    assert abs(metrics["search_all_in"] - 0.551) < 1e-9
    assert metrics["preflop_avg_raise"] == 0.0
    assert classify_canary(
        metrics["raw_all_in"], metrics["search_all_in"]) == "FAIL"
    assert classify_extra_canary_metrics(metrics)[0] == "FAIL"
    assert pilot_health_failures(metrics)
    assert not pilot_health_failures({
        "raw_all_in": 0.09,
        "search_all_in": 0.14,
        "strong_continue": 0.80,
        "normal_action_mass": 0.30,
    })

    weights = CURRICULUM_PROFILES["sixmax"]
    assert weights == {2: 0.125, 3: 0.10, 4: 0.125, 5: 0.15, 6: 0.50}
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    args = parse_args([
        "--variant", "small",
        "--curriculum-profile", "sixmax",
        "--canary-enforce-iteration", "100000",
        "--canary-fail-patience", "3",
    ])
    assert args.round_size == 25_000
    assert args.save_path.endswith("deep_cfr_v2.pt")


def run() -> bool:
    random.seed(7)
    torch.manual_seed(7)
    checks = [
        ("traversal contracts", check_traversal_contracts),
        ("network isolation and count encoding", check_network_isolation_and_counts),
        ("average-policy deployment and checkpoint", check_deployment_policy_and_checkpoint),
        ("probability canary and CLI", check_probability_canary_and_cli),
    ]
    for label, check in checks:
        check()
        print(f"[PASS] {label}")
    print("ALL SCHEMA-V2 DEEP CFR CHECKS PASSED")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
