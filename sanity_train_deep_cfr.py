"""Schema-v2 Deep CFR training pipeline sanity gate."""
from __future__ import annotations

import os
import tempfile

import torch

from bots.deep_cfr_bot import DEEP_CFR_SCHEMA_VERSION
from sanity_deep_cfr_v2 import run as run_v2_checks
from training.train_deep_cfr import parse_args, run_training


def run() -> bool:
    run_v2_checks()
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = os.path.join(tmp, "deep_cfr_v2_smoke.pt")
        args = parse_args([
            "--variant", "small",
            "--iterations", "10",
            "--round-size", "5",
            "--update-interval", "2",
            "--checkpoint-interval", "5",
            "--batch-size", "2",
            "--aivat-sims", "1",
            "--save-path", checkpoint,
            "--device", "cpu",
            "--disable-collapse-canary",
        ])
        result = run_training(args)
        assert result["status"] == "complete"
        assert result["final_iter"] == 10
        assert result["round_index"] == 2
        assert result["gradient_steps_taken"] > 0
        assert result["head_steps"]["strategy"] > 0

        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False)
        assert payload["schema_version"] == DEEP_CFR_SCHEMA_VERSION
        assert payload["algorithm"] == "multiway_deep_cfr_inspired"
        assert payload["last_fit_iteration"] == 10
        assert set(payload["optimizer_state_dicts"]) == {
            "advantage", "strategy", "value", "sizing"}
        assert set(payload["reservoirs"]) == {
            "regret", "strategy", "value", "sizing"}

        resume_args = parse_args([
            "--variant", "small",
            "--iterations", "12",
            "--round-size", "5",
            "--update-interval", "2",
            "--checkpoint-interval", "12",
            "--batch-size", "2",
            "--aivat-sims", "1",
            "--save-path", checkpoint,
            "--resume", checkpoint,
            "--device", "cpu",
            "--disable-collapse-canary",
        ])
        resumed = run_training(resume_args)
        assert resumed["status"] == "complete"
        assert resumed["final_iter"] == 12
        assert resumed["round_index"] == 3

    print("[PASS] real frozen-round smoke and resume")
    print("ALL SCHEMA-V2 TRAINING CHECKS PASSED")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
