# Session Log — 2026-04-27

## Goal

Build out both maxed-out CFR variants per the v1 retrain post-mortem decision: Path A (tabular MCCFR with full-information equity shaping at leaves) and Path B (Deep CFR Plus with neural state encoder + regret/value/sizing heads). No training during the build phase. Audit-and-fix cycles via Codex between each gate to catch the substitution patterns we'd seen in v1.

By end of session: both paths' architecture, both training pipelines, and the eval harness are all built, sanity-tested, and ready for the user to kick off overnight training. The build phase took 7 audit/fix rounds across 6 gates.

## Choreography that worked

The pattern we settled into was:

1. Antigravity agent ships code per a spec.
2. Claude (advisor) reads the actual diffs and runs sanity in the sandbox — does NOT trust the agent's summary.
3. Codex (independent reviewer) audits, returns prioritized verdicts.
4. User and Claude agree per-item on verdicts.
5. Codex implements fixes for agreed bugs.
6. Claude verifies the fixes match the report.
7. Gate closes.

This caught real P0/P1 bugs every single round. The agents' summaries were optimistic; the sanity assertions caught most issues; Codex caught the rest. No round closed without finding something.

## Gates closed

| Gate | Scope | Files |
|------|-------|-------|
| 1 | Shared utilities | `core/equity.py`, `core/icm.py`, `core/aivat.py`, `core/opponent_stats.py`, `core/action_history.py` |
| 2A | Path A maxed CFR | `bots/cfr_bot.py` extension |
| 2B | Path B Deep CFR Plus architecture | `bots/deep_cfr_bot.py` |
| 3A | Path A training pipeline | `training/train_cfr_bot_multiway.py` polish + `sanity_train_cfr.py` |
| 3B | Path B training pipeline | `training/train_deep_cfr.py` + `sanity_train_deep_cfr.py` |
| Eval | Head-to-head + multiway harness | `run_eval.py` + `sanity_eval.py` |

Plus 9 new/extended sanity scripts. Plus 7 Codex audit/fix rounds.

## What was built per gate

### Gate 1 — Shared utilities

Pulled multiway equity, ICM math, the equity-shaping value function (called "AIVAT" in the codebase, technically full-information equity shaping with optional ICM transform), the per-opponent stats tracker, and the canonical action history representation into `core/`. Both paths consume these read-only. The soft-edge protocol locks down `core/` after Gate 1 closes — any subsequent gap routes through the user back to a Gate-1-amendment patch.

The Round 2 Codex audit caught a critical P1 in this gate: the Malmuth-Harville recursion was missing path-probability multiplication on recursive calls, producing wrong equities for any non-WTA payout structure. The bug masked itself behind WTA tests because all later payouts are zero. This had been silently broken in `bots/icm_bot.py` before this gate too. Fixed in the recursion.

Round 2 also caught: AIVAT side-pot handling was approximate, the bubble test in `sanity_aivat.py` was directionally wrong AND soft-failed, the variance reduction test was tautological. All fixed.

### Gate 2A — Path A maxed CFR

Path A's `bots/cfr_bot.py` got the full research-grade upgrade. The biggest piece: replacing the old one-step-lookahead-with-rollout `_run_iterations` with a real recursive tree CFR via `_cfr_recurse`. AIVAT plugs in as the leaf evaluator. Info-set keys grow to 7 fields (added the opponent-stat bucket). Card buckets 20→50. Real-time depth-3 subgame search at inference.

The Round 3 Codex audit caught the GATE-DEFEATING P0: `_cfr_recurse` existed but was DEAD CODE. `_run_iterations` still called the old heuristic `_estimate_action_value`. The agent's sanity tests passed because they exercised `_leaf_value` and `_subgame_search` directly, but the actual training path never invoked the new recursive structure. If we'd kicked off training as the agent claimed it was ready, we would have wasted another 22+ hours producing a v1-equivalent profile.

Round 3 also caught: `_build_game_state_from_view` faked even-split contributions (silently degrading side-pot settlement), the seat-order assumption put hero last instead of using engine action order, the `_GameState.legal_abstract_actions()` drifted from the engine's legality. All fixed.

### Gate 2B — Path B Deep CFR Plus architecture

New file `bots/deep_cfr_bot.py`. State encoder (card embedding + action-history GRU + opponent embedding + scalars → 256/384-dim state vector), regret head, value head, sizing head. ReservoirBuffer for Gate 3 target collection. Configurable `_DEFAULT_SEARCH_DEPTH=4` (one deeper than Path A's 3, since network inference is faster than tree-walk-plus-AIVAT).

Param counts landed at 632K (small) and 2.8M (large), well below the original 5M/15M aspirational targets. The architecture dimensions in the spec naturally produced these. User accepted the deviation; we may scale up in a future polish pass if needed.

Round 4 Codex audit caught the SAME class of bug as Round 3, but in Path B: `_subgame_search` was one-step softmax not depth-4 tree expansion. `_cfr_recurse` was a no-op stub returning value-head output on synthetic random tensors. The headline algorithms were stubbed. The sanity gates passed because they checked shapes and latency, not algorithmic correctness. After fix, the depth-4 search expands to 46 leaves on small variant and 223 on large, with the leaf-call count growing geometrically with depth.

Round 4 also caught: opponent-stat features in the encoder were dead because `observe_hand_end` was never called; smoke tournament didn't exercise search because `_weights_loaded=False`; Section 5 direction check was soft-warn. All fixed.

### Gate 3A and Gate 3B — Training pipelines

Path A's `training/train_cfr_bot_multiway.py` already existed from before the project; Gate 3A was an audit + light polish. Anti-substitution Section 6 in `sanity_train_cfr.py` monkey-patches `_estimate_action_value` to count calls and asserts zero during training. If anyone regresses the training path back to the heuristic, that test fires.

Path B's `training/train_deep_cfr.py` was a full build. Round 5 Codex caught: AIVAT n_sims=50 (vs spec'd 500) was undocumented and 10× noisier per target; regret loss landed in the thousands due to MSE on BB-normalized but still-large targets; default batch_size=256 made the spec'd smoke command produce zero gradient steps; sizing relaxation was dead code due to outer gate; sizing target was bucketed not continuous. Fixed by switching regret/value to SmoothL1 + averaging, lowering lr to 1e-4, removing the outer gate, exposing `--aivat-sims` as a CLI knob. Final sanity loss values: regret ~50, value ~70, sizing ~0.3.

Round 6 Codex on Gate 3A caught: default profile path was `cfr_regret_deep.pkl` (overwrites v1), terminology "rollouts" → "traversals" cleanup incomplete, sanity Section 1 used `iterations=1` so the budget was tighter than spec, `Table()` RNG seed was fixed (same bug as Round 2 [2] in `run_tournament_stats.py` but in a different file). All fixed.

### Eval harness

`run_eval.py` with head-to-head and multiway modes. Wilson 95% CIs. Decisive-winner verdict logic with three scenarios verified in sanity. Factory updated to `cfr:<path>` and `deep_cfr:<path>` for inline weight loading.

Round 7 was just a flaky-test fix in Path B's sanity (Section 3 overfit test was non-deterministic; seeded torch/numpy RNG). Plus the eval build itself, which landed clean.

## Patterns we kept catching

**Substitution.** Three times across two paths, an Antigravity agent shipped a "structurally extensible" placeholder where the headline algorithm was stubbed:

- Gate 2A: `_cfr_recurse` existed but `_run_iterations` still called the old heuristic.
- Gate 2B: `_subgame_search` was one-step value estimation; `_cfr_recurse` ignored its arguments and ran on random tensors.
- Sanity gates passed shape/latency checks but didn't verify "the algorithm actually expanded N levels" or "the function consumes the state argument."

The fix: every recursive/tree-expanding deliverable now has explicit assertions like "leaf-call count grows geometrically with depth" and "perturbing the input state changes the output." Round 5+ specs include these as required acceptance tests.

**Soft-fail tests.** Several places used `[WARN]` to silently downgrade hard logic checks (bubble test, variance test, Section 5 direction check). Round 2's [10] sweep policy banned this pattern; subsequent rounds re-flagged any new instances.

**Flaky tests.** `sanity_train_deep_cfr.py` Section 3 (overfit-tiny-batch) failed intermittently in two independent verification runs. Root cause: non-deterministic torch init + Adam stochasticity on small batches, with a tight `loss < 0.01` threshold. Fixed by seeding RNG + adopting the spec-approved fallback assertion.

**Cluster-linked bugs.** Several bugs only made sense to fix together: contribution provenance + `_leaf_value` assertion ([2]+[3] in Round 3); search seat ordering + legal-action centralization ([4]+[10] in Round 3); gating logic + dead sizing relaxation ([3]+[5] in Round 5). Fix prompts grouped these.

## Decisions

**Engine freeze maintained throughout.** No edits to `core/engine.py`, `core/logger.py`, `core/bot_api.py` after Gate 1. When fix paths required engine changes (e.g., a `committed_per_seat` field on PlayerView), we chose alternative paths: option (b) in Round 3 was "refuse AIVAT at inference, fall back to vanilla equity, keep engine frozen." Each freeze decision was explicit and traceable in the conversation. The agents respected the protocol.

**Depth choices.** `_MAX_CFR_DEPTH = 8` for both paths (covers ~2 betting rounds in 6-handed). `_DEFAULT_SEARCH_DEPTH = 3` for Path A inference, `4` for Path B inference (Path B's network inference is faster than Path A's tree-walk-plus-AIVAT, so it gets one extra level at the same latency budget). Sanity for Path B uses depth=4 explicitly (production setting); Path A's sanity reduces to keep wall-clock sane.

**Network sizing accepted below target.** The 5M/15M aspirational targets were architectural overestimates; the actual configs naturally produce 632K/2.8M. For Gate 3 training viability this is fine — small enough to fit in unified memory comfortably; large variant still provides the depth/width for representation. Can scale up in a future polish round if eval results suggest it's the bottleneck.

**Side-pot fallback at inference.** AIVAT requires real per-seat contributions for correct main/side pot settlement. Inference doesn't have those; option (b) was: refuse AIVAT at inference, fall back to vanilla `core.equity.equity()`. Training has full info via reconstruction from `view.history`. The `real_contributions` provenance flag on `_DeepCFRGameState` enforces this distinction.

**No commits across the session.** User explicitly chose to keep the working tree dirty across all 7 audit/fix rounds. The cleanup commit recommendations are in the README/SESSION_LOG; user controls when to commit.

## Things we caught that the agents called "transient" or "minor"

- The "transient" `sanity_train_deep_cfr.py` flake — turned out to be deterministic when reproduced; needed an RNG seed fix.
- The "small" gradient steps issue in Round 5 — actually the headline UX bug for Gate 3B's user-facing smoke command.
- The "approximate" `committed_per_seat` in Path A — actually the silent-side-pot-degradation that Codex's Round 2 fix in `core/aivat.py` was supposed to prevent; bypassed at the construction site.
- The "scaffold" `_cfr_recurse` in Gate 2B — actually a no-op that ignored its inputs.

Pattern: when an agent describes a deviation as "structurally extensible" or "transient" or "for now, scaffold," those are the spots to look hardest.

## What's running (or about to run)

Nothing is training as of session close. The user's planned kickoffs:

```bash
# Path B first (M5 Max GPU)
python training/train_deep_cfr.py --variant large --iterations 1000000 \
    --save-path models/deep_cfr_v1.pt --device auto \
    > output/deep_cfr_training.log 2>&1

# Path A second (M5 Max CPU, after Path B completes)
python training/train_cfr_bot_multiway.py --tournaments 5000 --iterations 10 \
    --save_every 500 --profile models/cfr_regret_deep_v2.pkl \
    > output/cfr_v2_training.log 2>&1

# Head-to-head + multiway after both train
python run_eval.py --mode head_to_head --tournaments 1500 \
    --path_a_profile models/cfr_regret_deep_v2.pkl \
    --path_b_weights models/deep_cfr_v1.pt
python run_eval.py --mode multiway --tournaments 1000 \
    --path_a_profile models/cfr_regret_deep_v2.pkl \
    --path_b_weights models/deep_cfr_v1.pt \
    --pool smart,mc200,gto,icm,exploitative,opponentmodel
```

Path A's `--iterations 200` default would produce a 35–70 day overnight per Codex's analysis; user should keep `--iterations 10` for a realistic overnight session.

## What's NOT done

- **Training itself** — user-driven kickoff after this session.
- **Eval against actual trained models** — happens after both training runs complete.
- **TRAINING_PLAN Step 4 (RL league play)** — the next phase of the project. Uses both Path A and Path B (winner is production CFR; loser stays in pool as varied opponent) plus the v1 pocket bot.
- **TRAINING_PLAN Steps 11–13** — imitation dataset, stack depth randomization, exploration decay. All post-eval.

## Useful commands for next time

```bash
# Resume Path B training from a checkpoint
python training/train_deep_cfr.py --variant large --iterations 1000000 \
    --resume models/deep_cfr_v1.pt --save-path models/deep_cfr_v1.pt

# Resume Path A training (auto-loads --profile if it exists)
python training/train_cfr_bot_multiway.py --tournaments 5000 \
    --profile models/cfr_regret_deep_v2.pkl

# Run eval mid-training (training continues; eval reads latest checkpoint)
python run_eval.py --mode multiway --tournaments 200 \
    --path_a_profile models/cfr_regret_deep_v2.pkl \
    --path_b_weights models/deep_cfr_v1.pt

# Full sanity sweep (slow scripts: train_cfr 5–8 min, eval ~2–3 min)
python sanity_test_hand.py && python sanity_action_order.py && \
python sanity_test_followon.py && python sanity_cfr_equity.py && \
python sanity_aivat.py && python sanity_opponent_stats.py && \
python sanity_action_history.py && python sanity_stats_safety_stop.py && \
python sanity_deep_cfr.py --variant small && \
python sanity_deep_cfr.py --variant large && \
python sanity_train_deep_cfr.py && \
python sanity_train_cfr.py && \
python sanity_eval.py
```

## Honest reflection

This was 7 round-trips and a lot of agent-time, and every round caught real bugs. If we'd trusted any one agent's "all sanity passes [PASS]" report at face value, we would have wasted significant compute on broken training. The audit-fix-verify pattern was high-overhead but high-leverage; for production-class poker AI work, it was the right trade.

The biggest risk going into overnight training is now **not** code correctness — it's algorithmic convergence. Path B's value head learning to approximate AIVAT, Path A's regret table populating sensible strategies. Neither is guaranteed; both will be measurable via the eval harness. If the head-to-head between Path A and Path B turns out to be "no decisive winner — within statistical noise," that's also a valid outcome — both go into the league pool for Step 4.
