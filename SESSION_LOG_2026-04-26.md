# Session Log — 2026-04-26

## Goal

Get the CFR bot ready for an overnight retrain that produces a competitive multiway profile. The pre-existing profile (`cfr_regret_deep.pkl`) was heads-up-flavored and structurally limited. This session closed the gaps that mattered most before kicking off training.

## What changed

### Engine (`core/engine.py`, `core/logger.py`)

- **Logger filename collision fix.** Filenames now include microseconds + PID + monotonic counter and open in append mode. Without this, the M5's ~324 hands/sec could collide many JSONL files into the same filename and silently overwrite each other — would have wrecked the Step-1 dataset generation.
- **Fold-win path now logs results.** When a hand ended by everyone folding to one player, the engine returned early without calling `logger.log_result()`, `flush()`, or `close()`. This meant fold-win hands had decision rows but no result rows in the JSONL, biasing `--filter_winners` away from fold-equity wins.
- **Tournament safety-stop ranks honestly.** When `--max-hands` triggered with multiple survivors, all of them used to be tagged as position 1. Now they're chip-ranked, or marked as "?" if zero eliminations happened.
- **Bet/raise label fix when `to_call=0` and `current_bet>0`.** The classic BB-postflop-after-limps case used to expose a "bet" action when poker theory says it should be a "raise". Numeric outcome was the same but bots branching on action type saw the wrong shape.
- **`pot_before` recorded in every history entry.** Each action in `history` now carries the pot snapshot at action time. The CFR history tokenizer uses this so that a preflop pot-sized bet stays a "P" token forever, instead of drifting to "S" as the pot grows over later streets.

### ML training (`training/train_ml_bot.py`)

- **`verbose=True` removed from `ReduceLROnPlateau`.** That kwarg was deprecated in torch 2.2 and removed in 2.11. Kept manual LR-change logging by comparing `param_groups[0]["lr"]` before/after `scheduler.step()`.
- **BB position encoding fixed.** Training had `"BB": 0.5`; inference (`bots/ml_bot.py`) had `"BB": 0.3`. Trained models were silently mismapped at play time. Now both sides use 0.3.

### CFR — multiway equity (`bots/cfr_bot.py`)

The previous `_quick_equity` and `_postflop_bucket` sampled exactly one opponent hand per rollout. In 6-handed pots they overstated equity by ~30 points (AQ on A72: CFR estimate 0.95, true 5-way 0.55). The bot was being trained against heads-up math while playing 6-handed.

- Both functions rewritten as proper multiway MC: sample `n_opponents` hands per sim, hero must beat all, ties get 0.5 credit.
- Sim count bumped 20 → 100 for tighter estimates.
- `n_opponents` is a required parameter on both — no defaults that could mask future regressions.

### CFR — value function rewrite

The patched-multiway value function still had a structural ceiling: `bet_value > call_value` iff `fold_equity * (0.25 - showdown) > penalty`. Two regimes:
- showdown < 0.25: shoves looked good for marginal hands (shove bias)
- showdown > 0.25: bets always lost to calls (passive bias on premiums)

No tuning of the existing knobs could fix both at once. Replaced the formula with chip-EV math that treats each branch separately:

- Fold = 0 baseline.
- Check/call = `equity * (pot + cost) - (1 - equity) * cost` (real pot odds when there's a call; just `equity * pot` when checking).
- Bet/raise = `fold_equity * pot + (1 - fold_equity) * (equity * called_pot - bet_size) - risk_penalty`.
- Returns `ev / pot` so regrets stay bounded across pot sizes.

Now requires `call_amount` and `hero_stack` — both already on `PlayerView`, just plumbed through.

### CFR — research-baseline abstraction

These bring CFR up to "what every published bot has." Each one extends the info-set key, so the total key space grew ~60x vs the old profile. That's the cost of abstraction quality.

- **Position bucket** (4 categories: early/middle/late/blinds). Closes the README's open Known Limitation. UTG opens and BTN opens no longer collapse into the same node.
- **SPR (effective stack-to-pot ratio) bucket** (3 categories: low <5, mid 5-15, high >15). Tournament play needs this — the same hand at 2bb effective vs 200bb effective demands totally different play.
- **Bet sizes 4 → 6.** Added `bet_50` and `bet_75` between the existing 33/67/100/all_in.
- **History tokens 4 → 6.** Added `Q` (~50%) and `L` (~75%) bins so the new bet sizes don't collapse on read-back.
- **Card buckets 10 → 20.** Halves the equity-bucket boundary noise. Adds 2x to key space.

### CFR — abstraction hygiene

- **Phantom action dedup.** When multiple sizings clamp to the same concrete bet (e.g. all of bet_33/50/67 → raise 20 when min_raise > pot * 0.67), `_legal_abstract_actions` now keeps only the first and drops the rest. Stops CFR from accumulating distinct regrets for what is functionally one action.
- **Old-format profile keys discarded at load.** The 109k pre-multiway keys in `cfr_regret_deep.pkl` had only 2 colons; new keys have 5. Load-time filter drops anything with `< 5` colons and prints a clear count + warning. No silent fallback to dead nodes.

### Sanity script (`sanity_cfr_equity.py`)

Built up across rounds. Now has 6 test sections:
1. Postflop multiway equity monotonicity (AQ on A72, n_opponents 1/2/5)
2. Preflop multiway equity bounds (AA/KK/AKs/AQo/72o)
3. Action-value sanity (premium hands don't fold, trash folds, marginals don't shove)
4. Position bucket helper
5. SPR bucket helper
6. Info-set key includes position and SPR (silent-bug guard)

## Pre-flight verification

Before kicking off the retrain, all of the following passed:
- `python sanity_test_hand.py`
- `python sanity_action_order.py`
- `python sanity_test_followon.py`
- `python sanity_cfr_equity.py`
- 3-player and 6-player smoke tournaments
- `git diff --check` clean

The old profile was backed up to `models/cfr_regret_deep.PRE_MULTIWAY.bak` before training started writing the new format.

## What's running now

Overnight CFR retrain — `training/train_cfr_bot_multiway.py`, single-threaded.

```
caffeinate -i -m nohup python training/train_cfr_bot_multiway.py \
    --tournaments 100000 --iterations 200 --save_every 500 \
    > output/cfr_training.log 2>&1 &
```

PID was 51080 at start. Logs are stdout-buffered (no `-u` flag) so the log file is empty for the entire run — checkpoint file timestamps and a separate Python `pickle.load` were the real progress signals.

## Training stopped — final state and lesson learned

**Stopped at 01:29 on 2026-04-27, ~22 hours elapsed.** Profile snapshot:
- `models/cfr_regret_deep.pkl` = 104 MB
- ~499k info sets
- ~25.4B iterations
- ~40% of the 100k-tournament target

The profile was archived to **`models/cfr_regret_deep.v1_pre_equity_shaping.pkl`** as a "pocket" reference bot. It's the result of training the new abstraction (n_opp + position + SPR + 20 buckets + 6 sizings) against the heuristic value function for ~22 hours.

### Performance on stop

Two stats runs, 50 tournaments each:

**6-player field (smart, mc200, cfr, gto, icm, opponentmodel):**
- opponentmodel 44%, icm 30%, gto 14%, mc200 8%, smart 4%, **cfr 0%**
- Random baseline at 6 players = 16.7%. CFR finished significantly below random.

**9-player field (mc200, smart, ml, rl, cfr, icm, exploitative, gto, opponentmodel):**
- opponentmodel 46%, icm 28%, gto 10%, mc200 8%, exploitative 6%, **cfr 2%**, smart/ml/rl 0% each
- CFR's heads-up record was actually OK against most bots (beats smart 100%, mc200 82%, gto 66%, exploitative 94%) but lost to opponentmodel (38%) and icm (32%) — the same two that won most tournaments. CFR's avg position was 3.80 of 9, mid-pack on survival but rarely closing.

### What this confirms

The two structural ceilings that were documented during the session were exactly what limited the trained bot:

1. **The heuristic `_estimate_action_value` has biases (passive on premiums, slight shove bias on marginals) that no tuning fixes.** With a biased value signal, more CFR iterations don't fix the strategy — they bake the bias deeper. The 22 hours of training converged CFR to "what the value function says is optimal," not to actual good poker.
2. **No opponent modeling.** opponentmodel and icm — the two bots that consistently beat CFR — both use information CFR doesn't have. CFR plays the same against a maniac and a nit.

### What was wasted vs. what wasn't

**Wasted:** 22 hours of CPU time and the trained profile's competitive value. The v1_pre_equity_shaping.pkl is kept as a "pocket bot" for future reference (potential opponent in league play, or starting point for transfer learning) but it's not a tournament-winning bot.

**Not wasted:** every code change. Engine fixes, multiway equity, value function rewrite, position/SPR/bet-size abstraction, sanity script, decision-log integrity — all real and stay. equity shaping (Step 2) plugs into the same info-set keys; none of this work gets thrown away.

## Revised plan: build BOTH maxed-CFR variants in parallel, train sequentially

The new direction is to spec everything once, build it all, and train at the end — no more iterative train-evaluate-fix-retrain loops. There will be **two maxed-out CFR bots** built and trained, then compared head-to-head:

### Path A — Tabular MCCFR, fully bolted

Extends the current `cfr_bot.py` architecture. Keeps the regret-table-keyed-by-string approach. Adds:
- **Real tree CFR.** Replace `_run_iterations` (one-step lookahead with rollout) with recursive game-tree traversal that samples opponent strategies from the current profile and computes counterfactual utilities at every node.
- **equity-shaped value function** for leaf evaluation during tree traversal.
- **Per-opponent stat bucket** added to the info-set key (4 categories: tight-passive / tight-aggressive / loose-passive / loose-aggressive).
- **Finer card buckets** (50-100 per street, up from 20).
- **Real-time search at decision time** (depth-2 or depth-3 subgame solve using the value function for leaves).

Plus the MCCFR+ improvements (CFR+ techniques from Tammelin 2014, applied here to our sampling-based MCCFR — combination called MCCFR+):
- **Linear iteration weighting** on regret accumulation (newer iterations contribute more weight than older ones — same idea as Deep CFR Plus on Path B). Improves convergence speed and stability without changing the algorithm structure.
- **Regret matching with positive regrets only** (clip negative regrets at zero rather than letting them accumulate). Standard CFR+ trick, makes regret matching more stable and pushes the strategy distribution toward positive-regret actions faster.

Trade-off: info-set key space explodes into the millions. Won't fully populate via overnight training. But debuggable and incremental from the current code, and MCCFR+ gets more out of the iterations it does run vs vanilla MCCFR.

### Path B — Deep CFR Plus + continuous bet sizing (neural approximation, juiced)

Separate new file (`bots/deep_cfr_bot.py`). Replaces the regret table with neural networks. Adds:
- **Regret network.** `(state_features) → regret_vector[NUM_ACTION_TYPES]`. Generalizes across similar states. Replaces `self._nodes` dict entirely.
- **Value network (equity shaping).** `(state_features) → expected_chip_delta`. Trained on full-information realized outcomes.
- **Shared state encoder.** Card embeddings + action-history GRU + opponent-stat embeddings + position/SPR/n_opp scalars → ~256-dim state vector. Both networks share this front-end.
- **Real tree CFR** (same recursive structure as Path A, just with networks instead of dict lookups).
- **Real-time search at decision time** (same idea as Path A).
- **Deep CFR Plus algorithm.** Linear weighting on regret samples (newer iterations weighted more heavily, like CFR+ does for tabular). ~10-30% better convergence than vanilla Deep CFR for the same training time. Same code complexity, just a different sample-weighting policy in the regret-network training loop.
- **Continuous bet-sizing head.** Instead of selecting from {bet_33, bet_50, bet_67, bet_75, bet_100, all_in} discretely, the network has a SECOND output head that emits a continuous bet fraction in [0.0, 2.0] when the regret head selects "raise". Gives Path B finer sizing resolution than Path A — addresses the "what's optimal between 67% and 75%?" question that discrete abstraction can't answer.

Trade-off: full rewrite of the CFR bot, more complex training infrastructure (regret head + sizing head trained jointly), but scales properly and is more expressive than discrete abstraction. This is the Pluribus / DeepStack / ReBeL family of approaches.

**Note on apples-to-oranges:** the continuous-sizing head means Path A and Path B have different action spaces. A vs B head-to-head is still meaningful (they play in the same engine), but Path B has strictly more expressive output. If Path B wins the comparison, some of that win is "richer action space" rather than "neural beats tabular." That's fine for picking a production bot but worth noting when interpreting the eval.

### Path B network sizing

Target ~12-18M total parameters. Justification from `HARDWARE_BENCHMARK.md`:
- Models below ~5M params get only ~1.4x MPS speedup (kernel-launch overhead dominates)
- Models around ~17M params get 7.66x MPS speedup (sweet spot, GPU properly fed)
- Models above ~50M waste M5 Max time without meaningful quality gains

Suggested breakdown (~15M total):

**Shared state encoder (~5-8M params)** — does most of the work, both heads consume its 512-dim output:
- Card embeddings: 52 cards × 64-dim (~3K params)
- Card pooling layer (small attention or mean-pool): ~1M params
- Action-history GRU: 2 layers, 256 hidden (~500K params)
- Opponent-embedding aggregator: per-opponent 10 raw stats → 16-dim embedding via small MLP, then mean-pooled across opponents to a fixed 16-dim summary regardless of table size (~500K params for the MLP)
- Concat all of the above + position/SPR/n_opp/pot/stack scalars, MLP body 1024 hidden × 3 layers (~3-5M params)
- Output: 512-dim state vector

**Regret network (~3-4M params):** body 512 → 1024 → 1024 → 1024, output head → NUM_ACTION_TYPES.

**Value network (~3-4M params):** same body shape, output head → 1 scalar (chip-EV).

**Continuous bet-sizing head (~500K params):** branches off the state encoder, 512 → 256 → 1 with sigmoid scaled to [0.0, 2.0]. Only consumed when the regret head selects "raise".

### Memory footprint during training

~60MB for fp32 weights, ~120MB for Adam optimizer state, ~500MB-1GB for batched activations at batch 4096, plus the regret-sample replay buffer (probably 5-10GB for millions of samples). Total working set ~8-15GB. Fits cleanly in 64GB unified memory.

### Build at 15M, smoke-test at 5M

Parametrize the hidden widths so you can flip to a smaller variant (~5M params, half-width MLPs and GRU) for the first end-to-end smoke training run. Validates the data pipeline + training loop without burning 24h on a bug. Once smoke passes, scale back to 15M for the real training run.

### Shared components

Both paths share:
- `core/engine.py` (unchanged from this session's state)
- The equity-shaping training infrastructure (full-information self-play loop, equity-shaped reward computation, ICM transform for winner-take-all)
- The opponent-stat tracker (VPIP, AF, fold-to-cbet rolling stats per opponent)
- The action-history encoder (used as input feature in Path B, used to derive a stat-bucket in Path A)
- The eval harness (500-1000 tournament stats with Wilson CIs)

### Build approach

Build both paths in parallel. No training during the build phase. Both paths get a full sanity-test pass (extended `sanity_cfr_equity.py` for Path A; new `sanity_deep_cfr.py` for Path B) before any training kicks off.

### Training order

**Path B trains first** because it's GPU/RAM-bound and the M5 Max's 40-core GPU + 64GB unified memory is the dominant resource for matmul-heavy neural training (per HARDWARE_BENCHMARK.md, ~15 TFLOPS fp32 sustained, ~8x speedup on large nets). Approximate sequence:

1. Train the equity-shaped value network first (full-info self-play data → value targets). Standalone training, evaluatable on its own.
2. Train the regret network using the value network for leaf evaluation. Deep CFR loop.
3. Save final weights to `models/deep_cfr_v1.pt`.

**Path A trains second** because it's CPU-bound (table lookups, regret accumulation per node visit). M5 Max has 18 cores with ~10x scaling — the right tool for tabular MCCFR. Runs after Path B is done so they don't compete for memory bandwidth. Approximate sequence:

1. Use the equity-shaped value network from Path B as the leaf evaluator (already trained, just load it).
2. Run tabular MCCFR with real tree recursion, populate the regret table.
3. Save final regret table to `models/cfr_regret_deep_v2.pkl`.

### Eval

After both are trained, head-to-head + N-player tournament stats against the existing bot pool to determine which path performs better. The winner becomes the production CFR. The loser stays in the pool as a varied opponent for league play (TRAINING_PLAN.md Step 4).

### The "pocket" bot

`models/cfr_regret_deep.v1_pre_equity_shaping.pkl` is preserved as the FIRST-generation pocket bot (the result of training the new abstraction against the heuristic value function for ~22 hours, before equity shaping). Future use:
- League opponent in Step 4 (intentionally weaker baseline to vary the difficulty curve)
- Reference for "did equity shaping + tree CFR actually help?" — three-way comparison v1 (heuristic) vs v2 (Path A) vs Deep CFR (Path B) is the cleanest test

After Path A and Path B train, expect TWO additional pocket-class artifacts in `models/`:
- `models/cfr_regret_deep_v2.pkl` — Path A's tabular MCCFR with equity shaping + tree CFR + opponent stats + finer buckets
- `models/deep_cfr_v1.pt` — Path B's Deep CFR neural networks

### Realistic timeline

This is a 2-3 month project end-to-end. Each major component is 1-3 weeks of focused work. The build phase (no training) is the biggest chunk; training is overnight runs once the code is right.

## What we explicitly did NOT do (and why)

- **Real tree CFR / opponent strategy sampling.** Universal in research bots (Pluribus, DeepStack, Libratus all do it). Our `_run_iterations` is still one-step lookahead with rollout. Estimated 2-4 weeks of work. Lives in TRAINING_PLAN.md Step 15 ("nuclear" Deep CFR / ReBeL path). Not feasible before this overnight retrain.
- **Real-time search at decision time.** Same bucket as above. Pluribus uses depth-limited subgame solving at play time; we just look up the precomputed strategy. Step 15 territory.
- **Per-opponent stat features in CFR keys.** Step 6 of TRAINING_PLAN.md. Requires opponent-tracking infrastructure CFR doesn't have today. The exploitative bot uses these; CFR doesn't.
- **Concrete betting history (sizes, not tokens).** Step 5. Requires a learned encoder (GRU/transformer), not a key-space fix.
- **Equity-dilution-when-called multiplier.** Both brainstorm agents flagged this. We chose not to add it because it's an unprincipled hack and Step 2 (equity shaping) handles it naturally via realized outcomes.
- **Card buckets above 20.** Diminishing returns vs key-space cost. Can revisit if first retrain feels coarse.
- **Further tuning of `_estimate_action_value` constants.** We have two written-down value-function biases (passive on premiums, slight shove bias on marginals). Both are structural limitations that no constant-tweak fixes — they need the equity-shaping replacement (Step 2). Adding more tuning rounds was explicitly stopped.

## Useful commands for next time

```bash
# Check training progress without stopping it
ls -la models/cfr_regret_deep.pkl

# Inspect the profile contents
python -c "
import pickle
with open('models/cfr_regret_deep.pkl', 'rb') as f:
    d = pickle.load(f)
print(f'info sets: {len(d[\"nodes\"]):,}')
print(f'iterations: {d.get(\"total_iterations\", 0):,}')
"

# Stop training cleanly (saves checkpoint before exiting)
kill <PID>
# or
pkill -f train_cfr_bot_multiway

# Smoke test the trained bot
python run_local_match.py --players smart,cfr,gto --chips 300 \
    --sb 5 --bb 10 --max-hands 200

# Full sanity sweep
python sanity_test_hand.py && \
python sanity_action_order.py && \
python sanity_test_followon.py && \
python sanity_cfr_equity.py
```

## Roll-back if needed

If the new CFR plays worse than the old one and you want to revert to the heads-up-keyed profile:

```bash
cp models/cfr_regret_deep.PRE_MULTIWAY.bak models/cfr_regret_deep.pkl
```

But the code changes (multiway equity, value function rewrite, position/SPR keys) won't be able to use the old profile — every lookup will miss the 3-colon keys and fall through to heuristic. To truly roll back, you'd also need to `git checkout` the pre-session state of `bots/cfr_bot.py`, `core/engine.py`, `core/logger.py`, `training/train_ml_bot.py`, and `run_local_match.py`. The pre-session state is the most recent commit before this session's changes.
