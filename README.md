
# Texas Hold'em Bot

A poker engine with pluggable AI bots. Ships with tournament, heuristic, neural-network, reinforcement-learning, game-theoretic, and opponent-modeling strategies, plus a live tournament UI, batch statistics runner, and training pipelines for the ML, RL, and CFR bots.

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: PyTorch (>= 1.9), Matplotlib (>= 3.5), treys (>= 0.1.8).

## Running

**Tournament UI** (live chart, click Play to start):
```bash
python run_tournament.py
```

**Single tournament** (runs to completion, saves a chart to `output/`):
```bash
python run_local_match.py
```

**Batch statistics** (run many tournaments, report win rates):
```bash
python run_tournament_stats.py --tournaments 100 --chips 500
```

**Testing a specific RL model checkpoint**:
```bash
python run_local_match.py --rl_model models/rl_model_run3.pt
python run_tournament_stats.py --tournaments 50 --rl_model models/rl_model_run3.pt
python run_tournament.py --rl_model models/rl_model_run3.pt
```

The `--rl_model` flag automatically rewrites any `rl` entries in the `--players` spec to use the specified model path.

## Project Structure

```
.
├── core/                          Game engine, bot interface, decision logger
│   ├── engine.py                  Full hand lifecycle, hand evaluator, pot distribution
│   ├── bot_api.py                 Action, PlayerView, BotAdapter interfaces
│   └── logger.py                  Per-decision JSONL logger for ML training
│
├── bots/                          Bot implementations + factory
│   ├── __init__.py                Bot factory (create_bot, parse_players, escalate_blinds)
│   ├── monte_carlo_bot.py         Monte Carlo rollout equity estimation
│   ├── poker_mind_bot.py          Heuristic hand-tier classification (SmartBot)
│   ├── ml_bot.py                  Supervised learning (26-feature MLP)
│   ├── rl_bot.py                  PPO with GAE-lambda and value network
│   ├── cfr_bot.py                 Monte Carlo CFR (MCCFR, regret matching)
│   ├── icm_bot.py                 Tournament equity (Independent Chip Model)
│   ├── exploitative_bot.py        Opponent-tracking exploitation
│   ├── gto_bot.py                 GTO approximation with balanced mixed strategies
│   ├── opponent_model_bot.py      Bayesian hand-range modeling
│   └── tournament_hybrid_bot.py   Final tournament bot profiles
│
├── models/                        Saved weights and precomputed tables
│   └── five_card_table.pkl        Precomputed hand evaluator lookup (~45 MB)
│
├── training/                      Training scripts for ML, RL, and CFR bots
│   ├── train_ml_bot.py                Supervised learning on decision logs
│   ├── train_rl_bot_selfplay.py       Self-play curriculum (random -> heuristic -> self-play)
│   ├── train_multi_deep_rl_bot.py     Multi-player PPO (CFR + MC + GTO opponents)
│   └── train_cfr_bot_multiway.py      6-player deep-stack CFR training
│
├── logs/                          Auto-generated JSONL decision logs
├── output/                        Tournament charts (.png) and stats (.csv)
├── data/                          Training datasets (currently empty)
│
├── run_tournament.py              Live tournament UI (matplotlib, Play button)
├── run_local_match.py             Single tournament runner with chart output
├── run_tournament_stats.py        Batch statistics with multiprocessing
└── requirements.txt               Python dependencies
```

## Core Engine

The game engine (`core/engine.py`) handles the full hand lifecycle: blinds, betting rounds (preflop through river), street transitions, showdowns, side pots, and pot distribution. It includes a pure-Python hand evaluator backed by a precomputed lookup table covering all 2,598,960 five-card combinations (~45 MB, built once and cached to `models/five_card_table.pkl`).

`core/bot_api.py` defines the three interfaces all bots implement: `Action` (type + optional amount), `PlayerView` (read-only game state without opponent hole cards), and `BotAdapter` (requires `act(state) -> Action`).

`core/logger.py` writes per-decision JSONL logs used for ML training. Each entry captures hand ID, player, position, hole cards, board, pot, action chosen, and legal actions. For training data, create one **session-scoped** logger per tournament (`DecisionLogger(session_scoped=True)` passed to `Table.play_hand(..., logger=...)`, or simply `run_local_match.py --log-session`) — this writes a `session_start` header row that marks the file as one full tournament, which `train_ml_bot.py` requires for cumulative opponent-memory features. Per-hand logs (the engine's internal fallback) carry no header and are rejected for training by default.

## Player Specs

Bots are created via string keys passed to `create_bot()` or as comma-separated specs to `parse_players()`:

| Key | Bot | Notes |
|-----|-----|-------|
| `mc`, `mc<N>` | MonteCarloBot | Optional sim count: `mc200`, `mc500` (default 200) |
| `smart` | SmartBot | Also accepts `smartbot`, `heuristic` |
| `ml` | MLBot | Also accepts `mlbot` |
| `rl`, `rl:<path>` | RLBot | Optional model path: `rl:models/custom.pt` |
| `cfr` | CFRBot | Loads `models/cfr_regret_deep_v2.pkl` in inference mode |
| `deep_cfr` | DeepCFRBot | Loads schema-v2 `models/deep_cfr_v2.pt` in inference mode |
| `icm` | ICMBot | Also accepts `icmbot` |
| `exploitative` | ExploitativeBot | Also accepts `exploitativebot` |
| `gto` | GTOBot | Also accepts `gtobot` |
| `opponentmodel` | OpponentModelBot | Also accepts `opponentmodelbot` |
| `final`, `final_survival` | TournamentHybridBot | Survival profile |
| `final_aggro` | TournamentHybridBot | Aggro profile |
| `random` | RandomBot | Uniform random legal actions |

Example: `--players mc200,smart,rl,cfr` creates a 4-player table with auto-assigned IDs (P1-P4). Named seats: `--players P1=mc200,P2=smart`.

## Bots

### Monte Carlo Bot

Runs Monte Carlo simulations (default 200) to estimate equity against random opponent hands, then compares equity to pot odds. Adjusts aggression thresholds by table position (tighter early, looser on the button). No learning required, just brute-force probability. The strongest pure heuristic bot.

### SmartBot (Poker Mind Bot)

A heuristic bot that classifies hands into tiers (premium pairs, broadway cards, trash) preflop and uses the hand evaluator to estimate strength on a 0-1 scale postflop. Plays accordingly: bet strong hands, check/call medium ones, fold weak ones. Has a small bluff frequency (~7%) built in. Fast baseline reference.

### ML Bot

Supervised learning bot using a 3-layer feedforward network (PokerMLP: 26 input features, 128 hidden units, 6 output action classes). Trained on decision logs from other bots. Features include hand strength, pot odds, position, and opponent memory (cumulative per-opponent aggression, tightness, and VPIP across the tournament; checks do not count as VPIP). Training and inference share one feature builder (`core/ml_features.py`), so logged decisions and live `PlayerView`s produce identical vectors. Falls back to a hand-strength heuristic when the model is untrained or confidence is low.

Checkpoints carry a `feature_schema_version` marker; MLBot **refuses** legacy raw state dicts and wrong-version checkpoints (they were trained with incompatible feature semantics) and falls back to the heuristic. When reusing one MLBot instance across tournaments, call `reset_memory()` at each tournament boundary.

```bash
# 1. Generate session-scoped training logs (one .jsonl per tournament)
python run_local_match.py --players "smart,smart,mc100,random" --log-session

# 2. Train on them
python training/train_ml_bot.py --log_dir logs --epochs 8
python training/train_ml_bot.py --log_dir logs --filter_players P3
python training/train_ml_bot.py --log_dir logs --filter_winners
```

### RL Bot

Reinforcement learning bot using Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE-lambda). Uses a 512-unit policy network with dropout and a separate 512-unit value network (critic). Same 26-feature input as the ML bot. Rewards are normalized chip deltas for proportional credit assignment, with terminal bonuses for wins/losses. Exploration rate is fixed at 10% during training. Supports four training modes via separate scripts (see Training Scripts below).

```bash
python training/train_rl_bot_selfplay.py --episodes 50000
python training/train_multi_deep_rl_bot.py --episodes 50000
```

### CFR Bot

Game-theoretic bot using Monte Carlo Counterfactual Regret Minimization (MCCFR). Iteratively reduces regret across sampled game trajectories. (In this six-player, abstracted, decision-rooted setup that approximates — but does not provably converge to — equilibrium play; CFR's Nash guarantee applies to two-player zero-sum games without such approximations.) Maintains a persistent regret table (not a neural network) that updates across hands.

Key design details:
- **Card abstraction**: 10 preflop buckets (hand-strength tiers) and 10 postflop buckets (Monte Carlo equity percentiles from 20 rollouts).
- **Bet abstraction**: 6 abstract actions (fold, check/call, 33% pot, 67% pot, pot, all-in).
- **Action history**: compressed into 8-character tokens (F/K/C/S/M/P/A) for information-set keys.
- **Regret table**: persisted to disk between sessions so the strategy improves over multiple runs.
- **Inference mode**: skips online regret updates to avoid corrupting loaded strategies during play.

The active profile is `cfr_regret_deep_v2.pkl` (Gate 2A 7-field info-set keys, 6-player multiway, used by default at inference). It is generated by `train_cfr_bot_multiway.py` and gitignored.

### ICM Bot

Tournament equity-aware bot using Malmuth-Harville Independent Chip Model (ICM) calculations. Converts chip stacks into tournament equity (prize shares) and makes decisions that maximize equity preservation rather than raw chip EV. Plays aggressively with a large stack and tightens up when its own stack is at risk.

### Exploitative Bot

Adapts mid-session by tracking per-opponent statistics: VPIP, aggression factor (AF), and fold-to-aggression rate (FTA). Falls back to tight-aggressive (TAG) defaults until it has 5+ hands of history on an opponent, then exploits detected tendencies: bluffs against high-FTA players, value-bets against calling stations, and traps against hyper-aggressors.

### GTO Bot

Approximates Game Theory Optimal play using position-aware preflop hand-range charts (early, mid, late, blinds) and balanced mixed strategies postflop. Targets a 2:1 value-to-bluff ratio on the river. Continuation-bet frequency (60-70%), check-raise frequency (12-18%), and probe bets are all tuned for balance. Non-deterministic by design.

### Opponent Model Bot

Bayesian hand-range modeling. Maintains a probability distribution over five hand-strength buckets (trash, weak, medium, strong, premium) per opponent and updates via likelihood multipliers from observed actions. Runs Monte Carlo equity against the weighted opponent range rather than random hands for more accurate pot-odds calculations as the hand progresses.

## Training Scripts

### Pre-training validation gate

Before any clean CFR / Deep CFR retrain, run the canonical tiered gate. It wires
the standalone `sanity_*.py` scripts into one ladder (engine truth → abstraction →
feature schema → chip accounting → optional smoke training → eval readiness) and
exits nonzero if any selected gate fails.

```bash
.venv/bin/python sanity_validation_ladder.py --path deep-cfr   # fast/medium gates
.venv/bin/python sanity_validation_ladder.py --path cfr
.venv/bin/python sanity_validation_ladder.py --path both
.venv/bin/python sanity_validation_ladder.py --path both --full        # + slow smoke-train/eval (~11 min on M5 Max)
.venv/bin/python sanity_validation_ladder.py --path both --keep-going  # don't stop at first failure
```

Default mode skips the slow gates (and prints how to enable them); `--full` adds
them. See `TRAINING_PLAN.md` for when/how to run before a retrain.

### RL Training

Two scripts train the RL bot. Both share the same PPO update loop, GAE-lambda, and CLI arguments.

**train_rl_bot_selfplay.py** -- Self-play curriculum: random, heuristic, self-play (skips Monte Carlo entirely). Gracefully handles checkpoint loading across architecture changes. Saves snapshots every 500 episodes during self-play.

**train_multi_deep_rl_bot.py** -- Multi-player PPO. Pits the RL bot against CFR, Monte Carlo (200 sims), and GTO opponents simultaneously with random seat assignment each episode. No curriculum stages. Best used after the bot has a solid foundation from the self-play script.

### CFR Training

**train_cfr_bot_multiway.py** -- 6-player deep-stack training (1000 chips, 5/10 blinds, 1.5x escalation every 50 hands). Six CFR instances share one regret table. Saves to `models/cfr_regret_deep_v2.pkl`. Saves atomically (via .tmp + os.replace) and handles `KeyboardInterrupt` by checkpointing on exit.

```bash
python training/train_cfr_bot_multiway.py
python training/train_cfr_bot_multiway.py --tournaments 100000 --iterations 200
python training/train_cfr_bot_multiway.py --profile models/cfr_deep_v2.pkl
```

### ML Training

**train_ml_bot.py** -- Supervised learning on JSONL decision logs. Trains PokerMLP with Adam optimizer, ReduceLROnPlateau scheduler, 80/20 train/val split. Supports filtering by player (`--filter_players`) or winning hands only (`--filter_winners`). Requires **session-scoped** decision logs in `logs/` — generate them with `run_local_match.py --log-session` (one tournament per invocation, one file per tournament). Files without a `session_start` header (legacy per-hand logs) are rejected unless `--allow-legacy-logs` is passed, because cumulative opponent-memory features cannot be reconstructed from them. Saved checkpoints embed `feature_schema_version` so stale models cannot be silently loaded by MLBot.

> **Migration note (2026-06-10, Phase 2/2.1):** decision logs and ML checkpoints created before the shared feature builder (`core/ml_features.py`) are obsolete. Old logs lack position and session headers; old checkpoints (`models/ml_model*.pt` saved as raw state dicts) were trained with mismatched feature semantics and are refused at load time. Regenerate logs with `--log-session` and retrain.

```bash
python run_local_match.py --players "smart,smart,mc100,random" --log-session   # data
python training/train_ml_bot.py --log_dir logs --epochs 8                      # train
```

## Adding a Bot

Create a file in `bots/` and implement `act()`:

```python
from core.bot_api import Action, PlayerView

class MyBot:
    def act(self, state: PlayerView) -> Action:
        # state has: hole_cards, board, pot, to_call, legal_actions,
        #            stacks, position, history, etc.
        return Action("call")
```

Then register it in `bots/__init__.py` by adding a key-to-import mapping in `create_bot()`.

## Known Limitations

- **No web UI** -- visualization is matplotlib-only (local).
- **CFR abstraction** -- 20 buckets per street and 100 equity rollouts. Reasonable coverage for a hobby bot but well below research-grade systems (Pluribus uses thousands of buckets per street). Could go finer at the cost of training time.
- **CFR value function (`_estimate_action_value`) is heuristic, not learned.** Has two known structural biases: slight passive bias on premium hands (KK/QQ ranking `bet_33` above larger sizes in some spots) and slight shove bias on marginals. No constant-tweak fixes both. The proper replacement is the equity-shaped learned value function in TRAINING_PLAN.md Step 2.
- **CFR uses one-step lookahead with rollout, not real tree CFR.** Pluribus, DeepStack, and Libratus all do recursive opponent-strategy sampling. Ours doesn't. This is the "nuclear" Step 15 in TRAINING_PLAN.md.
- **No real-time search at decision time.** Strong research bots refine the precomputed strategy via depth-limited subgame solving at play time. We just look up the precomputed strategy.
- **Exploration decay** -- the RL bot's exploration rate (10%) is fixed; a decay schedule would help the bot sharpen its play in later training stages. (TRAINING_PLAN.md Step 7.)
- **No per-opponent stat features in CFR keys.** The exploitative bot tracks VPIP/AF/FTA per opponent; CFR plays the same strategy against a maniac and a nit. (TRAINING_PLAN.md Step 6.)
- **Concrete bet sizes lost in history abstraction.** The history is compressed into 6 size tokens (S/Q/M/L/P/A); exact dollar amounts are gone. A learned encoder over the raw action sequence would fix this. (TRAINING_PLAN.md Step 5.)

### Known Limitations recently fixed

- ~~No position encoding in CFR info-sets~~ -- fixed 2026-04-26. Position bucket (early/middle/late/blinds) now in info-set key.
- ~~ML bot feature alignment~~ -- the BB position encoding mismatch (`0.5` in training vs `0.3` in inference) is fixed. Other minor mismatches may still exist.
- ~~CFR equity is heads-up-only~~ -- fixed 2026-04-26. `_quick_equity` and `_postflop_bucket` now do proper multiway rollouts with `n_opponents`.

See `SESSION_LOG_2026-04-26.md` for the full change list and what's running now.

## Current CFR status (as of 2026-06-13)

**Path B schema v2 is ready for a fresh retrain.** The collapsed v1 checkpoint
is postmortem-only and is rejected for deployment or resume.

### Closed gates

- **Gate 1 — Shared utilities.** `core/equity.py`, `core/icm.py`, `core/aivat.py`, `core/opponent_stats.py`, `core/action_history.py`. Both paths consume these read-only.
- **Gate 2A — Path A maxed CFR** (`bots/cfr_bot.py` extension). Recursive tree CFR via `_cfr_recurse`, AIVAT leaf evaluator, opponent-stat bucket in info-set key (7 fields, 6 colons), card buckets 20→50, real-time depth-3 subgame search.
- **Gate 2B — Path B multiway Deep CFR-inspired architecture** (`bots/deep_cfr_bot.py`). Independent advantage, average-strategy, value, and sizing encoder/head networks; zero-initialized advantage output; seated/active player-count features; four reservoirs.
- **Gate 3A — Path A training pipeline** (`training/train_cfr_bot_multiway.py` polish + `sanity_train_cfr.py`). Default profile path is now `cfr_regret_deep_v2.pkl` to preserve v1 pocket bot.
- **Gate 3B — Path B schema-v2 training pipeline** (`training/train_deep_cfr.py` + `sanity_train_deep_cfr.py`). Frozen 25k-traversal external-sampling rounds, full traverser-action expansion, average-strategy deployment, sixmax-weighted 2-6 player curriculum, complete resumable checkpoints, and probability-based collapse canaries.
- **Eval harness** (`run_eval.py` + `sanity_eval.py`). Head-to-head and multiway modes, Wilson 95% CIs, decisive-winner verdict logic.

### Pre-existing artifacts

- `models/cfr_regret_deep.v1_pre_equity_shaping.pkl` — the v1 pocket bot from the 2026-04-26 retrain, preserved as a baseline / league opponent.
- `models/cfr_regret_deep.pkl` — same v1 file. Pointing `inference_mode=True` at this path now raises `RuntimeError` (0 valid keys after Gate 2A filter). Safe to keep on disk as a regression artifact or delete.

### Training kickoffs (user-driven)

```bash
# Path B first (GPU/RAM-bound, M5 Max):
python training/train_deep_cfr.py --variant large --iterations 1000000 \
    --curriculum-profile sixmax --canary-enforce-iteration 100000 \
    --canary-fail-patience 3 --save-path models/deep_cfr_v2.pt --device auto

# Path A second (CPU-bound, never concurrent with Path B):
python training/train_cfr_bot_multiway.py --tournaments 5000 --iterations 10 \
    --save_every 500 --profile models/cfr_regret_deep_v2.pkl
```

Schema-v2 rollout:

```bash
# 10k smoke, then a fresh 150k pilot.
python training/train_deep_cfr.py --variant large --iterations 10000 \
    --save-path models/deep_cfr_v2_smoke.pt --device auto
python training/train_deep_cfr.py --variant large --iterations 150000 \
    --save-path models/deep_cfr_v2_pilot.pt --device auto

# Six-player, fixed-seed, seat-rotated pilot against five existing bots.
python run_eval.py --mode pilot --tournaments 500 --seed 20260613 \
    --path_b_weights models/deep_cfr_v2_pilot.pt

# Final canary-clean promotion check against the strongest existing bot.
python run_eval.py --mode promotion --tournaments 1000 --seed 20260613 \
    --promotion-opponent gto --path_b_weights models/deep_cfr_v2.pt
```

After both train, run `python run_eval.py --mode multiway --tournaments 1000 ...` to determine which path becomes production CFR.
