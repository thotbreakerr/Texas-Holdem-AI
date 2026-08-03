> 🏛️ **Coming soon: The Colosseum** — a website where you can plug in your own bots and battle them against others. Stay tuned!

---

# Texas Hold'em Poker AI

A research sandbox for tournament poker agents. The goal is a strong bot for a
single **winner-take-all, 6-player** Texas Hold'em tournament. The repo pairs a
**dual rules engine** (a legacy in-repo engine plus a Poker-Engine adapter) with
a dozen bot families — game-theoretic (CFR / Deep CFR), learned (RL, supervised
ML), heuristic/equity, tournament-aware, and opponent-modeling — self-play and
supervised trainers, an evaluation harness with Wilson confidence intervals, and
a ~40-gate sanity ladder.

> **Status: in-progress research repo, not a finished product.** The engine
> migration (Poker-Engine, P1–P6) is complete and is the default. The headline
> CFR / Deep CFR models are **not yet trained to production** — see
> [Project Status](#project-status). Most components below are usable today; a
> few are scaffolding awaiting a clean training run.

---

## What's inside

- **Dual poker engine** — a tested Poker-Engine rules core (default) behind an
  adapter, with the original in-repo engine retained as a fallback and as the
  shared hand evaluator / primitives everything imports.
- **~13 bot families**, grouped:
  - *Game-theoretic*: CFR (tabular MCCFR), Deep CFR (neural, "Path B").
  - *Learned*: RL (PPO + GAE-λ), supervised ML (26-feature MLP).
  - *Heuristic / equity*: Monte Carlo rollouts, SmartBot hand-tiers, GTO approximation.
  - *Tournament*: hybrid `final_survival` / `final_aggro`, ICM.
  - *Opponent-aware*: exploitative (VPIP/AF/FTA), Bayesian opponent-model.
  - *Stress archetypes* (Phase 7): deliberate caricatures for robustness testing — **not** realistic models.
- **Training pipelines** for ML, RL, CFR, and Deep CFR.
- **Eval harness** (`run_eval.py`) with head-to-head / multiway modes and Wilson 95% CIs.
- **Sanity ladder** — 41 `sanity_*.py` gates, 35 wired into `sanity_validation_ladder.py` (the rest run standalone).
- **Tournament UI** (matplotlib) and a batch statistics runner.

---

## Repository layout

```
.
├── core/                       Rules engines, evaluator, shared game utilities
│   ├── engine.py               LEGACY engine + hand evaluator (2.6M-combo lookup) + primitives
│   ├── pe_engine.py            Poker-Engine adapter (default driver) — wraps the sibling repo
│   ├── engine_factory.py       make_table() / engine selection (pe | legacy)
│   ├── bot_api.py              Action, PlayerView, BotAdapter interfaces
│   ├── equity.py               Shared equity / rollout helpers
│   ├── icm.py                  Independent Chip Model (Malmuth-Harville)
│   ├── tournament.py           Tournament loop / blind escalation / finish order
│   ├── aivat.py                AIVAT variance-reduced leaf evaluation
│   ├── action_history.py       Action tokenization → tensors (Deep CFR / ML)
│   ├── ml_features.py          Single-source feature builder (train == serve)
│   ├── opponent_stats.py       Per-opponent VPIP / AF / FTA tracking
│   ├── table_order.py          Seat / dealer ordering helpers
│   └── logger.py               Per-decision JSONL logger for ML training
│
├── bots/                       Bot implementations + factory (create_bot, parse_players)
│   ├── monte_carlo_bot.py      Monte Carlo rollout equity vs. pot odds
│   ├── poker_mind_bot.py       Heuristic hand-tier bot (SmartBot)
│   ├── poker_mlp.py            Shared MLP network definition (not a playable bot)
│   ├── ml_bot.py               Supervised 26-feature MLP
│   ├── rl_bot.py               PPO with GAE-λ + value network
│   ├── cfr_bot.py              Tabular Monte Carlo CFR (MCCFR)
│   ├── deep_cfr_bot.py         Neural Deep CFR (Path B)
│   ├── icm_bot.py              ICM tournament-equity bot
│   ├── exploitative_bot.py     Opponent-tracking exploitation
│   ├── gto_bot.py              GTO approximation (balanced mixed strategies)
│   ├── opponent_model_bot.py   Bayesian hand-range modeling
│   ├── tournament_hybrid_bot.py Final tournament bot (survival / aggro profiles)
│   └── archetype_bot.py        Phase-7 stress archetypes (maniac, station, nit, …)
│
├── training/                   Training scripts (run as top-level, not a package)
│   ├── train_ml_bot.py         Supervised learning on decision logs
│   ├── train_rl_bot_selfplay.py    Self-play curriculum PPO
│   ├── train_multi_deep_rl_bot.py  Multi-opponent PPO
│   ├── train_cfr_bot_multiway.py   Path A: 6-player tabular CFR
│   └── train_deep_cfr.py       Path B: neural Deep CFR (external sampling)
│
├── sanity_*.py                 ~40 verification gates (engine, CFR, features, …)
├── sanity_validation_ladder.py Orchestrator that runs the gates by tier
│
├── run_tournament.py           Live tournament UI (matplotlib, Play button)
├── run_local_match.py          Single tournament + chip chart → output/
├── run_tournament_stats.py     Batch statistics (multiprocessing)
├── run_eval.py                 Path A vs Path B / multiway eval (Wilson CIs, --engine)
├── eval_final_bot.py           Tournament eval for the hybrid final bot
├── probe_deep_cfr.py           Deep CFR checkpoint action-distribution probe
├── benchmark_m5.py             M5 Max hardware benchmark
│
├── models/                     Weights + precomputed tables (gitignored; ~10 GB local)
│   └── five_card_table.pkl     Precomputed hand evaluator (~46 MB, shared by both engines)
├── logs/                       Auto-generated JSONL decision logs (gitignored)
├── output/                     Tournament charts (.png) and stats (.csv)
├── eval_runs/                  Ablation eval CSVs (manual convention via --output-csv; gitignored)
│
├── docs/                       Plans, reviews, and session logs (see Docs index)
├── PROGRESS.md                 Reverse-chron work log
├── requirements.txt            Python dependencies
└── README.md
```

> **Note on layering.** `core/` and `bots/` are importable packages; `training/`
> is a directory of top-level scripts. The layering is *mostly* downward
> (engine → bots → training/eval), with two deliberate exceptions worth knowing:
> `core/tournament.py` imports `finalize_finish_order` from `run_tournament_stats.py`
> and `escalate_blinds` from `bots`, and Deep CFR traverses its own abstract game
> tree rather than driving a `Table`.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies (floors): PyTorch (>= 1.9), Matplotlib (>= 3.5), treys (>= 0.1.8).
The project `.venv` currently runs Torch 2.x / Matplotlib 3.11. **All commands
below assume the project venv is active** (or prefix them with `.venv/bin/python`).

### Poker-Engine dependency (required for the default engine)

The **default** engine (`pe`) is a thin adapter over a separate **Poker-Engine**
repository that is **not vendored here**. It is resolved, in order, from:

1. `$THAI_POKER_ENGINE_PATH`, if set; otherwise
2. a sibling checkout at `../Poker-Engine` (next to this repo).

If neither is present, table construction under the default engine raises a clear
error. You can avoid the dependency entirely by selecting the legacy engine (see
below), which has no external requirements.

---

## Engine selection

`core/engine_factory.make_table()` builds a table with one of two engines:

| Impl | What it is | Notes |
|------|-----------|-------|
| `pe` (default) | Poker-Engine adapter (`core/pe_engine.py`) | Default since P6 (2026-07-06); ~2.9× lower engine overhead (real-field wall-clock and results on par) |
| `legacy` | Original in-repo engine (`core/engine.py`) | No external deps; still builds the hand evaluator + primitives; backs the sanity suite |

Selection priority is **explicit argument > `THAI_ENGINE_IMPL` env var > default**:

```bash
# Force the legacy engine for a whole process
THAI_ENGINE_IMPL=legacy python run_tournament_stats.py --tournaments 100

# run_eval exposes an explicit flag (default: pe)
python run_eval.py --engine legacy ...
python run_eval.py --engine pe ...
```

The legacy engine is *deprecated as a driver* but intentionally retained: it
still builds the 2,598,960-combo hand evaluator (cached to
`models/five_card_table.pkl`), supplies primitives (`RANKS`, `SUITS`, `Seat`,
`eval_hand`) imported across `core/` and the bots, and most sanity gates run
against it on purpose. `sanity_engine_parity.py` is the legacy-vs-PE parity gate.

---

## Running

**Tournament UI** (live chart, click Play to start):
```bash
python run_tournament.py
```

**Single tournament** (runs to completion, saves a chart to `output/`):
```bash
python run_local_match.py
python run_local_match.py --players "smart,smart,mc100,random" --log-session   # + training logs
```

**Batch statistics** (many tournaments, win rates):
```bash
python run_tournament_stats.py --tournaments 100 --chips 500
```

**Bot evaluation** (Path A vs Path B / multiway, Wilson 95% CIs, engine flag):
```bash
python run_eval.py --mode multiway --tournaments 1000 --engine pe
```
`--mode` also accepts `head_to_head`, `pilot`, `promotion`, and `curriculum` —
`pilot`/`promotion` are the model-promotion gates (with `--path_b_weights` /
`--promotion-opponent`); see [docs/plans/TRAINING_PLAN.md](docs/plans/TRAINING_PLAN.md).

**Final hybrid-bot eval** (tournament eval for `final_survival` / `final_aggro`,
no CFR/DeepCFR models loaded):
```bash
python eval_final_bot.py
```

**Using a specific RL checkpoint** — the `--rl_model` flag rewrites any `rl`
entries in the `--players` spec to that path (RL weights are gitignored, so
supply your own):
```bash
python run_local_match.py --rl_model models/your_rl_model.pt
python run_tournament_stats.py --tournaments 50 --rl_model models/your_rl_model.pt
python run_tournament.py --rl_model models/your_rl_model.pt
```

---

## Player specs

Bots are created via string keys passed to `create_bot()`, or as comma-separated
specs to `parse_players()`. Auto-assigned IDs (`P1`, `P2`, …) or named seats
(`P1=mc200,P2=smart`).

| Key(s) | Bot | Notes |
|--------|-----|-------|
| `mc`, `mc<N>` | MonteCarloBot | Optional sim count: `mc200`, `mc500` (default 200) |
| `smart`, `smartbot`, `heuristic` | SmartBot | Heuristic hand-tier bot |
| `ml`, `mlbot` | MLBot | 26-feature supervised MLP |
| `rl`, `rlbot`, `rl:<path>` | RLBot | Optional model path: `rl:models/custom.pt` |
| `cfr`, `cfrbot`, `cfr:<path>` | CFRBot | **No trained profile ships** — bare `cfr` warns and plays untrained |
| `deep_cfr`, `deepcfr`, `deep_cfr_bot` | DeepCFRBot | No schema-v2 weights produced yet (see Status) |
| `icm`, `icmbot` | ICMBot | Tournament equity (ICM) |
| `exploitative`, `exploitativebot` | ExploitativeBot | VPIP/AF/FTA exploitation |
| `gto`, `gtobot` | GTOBot | GTO approximation |
| `opponentmodel`, `opponentmodelbot` | OpponentModelBot | Bayesian hand-range modeling |
| `final`, `final_survival`, `final_aggro` | TournamentHybridBot | Survival / aggro profiles |
| `final_<profile>:p4\|telemetry\|station\|r2\|p5` | TournamentHybridBot | Ablation arms (Phase 4/5) |
| `random` | RandomBot | Uniform random legal actions |
| `maniac`, `maniac_trigger`, `maniac_mixed`, `overbet_merchant`, `calling_station`, `nit`, `folder`, `loose_passive`, `minraise`, `minraiser`, `baseline_sane`, `pressure_filler` | ArchetypeBot | Phase-7 stress archetypes (deliberate caricatures, not realistic opponents) |

Example: `--players mc200,smart,rl,cfr` → a 4-player table (P1–P4).

---

## Bots

### Monte Carlo Bot
Runs Monte Carlo simulations (default 200) to estimate equity against random
opponent hands, then compares equity to pot odds. Adjusts aggression by table
position (tighter early, looser on the button). The strongest pure heuristic bot.

### SmartBot (Poker Mind Bot)
Classifies hands into tiers (premium pairs, broadway, trash) preflop and uses the
hand evaluator to estimate strength 0–1 postflop. Bets strong hands, checks/calls
medium, folds weak, with a small (~7%) bluff frequency. Fast baseline reference.

### ML Bot
Supervised 3-layer MLP (`PokerMLP`: 26 features, 128 hidden, 6 action classes),
trained on decision logs. Features include hand strength, pot odds, position, and
cumulative per-opponent memory (aggression, tightness, VPIP; checks don't count as
VPIP). Training and inference share one feature builder (`core/ml_features.py`),
so logged decisions and live `PlayerView`s produce identical vectors. Checkpoints
carry a `feature_schema_version`; MLBot **refuses** legacy raw state dicts and
wrong-version checkpoints and falls back to a hand-strength heuristic. Call
`reset_memory()` at each tournament boundary when reusing an instance.

### RL Bot
PPO with Generalized Advantage Estimation (GAE-λ). A 512-unit policy network with
dropout plus a separate 512-unit value network (critic). Same 26-feature input as
the ML bot. Rewards are normalized chip deltas with terminal win/loss bonuses;
exploration is fixed at 10% during training. Trained via the two RL scripts below.

### CFR Bot (Path A)
Game-theoretic bot using Monte Carlo Counterfactual Regret Minimization (MCCFR).
In this six-player, abstracted, decision-rooted setup it *approximates* — but does
not provably converge to — equilibrium play (CFR's Nash guarantee is for
two-player zero-sum). Maintains a persistent tabular regret store.

- **Card abstraction**: 50 preflop buckets (strength tiers), 50 postflop buckets (MC equity percentiles, 200 rollouts).
- **Bet abstraction**: 8 abstract actions (fold, check/call, 33% / 50% / 67% / 75% / 100% pot, all-in).
- **Action history**: compressed to one-char action tokens (F/K/C/S/Q/M/L/P/A, variable length) for info-set keys.
- **Inference mode**: skips online regret updates so loaded strategies aren't corrupted during play.

The intended default profile path is `models/cfr_regret_deep_v2.pkl` (produced by
`train_cfr_bot_multiway.py`, gitignored). **No such file has been trained yet** —
bare `cfr` currently prints a warning and plays untrained.

### Deep CFR Bot (Path B)
Neural Deep CFR. Independent advantage, average-strategy, value, and bet-sizing
encoder/head networks; zero-initialized advantage output; player-count features;
reservoir buffers. Traverses its own abstract game tree via external sampling
rather than driving a `Table`. The intended default artifact `models/deep_cfr_v2.pt`
**has not been produced** (the 150k pilot was aborted at 100k — see Status).

### ICM Bot
Malmuth-Harville Independent Chip Model. Converts stacks into tournament equity
and maximizes equity preservation over raw chip EV — aggressive with a big stack,
tight when its own stack is at risk.

### Exploitative Bot
Tracks per-opponent VPIP, aggression factor (AF), and fold-to-aggression (FTA).
Falls back to tight-aggressive defaults until 5+ hands of history, then exploits:
bluffs high-FTA players, value-bets calling stations, traps hyper-aggressors.

### GTO Bot
Position-aware preflop range charts (early/mid/late/blinds) and balanced mixed
strategies postflop, targeting a 2:1 river value-to-bluff ratio. Non-deterministic
by design.

### Opponent Model Bot
Bayesian hand-range modeling: a probability distribution over five strength buckets
per opponent, updated via likelihood multipliers from observed actions. Runs Monte
Carlo equity against the weighted range rather than random hands.

### Tournament Hybrid Bot
The `final_survival` / `final_aggro` bot for the winner-take-all tournament, with
Phase 4/5 ablation arms (`:p4`, `:telemetry`, `:station`, `:r2`, `:p5`). See
[docs/plans/CLASS_TOURNAMENT_BOT_PLAN.md](docs/plans/CLASS_TOURNAMENT_BOT_PLAN.md).

### Archetype Bots (Phase 7)
Deliberate stress caricatures (maniac, calling station, nit, over-folder, …) used
to probe robustness. **Not** realistic opponent models — don't read their results
as real-field claims. See [docs/plans/PHASE7_STRESS_OPPONENTS_PLAN.md](docs/plans/PHASE7_STRESS_OPPONENTS_PLAN.md).

---

## Training

Model artifacts land in `models/` (gitignored). The CFR/Deep-CFR default paths
below are **intended outputs, not shipped files**.

### Pre-training validation gate
Before any clean CFR / Deep CFR retrain, run the canonical tiered gate. It wires
the standalone `sanity_*.py` scripts into one ladder (engine truth → abstraction →
feature schema → chip accounting → optional smoke training → eval readiness) and
exits nonzero on failure.

```bash
.venv/bin/python sanity_validation_ladder.py --path deep-cfr    # fast/medium gates
.venv/bin/python sanity_validation_ladder.py --path cfr
.venv/bin/python sanity_validation_ladder.py --path both
.venv/bin/python sanity_validation_ladder.py --path both --full        # + slow smoke-train/eval
.venv/bin/python sanity_validation_ladder.py --path both --keep-going  # don't stop at first failure
```

See [docs/plans/TRAINING_PLAN.md](docs/plans/TRAINING_PLAN.md) for when/how to run before a retrain.

### RL training
Two scripts share the same PPO loop, GAE-λ, and CLI.

- **`train_rl_bot_selfplay.py`** — self-play curriculum (random → heuristic → self-play), snapshots every 500 episodes.
- **`train_multi_deep_rl_bot.py`** — multi-player PPO against CFR, Monte Carlo, and GTO opponents with random seating.

```bash
python training/train_rl_bot_selfplay.py --episodes 50000
python training/train_multi_deep_rl_bot.py --episodes 50000
```

### CFR training (Path A)
**`train_cfr_bot_multiway.py`** — 6-player deep-stack training (1000 chips, 5/10
blinds, 1.5× escalation every 50 hands). Six CFR instances share one regret table.
Saves atomically; checkpoints on `KeyboardInterrupt`.

```bash
python training/train_cfr_bot_multiway.py --tournaments 100000 --iterations 200 \
    --profile models/cfr_regret_deep_v2.pkl
```

### Deep CFR training (Path B)
**`train_deep_cfr.py`** — external-sampling traversals over the bot's internal
abstract tree, round-boundary advantage refit, curriculum over 2–6 players,
resumable checkpoints, and probability-based collapse canaries (SIGINT-safe).

```bash
python training/train_deep_cfr.py --variant large --iterations 1000000 \
    --curriculum-profile sixmax --canary-enforce-iteration 100000 \
    --save-path models/deep_cfr_v2.pt --device auto
```

> **Known caveat:** `TRAINING_PLAN.md`'s MCCFR time estimates are off — see
> [docs/reviews/TRAINING_RUN_REVIEW_2026-07-01.md](docs/reviews/TRAINING_RUN_REVIEW_2026-07-01.md).

### ML training
**`train_ml_bot.py`** — supervised learning on JSONL decision logs (Adam,
ReduceLROnPlateau, 80/20 split). Requires **session-scoped** logs — generate them
with `run_local_match.py --log-session` (one file per tournament), or from custom
scripts by passing `DecisionLogger(session_scoped=True)` to
`Table.play_hand(..., logger=...)`. Files without a `session_start` header
are rejected unless `--allow-legacy-logs` is passed. Checkpoints embed
`feature_schema_version` so stale models can't be silently loaded.

```bash
python run_local_match.py --players "smart,smart,mc100,random" --log-session   # data
python training/train_ml_bot.py --log_dir logs --epochs 8                      # train
```

---

## Testing / sanity harness

The repo's correctness is guarded by 41 `sanity_*.py` gates and the
`sanity_validation_ladder.py` orchestrator (35 gates are wired into the ladder;
the rest, including `sanity_engine_parity.py`, run standalone). These are **gates**, not pytest unit
tests: many print `ALL CHECKS PASSED` / `SOME CHECKS FAILED` and exit nonzero on
failure. Run them from the **repo root** (the ladder invokes each gate by filename
with `cwd=` repo root).

```bash
.venv/bin/python sanity_validation_ladder.py --path both       # the canonical gate
.venv/bin/python sanity_engine_parity.py                       # legacy vs Poker-Engine parity
.venv/bin/python sanity_deep_cfr.py                            # Deep CFR structural gate
```

Most gates run against the **legacy** engine on purpose (`engine_impl='legacy'`)
so they test the reference rules; `sanity_engine_parity.py` is what proves the
Poker-Engine adapter matches it.

---

## Adding a bot

Create a file in `bots/` and implement `act()`:

```python
from core.bot_api import Action, PlayerView

class MyBot:
    def act(self, state: PlayerView) -> Action:
        # state has: hole_cards, board, pot, to_call, legal_actions,
        #            stacks, position, history, etc.
        return Action("call")
```

Then register a key → import mapping in `create_bot()` in `bots/__init__.py`.

---

## Known limitations

- **No web UI** — visualization is matplotlib-only (local).
- **CFR abstraction** — 50 buckets per street, 200 equity rollouts. Reasonable coverage but well below research-grade systems (Pluribus uses thousands of buckets per street).
- **CFR value function (`_estimate_action_value`) is heuristic, not learned.** Two known structural biases (slight passive bias on premium hands, slight shove bias on marginals). The proper fix is the equity-shaped learned value function in TRAINING_PLAN.md Step 2.
- **CFR uses one-step lookahead with rollout, not real tree CFR.** No recursive opponent-strategy sampling (the "nuclear" Step 15).
- **No real-time search at decision time** — we look up the precomputed strategy rather than doing depth-limited subgame solving.
- **RL exploration is fixed at 10%** — a decay schedule would help late-stage sharpening (Step 7).
- **No per-opponent stat features in CFR keys** — CFR plays the same strategy against a maniac and a nit (Step 6).
- **Concrete bet sizes lost in history abstraction** — sizes compress to tokens; a learned encoder over the raw action sequence would fix this (Step 5).

### Recently fixed
- ~~No position encoding in CFR info-sets~~ — fixed 2026-04-26.
- ~~CFR equity heads-up-only~~ — fixed 2026-04-26 (proper multiway rollouts).
- ~~ML bot BB position-encoding mismatch~~ — fixed (shared feature builder).

---

## Project status

*Snapshot as of 2026-07-07. The full running log is in [PROGRESS.md](PROGRESS.md).*

- **Engine migration — COMPLETE.** Poker-Engine (P1–P6) is merged to `main` and is
  the default (`DEFAULT_ENGINE_IMPL = "pe"`). Legacy remains reachable and backs
  the sanity suite. Parity is mutation-tested; the sanity ladder is green under
  both engines. See [docs/plans/ENGINE_MIGRATION_PLAN.md](docs/plans/ENGINE_MIGRATION_PLAN.md).
- **CFR / Deep CFR — NOT yet trained to production.** The 150k Path-B pilot was
  **aborted at the 100k gate** (root cause: advantage-refit underfit — the fix is
  a real ~2–4k fit-step budget plus fit-quality logging, since diagnosed). No
  schema-v2 artifact (`deep_cfr_v2.pt`) exists yet; `cfr_regret_deep_v2.pkl` has
  not been produced. See [docs/reviews/TRAINING_RUN_REVIEW_2026-07-01.md](docs/reviews/TRAINING_RUN_REVIEW_2026-07-01.md).
- **Tournament hybrid bot (Phases 3–7) — implemented and evaluated.** Winner-take-all
  logic, opponent tendencies, and stress-opponent robustness are in place.

---

## Docs index

**Living / master docs**
- [PROGRESS.md](PROGRESS.md) — reverse-chronological work log.
- [docs/plans/TRAINING_PLAN.md](docs/plans/TRAINING_PLAN.md) — master training strategy.
- [docs/plans/CLASS_TOURNAMENT_BOT_PLAN.md](docs/plans/CLASS_TOURNAMENT_BOT_PLAN.md) — hybrid-bot master plan.
- [docs/plans/HARDWARE_BENCHMARK.md](docs/plans/HARDWARE_BENCHMARK.md) — M5 Max sizing.
- [docs/trainer_reward_mode_design.md](docs/trainer_reward_mode_design.md) — forward-looking design seam.

**Historical / completed artifacts** (kept for provenance)
- Plans: [docs/plans/ENGINE_MIGRATION_PLAN.md](docs/plans/ENGINE_MIGRATION_PLAN.md), [PHASE4_TOURNAMENT_LOGIC_PLAN.md](docs/plans/PHASE4_TOURNAMENT_LOGIC_PLAN.md), [PHASE5_OPPONENT_TENDENCIES_PLAN.md](docs/plans/PHASE5_OPPONENT_TENDENCIES_PLAN.md), [PHASE7_STRESS_OPPONENTS_PLAN.md](docs/plans/PHASE7_STRESS_OPPONENTS_PLAN.md).
- Reviews / audits: [docs/reviews/](docs/reviews/) — `REVIEW_*`, `PHASE3_AUDIT_*`, `PHASE4_DEEPCFR_*`, `FIX_REPORT_*`, `TRAINING_RUN_REVIEW_*`.
- Session logs: [docs/sessions/](docs/sessions/).
