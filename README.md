> 🏛️ **Coming soon: The Colosseum** — a website where you can plug in your own bots and battle them against others. Stay tuned!

---

# Texas Hold'em Poker AI

A research sandbox for tournament poker agents. The goal is a strong bot for a
single **winner-take-all, 6-player** Texas Hold'em tournament. The repo pairs a
**dual rules engine** (a legacy in-repo engine plus a Poker-Engine adapter) with
a pool of heuristic, equity-based, tournament-aware, and opponent-modeling bots,
plus the tournament tooling around them: a live matplotlib UI, a batch
statistics runner, a hybrid-bot evaluation harness, and a sanity-gate test suite.

> **Status: engine + bot-arena repo.** The engine migration (Poker-Engine,
> P1–P6) is complete and is the default. The learning agents (MCCFR, Deep CFR,
> PPO, supervised ML) that were developed here have moved to a private research
> lab — see [Project status](#project-status). What remains is fully usable
> today: the engine, the baseline bot pool, and the tournament tooling that
> will power **The Colosseum**.

---

## What's inside

- **Dual poker engine** — a tested Poker-Engine rules core (default) behind an
  adapter, with the original in-repo engine retained as a fallback and as the
  shared hand evaluator / primitives everything imports.
- **9 bot families**, grouped:
  - *Heuristic / equity*: Monte Carlo rollouts, SmartBot hand-tiers, GTO approximation.
  - *Tournament*: hybrid `final_survival` / `final_aggro`, ICM.
  - *Opponent-aware*: exploitative (VPIP/AF/FTA), Bayesian opponent-model.
  - *Stress archetypes* (Phase 7): deliberate caricatures for robustness testing — **not** realistic models.
  - *Baseline*: uniform-random.
- **Tournament tooling** — live matplotlib UI, single-match runner with chip
  charts, multiprocessing batch statistics, and a hybrid-bot eval harness.
- **AIVAT** (`core/aivat.py`) — variance-reduced hand-value evaluation.
- **Sanity suite** — 14 standalone `sanity_*.py` gates covering engine rules,
  chip accounting, tournament logic, and bot behavior.
- **M5 Max hardware benchmark.**

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
│   ├── opponent_stats.py       Per-opponent VPIP / AF / FTA tracking
│   ├── table_order.py          Seat / dealer ordering helpers
│   └── logger.py               Per-decision JSONL logger
│
├── bots/                       Bot implementations + factory (create_bot, parse_players)
│   ├── monte_carlo_bot.py      Monte Carlo rollout equity vs. pot odds
│   ├── poker_mind_bot.py       Heuristic hand-tier bot (SmartBot)
│   ├── icm_bot.py              ICM tournament-equity bot
│   ├── exploitative_bot.py     Opponent-tracking exploitation
│   ├── gto_bot.py              GTO approximation (balanced mixed strategies)
│   ├── opponent_model_bot.py   Bayesian hand-range modeling
│   ├── tournament_hybrid_bot.py Final tournament bot (survival / aggro profiles)
│   ├── archetype_bot.py        Phase-7 stress archetypes (maniac, station, nit, …)
│   └── punisher.py             Pure preflop core for tag_punisher / wide_defender (+ equity table)
│
├── sanity_*.py                 14 standalone verification gates (engine, tournament, bots)
│
├── run_tournament.py           Live tournament UI (matplotlib, Play button)
├── run_local_match.py          Single tournament + chip chart → output/
├── run_tournament_stats.py     Batch statistics (multiprocessing)
├── eval_final_bot.py           Tournament eval for the hybrid final bot
├── benchmark_m5.py             M5 Max hardware benchmark
│
├── models/                     Precomputed tables (gitignored)
│   └── five_card_table.pkl     Precomputed hand evaluator (~46 MB, shared by both engines)
├── logs/                       Auto-generated JSONL decision logs (gitignored)
├── output/                     Tournament charts (.png) and stats (.csv)
├── eval_runs/                  Ablation eval CSVs (manual convention via --output-csv; gitignored)
│
├── docs/                       Plans (see Docs index)
├── requirements.txt            Python dependencies
└── README.md
```

> **Note on layering.** `core/` and `bots/` are importable packages; the
> runners are top-level scripts. The layering is *mostly* downward
> (engine → bots → runners), with one deliberate exception worth knowing:
> `core/tournament.py` imports `finalize_finish_order` from
> `run_tournament_stats.py` and `escalate_blinds` from `bots`.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies (floors): PyTorch (>= 1.9), Matplotlib (>= 3.5), treys (>= 0.1.8).
**All commands below assume the project venv is active** (or prefix them with
`.venv/bin/python`).

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
python run_local_match.py --players "smart,smart,mc100,random" --log-session   # + decision logs
```

**Batch statistics** (many tournaments, win rates):
```bash
python run_tournament_stats.py --tournaments 100 --chips 500
```

**Final hybrid-bot eval** (tournament eval for `final_survival` / `final_aggro`):
```bash
python eval_final_bot.py
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
| `icm`, `icmbot` | ICMBot | Tournament equity (ICM) |
| `exploitative`, `exploitativebot` | ExploitativeBot | VPIP/AF/FTA exploitation |
| `gto`, `gtobot` | GTOBot | GTO approximation |
| `opponentmodel`, `opponentmodelbot` | OpponentModelBot | Bayesian hand-range modeling |
| `final`, `final_survival`, `final_aggro` | TournamentHybridBot | Survival / aggro profiles |
| `final_<profile>:p4\|telemetry\|station\|r2\|p5` | TournamentHybridBot | Ablation arms (Phase 4/5) |
| `random` | RandomBot | Uniform random legal actions |
| `maniac`, `maniac_trigger`, `maniac_mixed`, `overbet_merchant`, `calling_station`, `nit`, `folder`, `loose_passive`, `minraise`, `minraiser`, `baseline_sane`, `pressure_filler` | ArchetypeBot | Phase-7 stress archetypes (deliberate caricatures, not realistic opponents) |
| `tag_punisher`, `wide_defender` | ArchetypeBot | Deterministic preflop-core archetypes: tight-aggressive punisher / wide price-sensitive defender (`bots/punisher.py`, zero RNG) |

Example: `--players mc200,smart,gto,icm` → a 4-player table (P1–P4).

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

`tag_punisher` and `wide_defender` are the two disciplined members of the family:
a tight-aggressive punisher and a wide, price-sensitive defender that share one
pure preflop core (`bots/punisher.py`) driven by a committed 169-class equity
table and keyed-hash mixing, so they play deterministically with zero RNG use.

---

## Testing / sanity harness

The repo's correctness is guarded by 14 standalone `sanity_*.py` gates. These
are **gates**, not pytest unit tests: they print `ALL CHECKS PASSED` /
`SOME CHECKS FAILED` and exit nonzero on failure. Run them from the **repo root**.

```bash
.venv/bin/python sanity_engine_parity.py        # legacy vs Poker-Engine parity
.venv/bin/python sanity_tournament_hybrid.py    # hybrid-bot behavior gate
.venv/bin/python sanity_aivat.py                # equity / ICM / AIVAT math
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

- **No web UI** — visualization is matplotlib-only (local). The Colosseum is
  being built to fix exactly this.
- **GTOBot approximates balance with hand-tuned frequencies** — it is not
  solver output.
- **The learning agents are not in this repo.** The MCCFR / Deep CFR / PPO /
  supervised-ML research continues privately (see Project status).

---

## Project status

*Snapshot as of 2026-08-12.*

- **Engine migration — COMPLETE.** Poker-Engine (P1–P6) is merged and is the
  default (`DEFAULT_ENGINE_IMPL = "pe"`). Legacy remains reachable and backs
  the sanity suite; parity is mutation-tested. See
  [docs/plans/ENGINE_MIGRATION_PLAN.md](docs/plans/ENGINE_MIGRATION_PLAN.md).
- **Tournament hybrid bot (Phases 3–7) — implemented and evaluated.**
  Winner-take-all logic, opponent tendencies, and stress-opponent robustness
  are in place.
- **Learning agents — moved to a private research lab.** The MCCFR (Path A),
  Deep CFR (Path B), PPO, and supervised-ML agents and their training
  pipelines were developed in this repo from April through August 2026 and now
  continue in a private repo, so Colosseum participants build their own agents
  rather than bootstrapping from ours. This public repo remains the engine,
  the baseline bot pool, and the tournament tooling.

---

## Docs index

- [docs/plans/CLASS_TOURNAMENT_BOT_PLAN.md](docs/plans/CLASS_TOURNAMENT_BOT_PLAN.md) — hybrid-bot master plan.
- [docs/plans/HARDWARE_BENCHMARK.md](docs/plans/HARDWARE_BENCHMARK.md) — M5 Max sizing.
- Historical plans (kept for provenance): [docs/plans/ENGINE_MIGRATION_PLAN.md](docs/plans/ENGINE_MIGRATION_PLAN.md), [PHASE4_TOURNAMENT_LOGIC_PLAN.md](docs/plans/PHASE4_TOURNAMENT_LOGIC_PLAN.md), [PHASE5_OPPONENT_TENDENCIES_PLAN.md](docs/plans/PHASE5_OPPONENT_TENDENCIES_PLAN.md), [PHASE7_STRESS_OPPONENTS_PLAN.md](docs/plans/PHASE7_STRESS_OPPONENTS_PLAN.md).
