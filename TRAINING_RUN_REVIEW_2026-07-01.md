# Training run review — live smoke runs of all trainers (2026-07-01)

Live review (no repo code changed; all smoke artifacts written to session scratch space).
Trigger: "bots that require training are not training right." Method: ran the project's
own validation ladder, then tiny smoke runs of all four trainers, an instrumented +
cProfiled 6-hand MCCFR probe, round-trip load tests of both smoke checkpoints, and
throughput extraction from the May production log.

## Bottom line

**The training code is not broken — nothing that requires training has actually been
trained.** Every sanity gate passes and all four trainers run and learn mechanically.
The "not training right" experience decomposes into:

1. **No trained models exist for the current schema.** `models/cfr_regret_deep_v2.pkl`
   and `models/deep_cfr_v2.pt` (the defaults everywhere) were never produced. The May
   1M-iteration Deep CFR run **aborted at iter 415k after 49h** (collapse canary tripped,
   final checkpoint withheld). No v2 run has been started since the June-13 fixes.
2. **MCCFR (Path A) is computationally infeasible at planned settings** — ~2.4 s per
   traversal, measured. The bare CLI default (50k tournaments × 200 iters/decision)
   prints its header and then nothing for weeks: it *looks* hung. TRAINING_PLAN.md's
   "2–4 hours for `--tournaments 5000 --iterations 10`" is off by 100–1000×.
3. **A handful of real bugs/traps** (F3–F5 below), including `--device cpu` being
   silently ignored.

Triage: fix Path A's leaf cost before any real MCCFR run; Path B can start its staged
rollout today with realistic wall-clock budgets (~5 days for the 1M run).

---

## Evidence run — all green

| Check | Result |
|---|---|
| `sanity_validation_ladder.py --path both --keep-going` | **32 passed, 0 failed**, 5 slow gates skipped (default mode), 115 s |
| Deep CFR smoke (`--variant small`, 300 iters, MPS) | Completes, losses move (a≈15→18, v≈24→22, p≈1.3→1.1), 3 rounds fit, checkpoint saved, exit 0 |
| Deep CFR checkpoint round-trip | Loads via `deep_cfr:<path>`, plays AKs BTN → raise (sensible) |
| MCCFR smoke (2 tournaments, `--iterations 5`, seed 42) | Completes, 17,319 info sets, 240 traversals, pkl saved, exit 0 |
| MCCFR pkl round-trip | Loads via `cfr:<path>` (17,319 info sets), inference 0.08 s |
| RL selfplay smoke (3 episodes) | Runs, saves model + CSV, exit 0 |
| ML trainer | Fails **loudly** on legacy logs (see F4) — correct behavior, blocked on data |
| Repo state after review | Working tree clean; `models/` untouched (newest file still Jun 14) |

---

## F1 — Path A (MCCFR): correct loop, intractable cost  **[blocker for any real run]**

Instrumented 6-hand probe (6-max, 1000 chips, `iterations=5`, all six seats sharing one
training CFRBot):

```
wall time:           520.8s  (86.8s/hand; 100% of it inside act())
act() calls:         43
total traversals:    215 / 215 expected  → NO silent skips
decision pts w/ 0 or partial traversals: 0
info sets:           18,393    recursion calls: 78,654
slowest single decision: 115.4s (preflop)
per-traversal cost:  ~2.4s
```

cProfile of one 8-decision hand: 99% of cumulative time under
`_leaf_value → core/aivat.py value → _chip_ev_value → _evaluate_showdown`:

- 33,953 showdown evaluations, 143,861 `eval_hand` calls, **3.16M `_score_five` calls**
  in a 4-second act().
- Cost drivers in `core/aivat.py:222` (`_chip_ev_value`):
  - **Flop leaves enumerate all C(45,2) ≈ 990 runouts exactly**, each a full multi-player
    showdown (≈21 combos × players each).
  - Preflop leaves: `n_sims=200` Monte Carlo (hardcoded at `bots/cfr_bot.py:1810/1829`).
  - Turn leaves: ~44 exact runouts. River: 1 (cheap).

Consequences:

- Bare `python training/train_cfr_bot_multiway.py` (defaults 50,000 × 200): first
  progress print is at episode 100 → **unreachable**; first checkpoint at episode 500 →
  unreachable. Header, then silence. This is the "not training" symptom.
- Plan's overnight run (5000 × 10): realistically **weeks–months**, not 2–4 h. The
  estimate predates the recursive `_cfr_recurse` + AIVAT-leaf rewrite (Apr 27) and was
  never re-benchmarked.
- Why gates never caught it: `sanity_train_cfr.py` deliberately runs `iterations=1`,
  200-chip stacks, 3 tournaments — it validates correctness, not throughput.

What is *not* wrong: no skipped traversals, regrets update, checkpoint save/resume and
inference round-trip all work. This is purely a per-traversal-cost problem.

Fix directions (est. 20–50× combined, none touch CFR math):
sample ~50 flop runouts instead of exact 990-board enumeration; cut preflop leaf
`n_sims` 200 → 50; memoize showdown evals per (board, alive-hands) within a traversal.

## F2 — Path B (Deep CFR): healthy and retrain-ready; budget ~5 days

- Smoke run trains, fits rounds, checkpoints, reloads, plays sensibly (see table).
- Throughput **~2.2–3.1 it/s on MPS** — identical in my small-variant smoke and the May
  large-variant log. Staged rollout costs at ~2.2 it/s:
  10k smoke ≈ **1.5 h** · 150k pilot ≈ **19 h** · 1M run ≈ **5.2 days**.
- May run post-mortem (`output/deep_cfr_large_clean_1m.log` tail): ran 49 h to iter
  415k at 2.3 it/s, then `[ABORT] Final checkpoint not saved because the all-in
  collapse canary tripped.` Last usable artifact: `warn_410000.pt`. The collapse root
  cause (abstract all_in regret row) was addressed in the June schema-v2 work; the fix
  has never been exercised by a production run.
- Reminder: the collapse canary only *enforces* from iter 100k
  (`--canary-enforce-iteration` default) — short runs never test collapse.
- `models/deep_cfr_phase0_schema_v2_anchor.pt` (Jun 14, the newest model file) is
  referenced by nothing in the repo — orphan of an ad-hoc run; don't mistake it for a
  trained v2.

## F3 — `--device cpu` silently ignored  **[bug, one-line fix]**

`pick_device` (`training/train_deep_cfr.py:134`) has branches for `"mps"` and `"cuda"`
but none for `"cpu"` — an explicit `cpu` falls through to the auto block and returns
MPS on this machine. Verified live:

```
pick_device('cpu')  -> mps     # wrong
pick_device('auto') -> mps
pick_device('mps')  -> mps
pick_device('cuda') -> cpu     # correct fallback (no CUDA)
```

Matters because the May abort happened on MPS and CPU is the natural numerics A/B test.

## F4 — ML bot: blocked on data (loud, correct error)

`train_ml_bot.py` raises: the only decision logs (`logs/`, 3 files, April) are legacy
per-hand logs without session headers, so cumulative opponent-memory features can't be
reconstructed. Regenerate with `run_local_match.py --log-session` (or pass
`--allow-legacy-logs` accepting train/serve skew).

## F5 — Silent untrained-CFR trap  **[trap, poisons evals/RL]**

Bare `cfr` spec (`bots/__init__.py:89`): if `models/cfr_regret_deep_v2.pkl` is missing
it falls back to `profile_path=None` → **blank bot (0 info sets), no warning**.
Verified live. Affected today, since no CFR pkl exists:

- `run_eval.py --pool cfr` pool entries (pool specs pass raw strings to `create_bot`);
- `train_multi_deep_rl_bot.py` opponent pool (`models/cfr_regret.pkl` also missing →
  RL trains vs an untrained "CFR" opponent, silently).

Explicit `cfr:<path>` / `deep_cfr:<path>` fail loudly (good) — run_eval's PATH_A/PATH_B
use the explicit form, so mainline eval fails fast rather than masking.

## F6 — RL selfplay trainer: mechanically fine

3-episode smoke runs and saves. Meaningfulness of its curriculum inherits F5 (opponent
quality), nothing new.

---

## Recommended order of work

1. **Start Path B rollout** (the one retrain-ready pipeline): 10k smoke → 150k pilot →
   1M (~5 days). Watch the canary from iter 100k.
2. **Path A leaf-cost fix before any real MCCFR run** (sampled flop runouts, lower leaf
   n_sims, per-traversal showdown memoization), then re-benchmark and update
   TRAINING_PLAN.md's estimate.
3. One-liners: `cpu` branch in `pick_device`; loud warning (or raise) on the bare-`cfr`
   missing-profile fallback.
4. Regenerate session logs if the ML bot still matters for the class tournament.

---

## Addendum — items 2 and 3 implemented (2026-07-01, same day)

- **F1 fix landed**: `core/aivat.py` gained opt-in `max_enumerate` (sampled turn/flop
  runouts) + `LeafScoreCache` (per-traversal completion/score memo); CFRBot gained
  `leaf_sims` / `leaf_enum_cap` (defaults preserve old behavior — only the trainer opts
  in); trainer defaults now `--iterations 10 --leaf_sims 100 --leaf_enum_cap 120` with a
  per-episode wall-time + ETA heartbeat for the first 10 episodes.
  Measured: probe 86.8 → 3.2 s/hand (~27×) on deep-stack hands; full-tournament
  workloads ~1.9–2.4 → **0.45 s/traversal** at defaults (0.32 at `50/60`), because
  short-stack all-in phases were never flop-enumeration-bound. Sampled leaf values are
  within 0.23 % of exact (50-round avg, cap 120). Realistic sizing now lives in
  TRAINING_PLAN.md step 8: 5000 tournaments ≈ 7–9 days; overnight ≈ 250–350.
- **F3 fixed**: `pick_device("cpu")` returns cpu (verified end-to-end, header prints
  `Device: cpu`). New Check 7 in `sanity_train_deep_cfr_signals.py`.
- **F5 fixed**: bare `cfr` fallback now prints an UNMISSABLE `UNTRAINED` warning to
  stderr; `sanity_review_findings.py` asserts both no-crash and the warning.
- Gates after all changes: full ladder **32 passed / 0 failed**; slow
  `sanity_train_cfr` ALL PASS; new `sanity_aivat.py` section (cache value-neutrality,
  turn exactness under cap, sampled-flop tolerance, RNG short-circuit) ALL PASS.
- F2 (start Path B rollout) and F4 (regenerate ML session logs) remain open — both are
  runs to launch, not code changes.
