# Engine Migration — Poker-Engine as the Rules Core (Implementation Plan)

> Status: **P1–P4 IMPLEMENTED (2026-07-06)** on branch
> `engine-migration-poker-engine` (uncommitted). D1 = clean break, D2 = yes
> (both approved). Poker-Engine patches are additive and default-off; PE's own
> 32 tests stay green. The legacy engine is untouched and remains the default.
> Full sanity ladder is a clean **37/37 under both legacy and PE**.
> **Remaining: P5 (A/B eval + perf), P6 (flip default).**
>
> Goal: replace `core/engine.py`'s betting/state machine with the engine from
> `../Poker-Engine`, while preserving the `PlayerView`/`BotAdapter` contract so
> bots, training, eval, and the sanity suite keep working unmodified.
>
> **Provenance.** Two code-grounded exploration passes (one per repo, 2026-07-06),
> then an 8-agent deep-spec pass resolving every load-bearing interface, then
> implementation + a mutation-tested parity harness. All file:line references
> below are verified.

## Implementation summary (what landed)

**Poker-Engine (`../Poker-Engine`), all additive / default-off:**
- `engine/cards.py`: `Deck(preset_order=...)` — inject an exact 52-card order.
- `agents/base.py`: `Observation.last_full_raise_size` — lets the adapter
  reconstruct the legacy legal-action bounds (incl. the short-all-in row).
- `engine/hand.py` `play_hand(..., preset_deck, rank_fn, legacy_compat)`:
  deck injection; external showdown ranking (fed legacy `eval_hand`); and the
  legacy-compat rule that any opening bet reopens. Plus an equity skip when
  `equity_sims <= 0`.
- `config.py`/`engine/tournament.py`: classic per-player ante (`player_ante`).
- `pyproject.toml`: flat-layout packaging (`engine`, `agents`, `server`,
  `config`, `cli`). New tests in `tests/test_migration_seams.py`.

**Texas-Holdem-AI:**
- `core/pe_engine.py`: `PokerEngineTable` (drop-in `Table`). Legacy unchanged.
- `core/engine_factory.py`: `make_table(rng, engine_impl)` — the single
  selection seam; lazy imports so the default legacy path never loads
  Poker-Engine. Wired into `run_tournament`, the training loops, the run
  utilities, and a `run_eval.py --engine` flag.
- `sanity_engine_parity.py`: legacy-vs-adapter parity harness (8 check groups,
  300-hand injected-card fuzz), mutation-tested to prove it catches divergences.

---

## 0. Objective and scope

**In scope:** dealing, blind/ante posting, betting rounds, closure logic, and
showdown/side-pot settlement — i.e. what happens *inside* `Table.play_hand()`
(`core/engine.py:331`). These are replaced by Poker-Engine's rules core
(`engine/hand.py`, `engine/betting.py`, `engine/state.py`).

**Out of scope (explicitly kept):**
- `eval_hand` + `_FIVE_CARD_TABLE` + `_FULL_DECK` + `RANK_TO_INT` + `EVAL_HAND_MAX`
  — 16 files sample `_FULL_DECK`, 12 use rank/suit constants, 8 use `EVAL_HAND_MAX`.
  Poker-Engine's evaluator is a per-call pure-Python function; ours is a
  precomputed C(52,5) table and equity MC is already the throughput bottleneck
  (~2.4 s/traversal on Path A). We keep ours.
- `core/equity.py`, `core/aivat.py`, `core/ml_features.py`, all of `bots/`,
  `run_eval.py`, the sanity suite.
- The `PlayerView` shape (`core/bot_api.py:11-29`) and legal-action dict shape.

**Why Poker-Engine:** tested rules layer (~560 lines of tests: side pots,
min-raise/short-all-in, heads-up order, determinism), single seeded master RNG
with byte-identical event logs, clean `LegalActions` precomputation.

---

## 1. Decisions needed before implementation

- **D1 — Baseline comparability vs clean break.** Must the swapped engine keep
  existing eval baselines comparable (hand-for-hand game-tree parity), or is the
  next training run a clean break?
  **Recommendation:** adapter synthesizes the exact legacy history format either
  way (cheap insurance, required by CFR reconstruction regardless); treat the
  next MCCFR/Deep CFR run as the clean break it already is (fresh runs were
  approved 2026-06-11). Scripted-deck parity is still gated in §5 because it
  validates *rules*, not baselines.
- **D2 — May `../Poker-Engine` be modified?** Needed for (a) a minimal
  `pyproject.toml` (it has none) so `.venv/bin/pip install -e ../Poker-Engine`
  works without `sys.path` hacks, and (b) the classic-ante patch (§3.4).
  **Recommendation:** yes, both changes are small and additive; alternative is
  vendoring a copy under `core/vendor/poker_engine/`, which forks the engine you
  like.

---

## 2. Verified interface facts (what the adapter must honor)

| Surface | Legacy (Texas-Holdem-AI) | Poker-Engine |
|---|---|---|
| Entry point | `Table.play_hand(seats, small_blind, big_blind, dealer_index, bot_for, on_event, log_decisions, logger, ante, blind_increase_every) -> Dict[pid, net_chips]` (`core/engine.py:331-336`) | `play_hand(players, button, blind_level, agents, writer, ...)` (`engine/hand.py:26`) |
| Cards | tuples `("A", "h")`; `RANKS="23456789TJQKA"`, `SUITS="cdhs"` | `Card(rank=2..14, suit=0..3)` NamedTuple (`engine/cards.py:25`) |
| Bot action | dict `{"type": fold/check/call/bet/raise, "amount": total-street}` | `Action(ActionType.FOLD/CHECK/CALL/RAISE_TO, amount=total-street)` (`engine/betting.py:20-47`) |
| Legal actions | list of dicts; bet/raise rows carry `min`/`max`, plus separate all-in row `{"min": max, "max": max, "all_in": True, "reopens": False}` (`core/engine.py:637-690`) | `LegalActions(can_fold, can_check, can_call, call_amount, can_raise, min_raise_to, max_raise_to)` (`engine/betting.py:50-94`) |
| History entry | `{street, pid, type, amount, to_call_before, pot_before}`; **calls record `min(stack, to_call)` actually paid** (`core/engine.py:796-811`) | `{street, seat, action, amount}` (`engine/hand.py:168,244`) |
| Ante | classic per-player `ante: int` param | **big-blind ante only** (`config.py:45`, `engine/hand.py:60-64`) |
| RNG | `Table(rng=...)`, default `random.Random(7331)` | caller-seeded: `Deck(rng)`, master RNG per tournament |

Raise semantics agree (raise-to totals, short all-in does not reopen). This is
what makes the adapter tractable.

**Load-bearing consumers of the history format:** `bots/cfr_bot.py`
(`_reconstruct_contributions_from_view`, `_infer_big_blind_from_view`,
`_infer_last_raise_size_from_view`), `core/opponent_stats.py`,
`core/action_history.py` (`extract_history` → Path A infoset keys and Path B
tensors). These are why history synthesis must be byte-exact.

---

## 3. Architecture

### 3.1 Adapter module: `core/pe_engine.py`
A `PokerEngineTable` exposing the **identical** `play_hand` signature and return
value. Internally it drives Poker-Engine's `HandState`/betting functions and
translates at the boundary:

1. **Cards:** two 52-entry dicts built once — `Card(r,s)` ↔ `(rank_str, suit_str)`
   via `RANKS[r-2]` and suit index in `"cdhs"`.
2. **Actions in (bot → engine):** `fold/check/call` map 1:1;
   `bet`/`raise` → `Action.raise_to(amount)` (amounts already share total-street
   semantics).
3. **Legal actions out (engine → bot):** from `compute_legal_actions()`:
   `can_check → {"type":"check"}`; `can_fold → {"type":"fold"}`;
   `can_call → {"type":"call"}`; `can_raise → {"type": "bet" if state.current_bet==0
   else "raise", "min": min_raise_to, "max": max_raise_to}` plus the legacy
   separate all-in row with `all_in`/`reopens` flags
   (`reopens = (max_raise_to - state.current_bet) >= state.last_full_raise_size`).
4. **History synthesis:** on every applied action, append a legacy-format entry
   `{street, pid, type, amount, to_call_before, pot_before}` computed from PE
   state *before* the action; calls record chips actually paid. Blind posts must
   appear (or not) exactly as the legacy engine does — verification item V1.
5. **PlayerView assembly:** unchanged fields, position labels still computed by
   the existing `core/table_order.py` logic from `dealer_index`.
6. **RNG:** pass the `Table` rng straight into PE's `Deck(rng)` — preserves the
   per-episode seeding model used by training loops.

### 3.2 Engine selection flag
`Table(engine_impl="legacy"|"pe")` (or env `THAI_ENGINE_IMPL`), default
`legacy` until §6 P6. One construction point; `run_eval.py`,
`core/tournament.py`, and training scripts pick it up without signature churn.

### 3.3 Settlement
PE `build_pots()`/`award_pots()` replace `_showdown_and_settle`, but ranking
uses **our** `eval_hand` (PE's `award_pots` takes `rank_by_seat` — we feed it
legacy scores, so no dependency on PE's evaluator). `core/aivat.py`'s mirrored
settlement is validated against PE settlement in the parity harness.

### 3.4 Classic-ante patch (Poker-Engine, additive)
`classic_ante: int = 0` in PE config/`play_hand`: every seated player posts
`min(stack, ante)` as dead chips (into `total_committed`, not
`street_committed`) before blinds; BB-ante path untouched; new PE test.
If D2 = "don't touch", fallback: adapter pre-seeds `PlayerState.total_committed`
before the hand — workable but uglier.

---

## 4. Rule-delta risk checklist (each becomes a parity test)

| # | Edge case | Legacy anchor |
|---|---|---|
| R1 | All-in closure: lone live player must still act vs shove | `sanity_engine_allin_closure.py` |
| R2 | Short all-in does not reopen (legacy `raise_blocked` vs PE `has_acted`) | `core/engine.py` betting round; PE `tests/test_betting.py:60-89` |
| R3 | Heads-up: button posts SB, acts first preflop, last postflop | both engines claim it; prove it |
| R4 | Preflop first-to-act UTG (3+ players), postflop from SB | legacy action-order sanity scripts |
| R5 | Short-stack call history amount = chips actually paid | `core/engine.py:796-803` |
| R6 | Odd-chip distribution rule matches legacy `_showdown_and_settle` | PE: clockwise from button-left; legacy: **verify (V2)** |
| R7 | Uncalled bet returned to bettor | PE single-eligible layer |
| R8 | 3-level multiway all-in side pots; folder money in layers, folder never eligible | PE `tests/test_side_pots.py` |
| R9 | Bet-vs-raise labeling in synthesized history matches legacy tokens | `core/action_history.py` sizing tokens |

Resolved verification items:
- **V1 (blind entries in history):** the adapter must **NOT** emit blind/ante
  history entries. `cfr_bot`'s `_reconstruct_contributions_from_view` /
  `_infer_big_blind_from_view` run their no-blind branch on real views and
  recover SB/BB from the first preflop `pot_before`; emitting `type:"blind"`
  entries would inflate VPIP and add spurious check tokens. The adapter records
  exactly the 6-key entries and nothing for blinds/antes. ✅
- **V2 (legacy odd-chip rule):** legacy sends the remainder to the earliest
  winner in `total_contrib` **dict-insertion order** (an incidental artifact,
  not left-of-button). The adapter adopts Poker-Engine's standard
  "first winner left of the button" rule — see **Intentional divergences**. ✅
- **V3 (PE action labels):** PE uses `"raise_to"` and `"post_small/big/ante"`.
  Irrelevant to the adapter, which synthesizes its own legacy-format history
  in `_PEAgent.act` rather than reusing PE's `context["history"]`. ✅
- **V4 (perf):** at trivial-bot settings the adapter is ~2.9× the legacy
  engine's wall-clock (was 5.5× before the equity-skip). Acceptable for a
  correctness-first adapter; real bot compute dwarfs engine overhead. Further
  tuning (per-decision PlayerView rebuild) is P5.

## Intentional divergences from the legacy engine (documented)

1. **Odd-chip split:** the leftover chip in a split pot goes to the first
   winner clockwise from the button (standard poker) rather than the legacy
   dict-insertion-order recipient. Totals conserved; winner sets identical.
   The parity harness compares split pots with a ±1-to-a-winner tolerance.
2. **Lone-runout check-downs:** once everyone but one player is all-in/folded,
   the legacy engine prompts the lone player to check each remaining street
   (a position-dependent quirk); the adapter elides these. They are
   **net-neutral in every case** (a bet into a dry side pot returns as
   uncalled), so no chip outcome changes — only degenerate decision nodes /
   check history entries in all-in runouts. The harness compares only
   *contested* decisions (actor has ≥1 acting opponent) and requires exact net.

The short-all-in **opening**-bet reopen rule *is* reproduced (via
`legacy_compat`) because it affects contested legal actions and thus net.

---

## 5. Parity harness — `sanity_engine_parity.py`

Scripted decks (inject fixed `Deck` order into both engines) + scripted bots
(fixed action sequences). For each R1–R9 case and a fold-around/BB-walk case,
assert equality of: net chips per player, pot layers, the legal-action set at
every decision point, the (adapter-translated) history stream, and showdown
winners. This gates the swap regardless of the D1 answer.

---

## 6. Rollout phases

- **P0 — interface verification.** ✅ Done 2026-07-06 (§2).
- **P1 — PE packaging + classic-ante patch + seams.** ✅ PE suite green (32).
- **P2 — adapter** `core/pe_engine.py` + `make_table` selection, default
  `legacy`. ✅
- **P3 — parity harness green** (8 groups incl. R1–R9 + 300-hand fuzz),
  V1–V4 resolved, mutation-tested. ✅
- **P4 — sanity suite** run with `impl=pe`; triage every diff. ✅ *(fast/medium
  tiers)* — torch env rebuilt on Python 3.12 (torch 2.12.1); the validation
  ladder passes **31/32 under legacy and 31/32 under PE** (5 slow gates skipped
  on both). The single PE failure is `sanity_test_hand`, which monkeypatches the
  legacy-private `table._betting_round` for street-boundary instrumentation —
  test-harness coupling, not a behavioral divergence; the behavior it checks is
  covered by `sanity_action_order` (passes under PE) and the parity harness. The
  suite was run under PE via a `sitecustomize.py` shim that swaps
  `core.engine.Table → PokerEngineTable` when `THAI_ENGINE_IMPL=pe`, so no
  regression-test files were edited. **`--full` gates:** legacy **37/37**; PE
  **37/37** (2026-07-06, after the fix below). Every gate passes under PE,
  including all four training smoke pipelines (deep_cfr, train_deep_cfr,
  train_cfr, train_deep_cfr_abort) and the full eval harness (head-to-head,
  multiway, Wilson CIs, verdict logic, factory loading). Perf: PE `--full`
  649s vs legacy 466s (~1.4× overall; the eval gate 411s vs 226s, consistent
  with the ~2.9× engine overhead diluted by bot/NN compute). **Clean-sweep fix
  (2026-07-06):** `sanity_test_hand` previously monkeypatched the legacy-private
  `table._betting_round` purely to log street boundaries — the sole PE gap. It
  now derives the streets played from the `RecordingBot`s' observed
  `view.street` (engine-agnostic), so it passes under both engines; this also
  clears `sanity_eval`, whose only PE failure was a regression subprocess that
  re-ran `sanity_test_hand`. No other regression-test files touched.
- **P5 — A/B eval:** `run_eval --engine pe` vs `--engine legacy`, same seeds;
  scripted-deck subset identical, full-run chip distributions statistically
  indistinguishable; finish the perf tuning (per-decision PlayerView rebuild).
- **P6 — flip default**, mark legacy deprecated in-code; next MCCFR / Deep CFR
  runs (already planned as fresh) train on PE impl.

**Call-site wiring (done 2026-07-06):** engine selection lives in
`core/engine_factory.py :: make_table` (lazy imports, so the legacy path never
loads Poker-Engine). Wired through it: `core/tournament.py::run_tournament`
(new `engine_impl` param; the funnel `run_eval` already routes through it),
`run_local_match.py`, `run_tournament_stats.py`, and the three training loops
(`train_cfr_bot_multiway`, `train_rl_bot_selfplay`, `train_multi_deep_rl_bot`).
`run_eval.py` gained a `--engine {legacy,pe}` flag. Selection precedence:
explicit arg > `THAI_ENGINE_IMPL` env > `legacy`. The ~50 `sanity_*.py` scripts
still hard-code `Table()` on purpose — they are legacy regression tests; P4 runs
them under `impl=pe` via a separate env-aware harness pass, not by editing each.

**Env note:** the project `.venv` is internally split (`python`=3.12 empty,
`pip`→3.14) and has no torch. P1–P3 are pure-stdlib so unaffected; P4's full
sanity suite will need a torch-capable interpreter.

**Effort:** P1–P3 done. P4–P5 ≈ 1–2 days + eval wall-clock once the env and
call-site wiring are in place. Retraining cost attributable to the swap: none,
if flipped before the next run.

---

## 7. Done-when

- [x] Parity harness green (R1–R9 + fuzz), V1–V4 resolved and documented.
- [x] Poker-Engine suite green (32) with all patches default-off.
- [x] Call sites wired through `make_table`; torch env rebuilt (Python 3.12).
- [x] Sanity ladder under `impl=pe`, incl. `--full`: legacy 37/37, PE 37/37 —
      clean sweep after making `sanity_test_hand` engine-agnostic; no divergences.
- [ ] A/B `run_eval` shows no regression; perf within budget (V4 tuning) (P5).
- [ ] Default flipped to `pe`; this doc updated to IMPLEMENTED (P6).

---

## Appendix A — Implementation kickoff prompt

**Goal:** Implement P1–P3 of `ENGINE_MIGRATION_PLAN.md`: package Poker-Engine,
add its additive classic-ante support, build `core/pe_engine.py` exposing the
legacy `Table.play_hand` contract on top of Poker-Engine's rules core behind an
`engine_impl` flag (default `legacy`), and land `sanity_engine_parity.py` green.

**Context:** Repos at `~/Desktop/Projects/Texas-Holdem-AI` (consumer) and
`~/Desktop/Projects/Poker-Engine` (rules core). Interface contract and verified
line references in §2; translation rules in §3; test matrix in §4–5. Use
`.venv/bin/python`. History synthesis must be byte-exact
(`{street, pid, type, amount, to_call_before, pot_before}`, calls record
`min(stack, to_call)`), because `bots/cfr_bot.py` and `core/opponent_stats.py`
reconstruct state from it.

**Constraints:** Do not modify `PlayerView`, legal-action dict shapes, bots, or
trained-artifact loaders. Do not adopt PE's hand evaluator — settlement ranks
come from legacy `eval_hand`. Legacy engine stays the default until P6. No
commits without explicit approval; no Co-Authored-By trailer.

**Done-when:** PE tests green after ante patch; parity harness passes R1–R9
with V1–V3 resolved in-code or documented; full legacy sanity suite still green
with `impl=legacy` (no regression while flag is off).
