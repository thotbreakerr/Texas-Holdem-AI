# Fix report — review findings (2026-05-28)

Implemented the confirmed P0 / top-P1 fixes from the code review. Scope kept
tight: no unrelated refactors, no training runs. All bot/engine/math fixes are
covered by focused regression tests.

## Changed source files

| File | Fix |
|---|---|
| `core/engine.py` | **P0** `all_live_equal()` now closes a street only when `len(live)==0`; a lone live player must act when their contribution trails the highest (incl. all-in). Previously a shove against a single opponent skipped that opponent's call/fold. |
| `bots/cfr_bot.py` | **P1** Pot-relative raise sizing. New `_raise_sizing_target` (`target = current_bet + frac*(pot+to_call)`), `_current_bet_from_view`; threaded `to_call`/`current_bet` through `_legal_abstract_actions`, `_abstract_to_concrete`, `_GameState.legal_abstract_actions`, `apply_action`, `act`, `_heuristic_action`, `_search_fallback`. Buckets no longer collapse to min-raise when facing a bet. Defaults preserve opening-bet behaviour. |
| `bots/deep_cfr_bot.py` | **P1** (#3) Postflop sizing head no longer overrides the bucket fraction — clamped to a band around the bucket's frac so `bet_33 ≠ bet_100`. (#4) `build_network_input` reconstructs each opponent's committed amount (was hardcoded 0 at inference) to match training feature semantics; safe fallback to 0. |
| `bots/ml_bot.py`, `bots/rl_bot.py` | **P1** `_estimate_hand_strength` normalizes preflop (<5-card) heuristic scores on the preflop scale (`2*max_rank+40`) instead of `EVAL_HAND_MAX`, which had collapsed every preflop hand to ~0 (AA folded to any bet). |
| `core/icm.py` | **P1** Last survivor takes only their single reachable payout slot (was summing all trailing slots → stack-insensitive output). Added zero-payout pruning → winner-take-all is O(N), not O(N!). |
| `core/aivat.py` | **P1** Tournament fold baseline now distributes the pot to non-hero alive seats (proportional) so the fold world and play world share the same chip total — the ICM delta is chip-conserving. `chip_ev` path unchanged. |
| `training/train_deep_cfr.py` | **P1** Mid-run canary FAIL now `break`s instead of bare `raise`, so the documented `status="aborted"` return runs (and `main()` exits nonzero) instead of an uncaught traceback. |
| `run_eval.py` | **P2** Tier-3 target `1/6`→`1/7` (7-player field); `avg_position` None-guarded in curriculum print; max-hands-with-survivors resolves a deterministic chip-leader winner; documented the intentional per-tournament bot reload. |

## New / changed test files

New: `sanity_engine_allin_closure.py` (HU shove **and** 3-way two-all-in side-pot
conservation), `sanity_preflop_strength.py`, `sanity_icm_payouts.py`,
`sanity_cfr_sizing_parity.py` (train/deploy parity), `sanity_deep_cfr_fixes.py`,
`sanity_train_deep_cfr_abort.py`, `sanity_run_eval_fixes.py`.

Updated: `sanity_aivat.py` (fold-baseline expectations derived by hand; losing≈0
/ winning>0 reframe), `sanity_cfr_equity.py` (TEST 15 passes betting context to
both paths; 72o assertion uses the sizing-independent `regret[fold] > regret[call]`
invariant — verified true under both old and new sizing).

## Test results (`.venv/bin/python`)

All pass:
- New regressions: `sanity_engine_allin_closure`, `sanity_preflop_strength`,
  `sanity_icm_payouts`, `sanity_cfr_sizing_parity`, `sanity_deep_cfr_fixes`,
  `sanity_train_deep_cfr_abort`, `sanity_run_eval_fixes`.
- Required suites: `sanity_review_findings`, `sanity_aivat`, `sanity_deep_cfr`,
  `sanity_train_deep_cfr`, `sanity_test_hand`, `sanity_test_followon`.
- `py_compile` clean on all touched files.

Pre-existing failures (NOT caused by these changes — proven by reverting the
engine + sizing changes; failures persist):
- `sanity_cfr_equity.py` / `sanity_train_cfr.py`: "smoke tournament crashed" —
  `bots/poker_mind_bot.py:227` calls `max()` over empty `opp_hands` on the river
  (a latent bug in the `smart` bot, out of scope for these findings).
- `sanity_eval.py`: also has CFR-missing-profile fixture failures (CFRBot's
  intentional loud-fail on a missing profile path). Its run_eval-specific checks
  (Wilson CI, verdicts) pass.

## Known limitations (intentionally not fixed)

1. **CFR profile needs retraining.** The sizing fix changes realized bet amounts,
   which shift the history-token info-set keys; the existing trained pickle's
   facing-bet keys won't be hit at inference until retrained. The fix is correct;
   benefit requires a retrain (out of scope — no training run requested).
2. **`_build_game_state_from_view` reconstructs `big_blind` imperfectly**, so the
   reconstructed (training) min-raise can exceed the real (deploy) one; bet
   buckets pinned in that floor gap differ by the clamp (surfaced as the
   boundary-pinned bucket in `sanity_cfr_sizing_parity`). Pre-existing and
   orthogonal to pot-relative sizing.

## Resolution update (2026-05-28, follow-up gate-blocker fix)

The "Pre-existing failures" listed above are **RESOLVED**. They were the same
class of bug (empty opponent-hand candidate set) and a stale eval fixture:

- `bots/poker_mind_bot.py`, `bots/gto_bot.py`, `bots/exploitative_bot.py`:
  guarded the Monte-Carlo equity functions with `num_opponents = max(1, …)`.
  When every opponent is all-in the caller passed `num_opponents=0`, leaving
  `opp_hands` empty and crashing `max()` on the river. The same latent crash
  existed in `gto`/`exploitative` (both in `DEFAULT_POOL`), so all three were
  fixed. New regression: `sanity_smart_bot_river.py`.
- `sanity_eval.py`: fixture-only fix. Sections 1/2 pointed Path A (and, once
  CFR stopped masking it, Path B) at non-existent artifacts; they now mint a
  temporary valid CFR profile + a temporary random Deep CFR checkpoint.
  Section 5 was rewritten to the real contract (bare `cfr` degrades to the
  heuristic; explicit `cfr:<missing>` / `deep_cfr:<missing>` raise loudly).

`sanity_cfr_equity.py`, `sanity_train_cfr.py`, and `sanity_eval.py` now pass.
These gates — plus the rest — are wired into the canonical pre-retrain gate
`sanity_validation_ladder.py` (see `TRAINING_PLAN.md`). No production logic in
the protected files was changed; no training was run.
