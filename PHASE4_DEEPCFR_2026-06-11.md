# Phase 4 — Deep CFR (Path B) retrain readiness (2026-06-11)

> **Superseded on 2026-06-13 by schema v2.** The shadow/staged all-in
> curriculum and shared-network v1 design documented below are retained only
> as history. Current code trains every legal action from traversal one, uses
> independent advantage/average-strategy/value/sizing networks, deploys the
> average strategy, uses the sixmax-weighted curriculum, enforces probability
> canaries from 100k with patience 3, and rejects v1 resume/deployment.

Scope: `bots/deep_cfr_bot.py`, `core/action_history.py`, `training/train_deep_cfr.py`.
Path A (`bots/cfr_bot.py`, `train_cfr_bot_multiway.py`) and the engine were NOT
touched. No training launched. Background: `REVIEW_2026-06-05.md` (B1–B5),
`REVIEW_2026-06-09.md` (I3/I5/I6/I9, M4), `sanity_review_findings.deepcfr.json`,
and the Phase 3.1 addendum in `PHASE3_AUDIT_2026-06-10.md`.

Already fixed before this phase (verified present, not redone): call-amount
history parity (engine records paid chips), opponent committed reconstruction
at inference (`build_network_input`), sizing-head ±25% bucket refinement
(`_abstract_to_concrete`), shadow-phase expansion mask == policy mask
(Key Change #2 in `_cfr_recurse`).

Each fix below landed as a separate, reviewable change with its own gate; every
gate was run against the PRE-fix code (via `git stash`) to confirm it fails
there, then against the post-fix code to confirm it passes.

---

## Fix 1 — All-in history label train/serve parity (I3 / B1 remnant)

**Verified REAL.** `_DeepCFRGameState.apply_action` emitted
`ActionEvent(action="all_in")` for the abstract shove, but the engine has no
all-in action type — `core/action_history.extract_history` can only ever
produce `bet`/`raise` for a shove. The history GRU's all-in one-hot channel
(idx 5 → tensor channel 19) was therefore trained on prefixes that never occur
at play time, and the bet/raise channels were under-trained on shove lines —
precisely the post-all-in decision contexts the all-in curriculum hardens.

**Change** (`bots/deep_cfr_bot.py` apply_action): the `label == "all_in"`
branch now emits `action_type = spec["type"]` (bet/raise), matching the engine.
`event_amount` already recorded the committed total (engine semantics) — kept.
`"all_in"` stays in the `_ACTIONS` vocabulary, so tensor shapes and previously
trained checkpoints are unchanged; the channel simply goes unused (inference
encodes engine histories, which never contained it).

**Consumers audited:** repo-wide grep for `history_events` /
`ActionEvent.action == "all_in"`. One consumer updated:
`sanity_deep_cfr.py` Section 11's `biased_value_head` test stub now detects the
shove line from the actor's post-action stack (`stacks[seat] <= 0` after a
bet/raise) instead of the dead label. `tokenize()` keeps accepting `all_in`
(legacy events) but neither producer emits it anymore (doc note added).

**Gate:** `sanity_deep_cfr_history_parity.py` — plays equivalent hands in the
real engine (`Table.play_hand` driven by scripted bots that map abstract
actions through `_abstract_to_concrete`, the inference mapping) and in the tree
(`apply_action` on the same abstract scripts), then asserts identical
(street, seat, action, amount, pot_before) sequences. Scenarios: a preflop
shove + call-for-less (3-handed), a heads-up limp pot with a flop bet_33 raise
that clamps to the min-raise, and a heads-up postflop shove. Also asserts no
tree event ever carries the engine-impossible `all_in` label.
Pre-fix: FAILs on both shove scenarios (`('preflop', 1, 'all_in', 150, 40)`).
Post-fix: PASS.

## Fix 2 — Finiteness guard before optimizer.step() (B2 / I5)

**Verified REAL.** `train_step` ran `backward → clip_grad_norm_ → step` with no
finiteness check; `clip_grad_norm_` propagates NaN (NaN total norm → NaN clip
coefficient → every gradient NaN → `step()` writes NaN into every parameter),
and a NaN net can slip past the warmup canary (NaN regrets → uniform policy →
all-in masked → PASS).

**Change** (`training/train_deep_cfr.py`):
- `train_step` checks `torch.isfinite(total_loss)` before `backward()`; on
  non-finite it skips the optimizer step (parameters untouched) and returns
  `nonfinite_skip=True` plus the three component losses (still loggable —
  the NaN component identifies the bad head).
- `run_training` logs each skip with the component losses, counts total +
  consecutive skips, and aborts (status `"aborted"`,
  `abort_reason="nonfinite_loss"`, nonzero exit via `main()`) after
  `NONFINITE_SKIPS_ABORT_THRESHOLD = 50` CONSECUTIVE skips — silent
  perma-skipping is its own failure mode. A successful finite step resets the
  consecutive counter. Unlike a canary abort, the final checkpoint IS saved
  (parameters are finite — every bad step was skipped).
- The cumulative counter is persisted as `nonfinite_skips` in checkpoint
  metadata and restored on `--resume` (`load_checkpoint(meta_out=...)`).

**Gate:** `sanity_deep_cfr_nonfinite_guard.py` — (1) NaN injected into a regret
target → skip reported, parameters bit-identical, still finite; (2) control:
finite batch steps and changes params; (3) run with every `train_step`
monkeypatched to skip → aborts exactly at iteration 50 with
`abort_reason="nonfinite_loss"`, checkpoint exists with `nonfinite_skips: 50`;
(4) alternating skip/step never aborts (consecutive reset) while still counting
totals. Pre-fix: FAILs check 1 ("parameters changed despite non-finite loss",
"NaN leaked into parameters"). Post-fix: PASS.

## Fix 3 — Iterations default footgun (B4 / I9)

**Verified REAL.** `--iterations` defaulted to 100_000 <
`ALL_IN_DEPLOY_ITERATION = 150_000`, so a default run saved a final model whose
inference permanently masks all-in.

**Change** (`training/train_deep_cfr.py`):
- Default raised to 1_000_000 (TRAINING_PLAN step 7).
- `checkpoint_with_canary` takes `final=` (periodic saves pass
  `t >= args.iterations`; the finally-block save passes `True`). A FINAL save
  with `iteration < all_in_deploy_iteration` prints a loud multi-line warning
  and stamps `"shadow_only": true` into checkpoint metadata (both the
  canary-promoted and `--disable-collapse-canary` paths). Warn-and-stamp, NOT
  refuse — `sanity_train_deep_cfr.py`'s ~100-iteration smoke runs keep passing.
  Mid-run checkpoints are not stamped (being below the gate is normal there).

**Gate:** covered in `sanity_train_deep_cfr_signals.py` (checks 3+4): the
interrupted under-gate final checkpoint carries `shadow_only: true`, and
`parse_args` defaults to 1,000,000 (≥ the deploy gate). Verified manually too:
a 20-iteration run prints the warning and stamps the checkpoint.

## Fix 4 — Signal handling (B5 / M4)

**Verified REAL.** SIGINT set a flag and broke the loop, returning
`status="complete"` / exit 0 (an orchestrator keying on exit codes treats an
under-trained model as finished); SIGTERM was untrapped (OS kill loses the
in-flight checkpoint).

**Change** (`training/train_deep_cfr.py`): one handler traps BOTH SIGINT and
SIGTERM, records the signal number, finishes the current traversal, saves the
final checkpoint (existing finally-block machinery), reports
`status="interrupted"` (+ `signal`) in the summary, and `main()` exits
`128 + signum` (130 for SIGINT, 143 for SIGTERM). Original handlers restored
in the finally block. Canary aborts still return `status="aborted"`.

**Gate:** `sanity_train_deep_cfr_signals.py` — real subprocess runs; SIGTERM
mid-run → checkpoint exists + exit 143; SIGINT mid-run → checkpoint + exit
130; the summary says "Training interrupted", never "Training complete.".

## Fix 5 — Training-state curriculum (I6) + blind-invariant features

**Verified REAL.** `build_initial_state` was the only source of training
states: always 6 seats, equal 1000 chips, 5/10 blinds. Eval starts 500 chips
(50 BB vs 100 BB training) and tournaments are decided heads-up, short-stacked,
at escalated blinds — all out-of-distribution. Additionally verified: the
heads-up branch of the old builder was WRONG (seat 1 posted SB, seat 0 BB —
the engine has the button post SB heads-up), latent because n=2 was never
exercised; and the feature encoding was NOT blind-invariant (raw-chip
log-normalization in scalars, opponent stacks, and history amounts), so varying
depth at fixed blinds would not have covered blind escalation.

**Changes:**
- `core/action_history.py`: `REF_BIG_BLIND = 10`; `to_tensor` gains a
  `big_blind` parameter and rescales event amount/pot to the reference blind
  (`x * 10 / big_blind`) before the existing log-normalization.
- `bots/deep_cfr_bot.py`: `_build_scalars` (pot/to_call/stack) and
  `_fill_public_opp_features` (opponent stack channel) rescale the same way;
  `_DeepCFRGameState.to_network_input` passes the tree's exact `big_blind`;
  `build_network_input` passes `_infer_big_blind(view)` (bounded existing
  reconstruction; exact for standard blind posting). At `big_blind == 10` the
  rescale factor is exactly 1.0 — verified byte-identical tensors vs the
  pre-change encoder for both the tree and PlayerView paths, so existing
  checkpoints see unchanged inputs in standard 5/10 games. SPR, committed/pot,
  and the one-hots were already ratios/labels. Known limitation kept: the
  opponent stack channel still saturates at 100 BB (pre-existing semantic);
  hero depth stays fully resolved via the log-scaled scalar.
- `training/train_deep_cfr.py`: `build_initial_state` accepts per-seat
  `stacks`, fixes the heads-up blinds (button posts SB), and takes the preflop
  order from the shared `core.table_order.street_action_order` (identical to
  the old hardcoded order for n=3..6), excluding seats already all-in from
  posting a blind (the engine skips them with no history entry). New
  `sample_curriculum_state(iteration)`: player count uniform over {2..6},
  hero seat `(t-1) % n` (rotation preserved), stacks 10–200 BB log-uniform in
  chips at fixed 5/10 blinds — 50% one shared depth, 50% independent per-seat
  depths. The training loop uses it; the banner prints the curriculum.

**Test maintenance:** `sanity_train_deep_cfr.py` Section 4b's "default-size
short run warns" check assumed a 50-iteration run at `--batch-size 256` makes
zero regret gradient steps. Curriculum states (deep-stack / heads-up trees
produce more hero decision nodes per traversal) now fill 256 regret samples
within 50 iterations, so the zero-step premise broke while the warning logic
it tests stayed intact. The check now forces the condition with an explicitly
unreachable `--batch-size 100000` (comment documents why). All other sections
pass unchanged.

**Gate:** `sanity_deep_cfr_curriculum.py` —
1. 600 sampled states structurally valid (blind seats/amounts, pot, engine
   action order, hero rotation, card accounting); all of n=2..6 sampled; both
   depth modes occur; depths span [100, 2000] chips into both tails.
2. Action-order parity vs the REAL engine for n=2..6: scripted all-call/check
   hands compared event-by-event (street, seat, action, amount, pot_before)
   between `extract_history(engine view)` and the production builder's tree.
3. `to_network_input`/`opp_mask` for n<6: mask sums to n−1, padded rows zero.
4. Blind posting with stacks < blind: `min()` posting, no negative stacks,
   all-in blind seats excluded from action order, call-for-less honored.
5. Scale invariance: ×10 chips AND blinds → bit-identical input tensors (tree
   encoder and PlayerView encoder, blind inferred from history), plus a
   negative control (×10 chips at FIXED blinds must change features).
Pre-fix: the invariance checks FAIL (history channels diverge, max abs diff
≈0.248). Post-fix: PASS.

---

## Out of scope (deliberately not done)

No average-strategy/policy head (I4), no sizing-head loss change (MSE stays),
no raise-target formula change (B3 stays self-consistent as-is), no Path A or
engine changes.

## Validation (all run locally with torch, `.venv/bin/python`)

| Gate | Result |
|---|---|
| `sanity_validation_ladder.py --path deep-cfr --keep-going` | **23 passed, 0 failed** (incl. all four new gates wired in: history_parity + curriculum in Tier 3, nonfinite_guard + signals in Tier 5 fast) |
| `sanity_deep_cfr_history_parity.py` | PASS (FAILs pre-fix) |
| `sanity_deep_cfr_nonfinite_guard.py` | PASS (FAILs pre-fix) |
| `sanity_train_deep_cfr_signals.py` | PASS (SIGTERM→143, SIGINT→130, shadow stamp, 1M default) |
| `sanity_deep_cfr_curriculum.py` | PASS (invariance FAILs pre-fix) |
| `sanity_deep_cfr.py` (small + large) | PASS |
| `sanity_train_deep_cfr.py` (full trainer suite incl. smoke runs) | PASS |
| `sanity_train_deep_cfr_abort.py` | PASS (canary abort semantics unchanged) |
| `sanity_deep_cfr_fixes.py` | PASS |
| `sanity_action_history.py` | PASS (`to_tensor` default behavior unchanged) |
| Encoder compatibility probe | New vs pre-change encoders byte-identical at big_blind=10 (tree + view paths) |

## Retrain notes

- These changes alter the TRAINING DATA distribution (curriculum, shove
  history labels), not network shapes — old checkpoints still load and play,
  and their 5/10 encodings are unchanged. A fresh Path B run is required to
  benefit; per TRAINING_PLAN step 7: `--variant large` with the (new) default
  `--iterations 1000000`.
- Expect somewhat slower per-iteration wall-clock vs the old fixed 6-max
  config on average (deeper stacks → bigger trees at depth 8), partially
  offset by the 2-5 player states.

---

# Phase 4.1 addendum — Codex audit follow-ups (2026-06-11, pre-commit)

An independent review (Codex) confirmed the Phase 4 fixes as materially real
and correctly scoped, but flagged three commit blockers. Each was verified
REAL by running the strengthened gates against the pre-4.1 trainer (evidence
below), then fixed.

## 4.1-1 — Emergency checkpoints must never be blocked by the collapse canary

- **Issue (verified REAL).** The trainer's `finally` block routed EVERY final
  save through `checkpoint_with_canary`, which raises on a FAIL verdict. An
  interrupt (SIGINT/SIGTERM) or the 50-consecutive-nonfinite abort landing
  while the probe read FAIL lost the checkpoint entirely — contradicting the
  abort message's own promise ("the final checkpoint is still saved"). Worse,
  the FAIL inside `finally` set `abort_without_save`, which RELABELED a
  nonfinite abort as `abort_reason="collapse_canary"`, masking the real cause.
  Pre-fix evidence (pre-4.1 trainer + new gates): nonfinite Check 5 →
  `abort_reason='collapse_canary'`, checkpoint missing, probe consulted 1×;
  signals Check 5 → checkpoint missing, probe consulted 1×.
- **Change** (`training/train_deep_cfr.py`). New nested helper
  `save_emergency_checkpoint(iteration, reason)`: a direct `save_checkpoint`
  to `args.save_path` that never consults the canary (also avoiding ~150
  probe inference calls inside a SIGTERM grace window where a supervisor may
  escalate to SIGKILL). The `finally` block uses it whenever
  `interrupted["flag"] or nonfinite_abort` — unconditionally, even if a
  periodic checkpoint already covered that iteration, so `save_path` always
  ends holding the latest state with the correct final `shadow_only` stamp.
  The stamp decision + loud warning moved into a shared `final_shadow_stamp()`
  helper used by both save paths. Emergency saves never touch the `.safe`
  copy — the deploy-grade artifact remains the canary-vetted save/safe pair.
  Unchanged on purpose: a mid-run probe FAIL at a PERIODIC checkpoint still
  aborts WITHOUT saving (that probe actually ran against the live network;
  `sanity_train_deep_cfr_abort.py` passes unmodified).
- **Gates (canary ENABLED — no `--disable-collapse-canary` anywhere).**
  `sanity_deep_cfr_nonfinite_guard.py` Check 5: threshold abort with
  `quick_canary_probe` stubbed to a guaranteed-FAIL verdict plus a call
  counter → status stays `aborted/nonfinite_loss`, checkpoint written at
  iteration 50 with the skip counter, probe call count must be 0.
  `sanity_train_deep_cfr_signals.py` Check 5: same proof on the SIGINT path.
  The signals gate's subprocess launches also dropped
  `--disable-collapse-canary` entirely (periodic checkpoints never fire at
  the test intervals and emergency saves are canary-free by design, so the
  runs stay deterministic without the smoke-test flag).

## 4.1-2 — Signal iteration accounting off-by-one

- **Issue (verified REAL).** `final_iter = min(t, args.iterations) if 't' in
  dir() else start_iter` reused the loop variable. A signal during iteration
  t breaks at the TOP of iteration t+1, where the loop variable already reads
  t+1 → report and checkpoint metadata claimed t+1 with only t completed, and
  `--resume` then started at t+2, silently skipping one iteration (it also
  claimed `start_iter + 1` when the signal landed before any iteration ran).
  Pre-fix evidence: signals Check 5 reported `final_iter=8` for a SIGINT
  raised during iteration 7.
- **Change.** Explicit `completed_iter` counter: initialized to `start_iter`,
  set to `t` immediately after iteration t's traversal returns (the gradient
  step and checkpoint that follow are interval-based aggregates over the
  buffers, not part of "did iteration t run"). `final_iter = completed_iter`
  feeds every report, result dict, and checkpoint stamp. Natural completion
  (== `--iterations`), the nonfinite abort (still exactly 50), and the canary
  abort are numerically unchanged.
- **Gate.** signals Check 5: the traversal stub raises a REAL signal
  (`os.kill` to its own pid) while iteration 7 runs → the handler sets the
  flag, iteration 7 finishes, the loop breaks at the top of 8 → asserts
  `result["final_iter"] == 7` and checkpoint `iteration == 7`, with the
  shadow stamp preserved.

## 4.1-3 — Portable gates

- **Issue (verified REAL).** All four new gates hard-coded
  `/Users/jaroslavaupart/...` (three as a `sys.path.insert`, the signals gate
  as its `REPO` constant) — they would fail in any other clone.
- **Change.** All four derive the repo root from
  `os.path.dirname(os.path.abspath(__file__))` (the gates live at the repo
  root). Verified: no EXECUTABLE changed-or-new file in this commit (trainer
  + gates) contains an absolute machine path — this audit document quotes
  `/Users/jaroslavaupart/...` above as prose evidence of the pre-fix bug,
  which is harmless to portability (precision fixed in 4.2; the original
  wording claimed "no file"). NOTE: 13 OLDER sanity files from previous phases share the
  same hard-coded pattern (e.g. `sanity_train_deep_cfr_abort.py`,
  `sanity_deep_cfr_fold_collapse.py`, `sanity_icm_payouts.py`); left
  untouched here as out of Phase-4 scope — a trivial follow-up sweep.

## Validation (4.1, full battery re-run after the fixes)

| Gate | Result |
|---|---|
| `sanity_validation_ladder.py --path deep-cfr` | **23 passed, 0 failed** (37.7s) |
| `sanity_validation_ladder.py --path cfr` | **21 passed, 0 failed** (55.5s) |
| `sanity_deep_cfr_history_parity.py` | PASS |
| `sanity_deep_cfr_curriculum.py` | PASS |
| `sanity_deep_cfr_nonfinite_guard.py` (5 checks, canary enabled) | PASS — Check 5 FAILs on the pre-4.1 trainer (`abort_reason='collapse_canary'`, checkpoint missing, probe consulted 1×) |
| `sanity_train_deep_cfr_signals.py` (5 checks, no `--disable-collapse-canary`) | PASS — Check 5 FAILs on the pre-4.1 trainer (`final_iter=8` for 7 completed, checkpoint missing, probe consulted 1×) |
| `sanity_train_deep_cfr.py` | ALL PASS |
| `sanity_deep_cfr.py` | ALL PASS |
| `sanity_deep_cfr_fixes.py` | ALL PASS |
| `sanity_train_deep_cfr_abort.py` | ALL PASS (periodic canary FAIL still aborts without saving — `checkpoint_saved=None`) |
| `sanity_action_history.py` | ALL PASS |
| `py_compile` (trainer + 4 gates) | OK |
| `git diff --check` | clean |

# Phase 4.2 addendum — Codex P1 follow-up (2026-06-12, pre-push)

## 4.2-1 — Emergency exits outrank a canary abort

- **Issue (verified REAL — Codex P1).** When SIGINT/SIGTERM arrived WHILE a
  periodic collapse-canary probe was running and that probe then FAILed,
  `checkpoint_with_canary` raised, the except handler set
  `abort_without_save`, and the `finally` block checked `abort_without_save`
  FIRST — so the emergency branch never ran. Meanwhile the result tail checks
  `interrupted["flag"]` FIRST. Net effect: the run reported
  `status="interrupted"` with a valid `final_iter`, yet saved NOTHING — the
  exact gap the 4.1 "unconditional emergency checkpoint" guarantee was meant
  to close.
- **Change.** Reordered the `finally` branches in
  `training/train_deep_cfr.py`: `interrupted["flag"] or nonfinite_abort` is
  now checked BEFORE `abort_without_save`, so interrupt and nonfinite-abort
  exits ALWAYS take the canary-free `save_emergency_checkpoint` path. A
  periodic canary FAIL with no emergency active still aborts without saving
  (unchanged — `sanity_train_deep_cfr_abort.py` passes unmodified), and the
  FAILing probe still blocks `.safe`/`.warn` promotion in the double-flag
  case (the emergency save only refreshes `args.save_path`). When both flags
  are set, a post-mortem NOTE line records that a canary FAIL was also
  observed.
- **Gate.** signals Check 6 (`interrupt_during_failing_canary_check`): the
  probe stub raises a REAL SIGINT mid-probe, then returns a guaranteed-FAIL
  verdict at the first periodic checkpoint (iteration 5). Asserts
  status="interrupted" + signal=SIGINT (not a canary abort), `final_iter == 5`,
  emergency checkpoint at `--save-path` with `iteration == 5` +
  `shadow_only: true`, `result["checkpoint_saved"]` set, NO `.safe`/`.warn_5`
  artifact, and exactly 1 probe call (the periodic one — the emergency save
  adds none). Canary ENABLED; no `--disable-collapse-canary`.
- **Pre-fix evidence (gate run against the 4.1 trainer before the reorder).**
  `status='interrupted', signal=2, final_iter=5, canary_probe_calls=1` and
  "[FAIL] — no checkpoint saved: the FAILing canary blocked the emergency
  save (pre-4.2 bug)" — all other assertions passed, isolating the
  precedence bug exactly.

## 4.2-2 — Doc precision

- The 4.1-3 claim "no changed-or-new file in this commit contains an
  absolute machine path" was too broad: THIS audit document quotes
  `/Users/jaroslavaupart/...` as prose evidence of the pre-fix bug. Reworded
  to scope the claim to EXECUTABLE files (trainer + gates), where it is
  verified true.

## Validation (4.2, full battery re-run after the fix)

| Gate | Result |
|---|---|
| `sanity_validation_ladder.py --path deep-cfr` | **23 passed, 0 failed** (35.0s) |
| `sanity_validation_ladder.py --path cfr` | **21 passed, 0 failed** (51.2s) |
| `sanity_train_deep_cfr_signals.py` (now 6 checks, canary enabled) | ALL PASS — Check 6 FAILs on the pre-4.2 trainer (checkpoint missing) |
| `sanity_deep_cfr_nonfinite_guard.py` | ALL PASS |
| `sanity_deep_cfr_history_parity.py` | ALL PASS |
| `sanity_deep_cfr_curriculum.py` | ALL PASS |
| `sanity_train_deep_cfr.py` | ALL PASS |
| `sanity_deep_cfr.py` | ALL PASS |
| `sanity_deep_cfr_fixes.py` | ALL PASS |
| `sanity_train_deep_cfr_abort.py` | ALL PASS (no-emergency canary FAIL still aborts without saving) |
| `sanity_action_history.py` | ALL PASS |
| `py_compile` (trainer + signals gate) | OK |
| `git diff --check` | clean |

# Phase 4.3 addendum — shadow-phase canary maturity (2026-06-12)

The first fresh Phase-4 curriculum run exposed a canary-policy mismatch at
iteration 5,000: the run was still in the all-in shadow phase (all-in excluded
from regret targets and masked at inference), but the raw/search all-in canary
was already blocking. Shared-encoder updates can therefore make the
intentionally untrained all-in row transiently dominate and abort a healthy
run before the row becomes meaningful. Historical logs confirm the behavior
was seed-dependent: one earlier run passed 25k and falsely aborted at 30k,
while another seed survived the same shadow phase.

**Change:** all canary metrics are diagnostic before
`all_in_deploy_iteration` (150k by default), and the worst would-be status is
printed. Periodic checkpoints continue to update the resumable primary/safe
artifacts. At and after the deploy boundary, the complete canary is enforced
unchanged: WARN writes only a side checkpoint and FAIL aborts without saving.
Mid-run FAIL handling now also prints the full exception with exact metrics and
threshold reasons instead of only the generic final footer.

**Regression coverage:** `sanity_train_deep_cfr_abort.py` now proves both
sides with a forced 99% all-in probe: pre-deploy completes and saves; mature
training aborts without saving and returns the documented result. The signal
race gate forces its canary to maturity so Phase 4.2's double-flag behavior
remains covered.
