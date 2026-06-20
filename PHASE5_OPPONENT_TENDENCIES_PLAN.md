# Phase 5 — Opponent Tendencies (Implementation Plan)

> Status: **v1 IMPLEMENTED in the working tree (uncommitted), OFF by default.**
> All 34 sanity checks pass (`.venv/bin/python sanity_tournament_hybrid.py`,
> CHECK 20–34 cover Phase 5). The code matches this v1 scope: `OpponentProfiles`
> substrate, **station suppression** + **strict R2 all-in-call relief** active only
> when enabled, every other read **log-only**. `p5_enabled` defaults `False` (normal
> `final_survival`/`final_aggro` play is unchanged); Phase 5 is turned on only via
> the eval ablation arms in `bots/__init__.py` (`telemetry`/`station`/`r2`/`p5`).
> Remaining work is the **eval campaign + default-ON decision** (§7), not the code.
> Bot: `bots/tournament_hybrid_bot.py` (`TournamentHybridBot`, profiles `survival` / `aggro`).
> Phases 1–4 are complete **in code** (legal-safety, preflop ranges, postflop
> MC-equity vs pot odds, tournament stack-rank logic + future-edge tax).
>
> **Provenance.** Designed via a `grill-me` interview conducted against ChatGPT Pro
> in the browser, then hardened by two adversarial reviews: an internal
> code-grounded multi-agent review, and a fresh-context ChatGPT second opinion.
> Both reviews are folded directly into the body below; their findings and the
> decisions they changed are recorded in **Appendix A**. All numeric constants are
> **proposed defaults, subject to eval tuning** — the structure is the commitment.

---

## 0. Objective and the v1 scope decision

### 0.1 Objective (from `CLASS_TOURNAMENT_BOT_PLAN.md`, Phase 5)
Track simple per-opponent stats (aggression freq, call freq, fold/passivity,
large-bet freq) and use them **lightly**: bluff less into calling stations, steal
more vs folders, trap/call stronger vs maniacs. Plus the two items Phase 4
deferred here:
- **R2** — stop folding **+chipEV all-in CALLS** to opponents who overjam (the #1
  EV leak in WTA: there is no prize for letting a bad bot punt chips to someone
  else).
- **R8** — turn Phase 4's blind one-gear chip-leader/bluff nudges into **measured**
  exploits.

Acceptance: adapts within a match with no external data; resets cleanly between
tournaments.

### 0.2 The governing principle: downside asymmetry under censored data
Two facts dominate the entire design:
1. **The observation stream is badly censored and policy-conditioned** (§1.2). Reads
   built on *folds* are unreliable (we never see the folds that end a hand to our
   own bet); reads built on *observed calls* are reliable.
2. **Winner-take-all is downside-asymmetric.** A wrong chip-risking decision (a bad
   all-in call, an over-aggressive steal into a reshove stack) can end the
   tournament; the upside of a right one is a few chips. 2nd pays the same as 6th.

So Phase 5 splits its exploits by *risk direction* and ships them in that order:

| Exploit | Direction | Reliability | v1 status |
|---|---|---|---|
| **Station suppression** (bluff less vs proven callers) | reduces risk | high (built on *observed calls*) | **ACTIVE** |
| **Strict all-in-call relief** (R2: don't fold +chipEV calls vs confirmed spewy jammers) | takes +chipEV risk under a hard gate | medium (built on *observed jams*) | **ACTIVE, dedicated gate** |
| Overfolder offensive widening / aggro 2nd gear | adds risk | low (built on *folds*, half censored) | **log-only** |
| Passive/tight action-conditional widening | adds risk | low | **log-only** |
| Thinner value vs stations | adds marginal risk | medium | **log-only** |
| Preflop R2 call-range widening | adds tournament-life risk | low (range invention) | **deferred (§9)** |

**v1 = `OpponentProfiles` telemetry + station suppression + strict all-in-call
relief. Everything else is computed and logged but does not move a decision** until
eval proves the read is real and not a tournament-phase artifact (§7). The full
exploit set is preserved as the deferred design in **§9**.

---

## 1. Engine constraints and cross-cutting principles (LOCKED)

### 1.1 Hard engine facts (verified in code)
1. The bot only implements `act(view)`; it is called **only on its own turn**.
   There is **no showdown / hole-card reveal hook** → reads are
   **action-frequency only**.
2. `view.history` is **per-hand** (re-initialised every hand, `core/engine.py:408`).
   Entry: `{street, pid, type ∈ fold/check/call/bet/raise, amount, to_call_before,
   pot_before}`. For `bet`/`raise`, `amount` is the **raise-TO total**; for `call`
   it is **chips actually paid** (`core/engine.py:801-811`). Cross-hand tendencies
   must be accumulated in the bot's own counters.
3. `view` also exposes `stacks`, `opponents`, `acting_opponents`,
   **`all_in_opponents`**, `seat_indices`, `hand_id` (`core/bot_api.py:11-29`).
   There is **no `session_id`** — dedup keys on `(hand_id, index)` (§2.2).
4. `reset_memory()` is the tournament boundary (wired at the GUI
   `run_tournament.py:320`). The eval path (`core/tournament.py`) does **not** call
   it — `eval_final_bot.py` instead constructs a **fresh bot per tournament**; that
   "fresh bot per tournament" is the reset contract (CHECK 31).

### 1.2 The censored-fold principle (the most important constraint)
There is no hand-end hook, so hero only sees actions up to its own turn. **Folds
(and calls, and all-in calls) that *end the hand* to hero's own bet are never
observed.** Counting only the calls/raises we do see biases every opponent toward
looking **stickier than they are**. Therefore:
- **Never infer a fold from silence.** Attribute a fold only when present in the
  observed prefix.
- **Asymmetric, conservative labels.** "Calling station" needs *direct observed
  call* evidence; "overfolder" needs *direct observed fold* evidence. We accept
  under-crediting folders — which is exactly why offensive (fold-based) exploits
  are log-only in v1.
- **No statistical debiasing** (no fold-upweighting, no tilted prior). Censoring is
  structurally correlated with hero's own aggression; a correction would be fake
  precision.

### 1.3 State-confounding (the biggest risk — design around it)
A per-opponent action rate blends **personality** with stack depth, table size,
position, street, whether hero is still in the hand, the bot's own policy, blind
level, and tournament phase. In a fast-blind 6-WTA, stack depth swings behavior
hard (nitty deep → shove-heavy short). Untreated, Phase 5 can "adapt" to
**tournament phase** rather than **opponent weakness** — a failure mode that passes
a toy stress field and loses to real bots. Mitigations: (a) the v1 active reads are
the least confounded (station = postflop calls; R2 = explicit jam spot with a
`<12bb` exclusion); (b) telemetry **records stack-depth context** with every read
so confounding is measurable before any offensive module is activated;
(c) stack-depth context separation is a hard requirement before §9 modules go live.

### 1.4 Light-touch, deterministic, fail-closed
- Reads adjust **frequencies and thresholds only** — never invent ranges, never
  bypass the Phase 1 legal sanitizer, Phase 3 chipEV/pot-odds sanity, or Phase 4
  stack-risk gates.
- **Deterministic:** pure integer counters → `p_hat` is a pure function of observed
  history; reuse `_freq_gate`/`_stable_hash`; never Python's builtin `hash()`;
  canonical ordering for any opponent reduction (§5.3).
- **Neutral default == Phase 4:** zero data → `read_strength == 0` → no-op.
- **Fail-closed + logged:** any exception/non-finite intermediate inside Phase 5
  falls back to the exact Phase 4 decision and increments `p5_error_count` (a
  silent fail-closed bug otherwise looks identical to "Phase 5 is safe").

---

## 2. `OpponentProfiles` — the shared substrate (all v1)

A separate object (not a subclass of `core/ml_features.py:OpponentMemory`, which is
ML/CFR-facing and must keep stable feature semantics). It copies only the small
dedup/reset *pattern*.

### 2.1 Stat schema
**TIER 1 (computed in v1; only `station_score` and the jam stats *act* in v1):**
| Stat | Definition | Used by (v1) |
|---|---|---|
| `vpip` | `vpip_seen / preflop_action_seen`, once per `(hand, pid)`. Forced blind ≠ VPIP; BB check ≠ VPIP; SB complete/call = VPIP. | looseness conjunction (R2) |
| `preflop_aggression_rate` | `preflop_raise_seen / preflop_action_seen`, once per hand. | looseness conjunction (R2) |
| `postflop_aggression_freq` | `(bet+raise)/(bet+raise+call+check)` — **checks included**. | looseness conjunction (R2) |
| `fold_to_pressure` | `fold/(fold+call+raise)` facing genuine pressure (§2.3). | telemetry (overfolder, §9) |
| `station_score` | `call/(call+raise)` **facing real pressure only** — *no `(check+call)` fallback* (it conflates passivity with calling). | **station suppression (§3)** |
| `large_bet_count`, `jam_like_count` | split (§2.4). | **R2 spewy flag (§4)** |

**Log-only (computed, never acts in v1):** `blind_fold_to_steal` (structurally
censored — when hero opens, the blinds act after hero and their fold ends the hand
unseen; the only un-censored sample measures table fold-to-open, not fold-to-hero),
`fold_to_cbet`, `3bet`/`4bet`.

### 2.2 Accumulation, dedup, reset
On **every** `act(view)`:
```
rebuild hand_state by walking the full view.history IN ORDER
for each entry by index:
    classify the entry using the CURRENT hand_state (before applying it)
    if (hand_id, index) not in seen:
        seen.add((hand_id, index))
        if entry.pid != hero: update that opponent's counters
    apply the entry to hand_state            # hero actions stay in hand_state for context
```
- Reconstruct the prefix every call (labels need earlier same-hand actions); **only
  the counter increment is deduped.**
- **Once-per-hand stats** (`vpip`, `preflop_aggression_rate`) use per-`(hand, pid)`
  flags so limp→call→raise is **one** VPIP.
- **Reset:** `reset_memory()` clears all profiles + `seen`. Backstop: in `act()`,
  if `hand_id` regresses below the last-seen (single-table, no re-entry), auto-reset
  — a missed reset then fails toward "no data," never blends two tournaments.

### 2.3 Facing-pressure state machine (classify before applying)
```
postflop: facing_pressure = (to_call_before > 0) and type ∈ {fold, call, raise}
preflop : facing_pressure = (to_call_before > 0) and prior_preflop_raise_seen
                            and type ∈ {fold, call, raise}
```
Limp / SB-complete = VPIP but **not** pressure; BB check = neither.

### 2.4 Large-bet / jam detection (prefer the reliable signal)
- **Primary jam signal:** `view.all_in_opponents` observed at hero's decision points
  — when a villain whose last aggressive action is in the prefix appears all-in,
  count `jam_like_count` (this sidesteps the raise-TO-amount ambiguity entirely).
- **Secondary (overbet) signal:** `incremental_raise = amount − prior_contribution_
  this_street` (reconstructable from the per-street prefix);
  `risk_fraction = incremental_raise / effective_stack`. Coarse thresholds only.
- **`large_bet_count`** ≠ jam: a pot-sized value bet is not spew. Keep the two
  counters separate.
- **Hard exclusion:** a shove at `< ~12bb` is **standard short-stack poker**, not
  spew — never counts toward `jam_like_count`.

### 2.5 Confidence (shrinkage, gated)
```
p_hat = (k + p0 * w) / (n + w)          # k positives, n opportunities
read_strength = clamp((p_hat − threshold) / band, 0, 1)
delta = max_delta * read_strength       # do NOT also multiply by n/(n+w) — double-damping
```
- `w` ≈ 10 default (the §16-R8 "lower w to avoid a no-op" worry applied to the
  *offensive* modules, which are log-only in v1; the active modules don't need it).
- **Shrinkage alone is too eager** (with `p0=0.45, w=6` three folds already clear
  0.58). So each *active* module carries an **opportunity guard**:
  - **Station suppression** may be relatively eager — it pulls bluffs *down*, a safe
    direction — but still requires `pressure_response_n ≥ 4`.
  - **R2** risks chips → strict guards (§4).
- Fixed neutral priors; **no self-calibration** (a current-field average is
  policy- and stack-state-biased → feedback loops).

---

## 3. v1 ACTIVE module 1 — Station suppression

**Read:** `station_score_hat ≥ 0.60` with `pressure_response_n ≥ 4` (direct calls
facing real pressure; no `(check+call)` fallback).

**Action:** scale **down** our own bluffing into that villain:
- pure-bluff frequency × ~0.5–0.6;
- generic cbet frequency × ~0.7–0.8;
- **spare semibluffs with real equity** (don't kill profitable equity-realization);
- **never bluff-raise** a confirmed station;
- value behaviour unchanged in v1 ("thinner/bigger value" is log-only — §9).

This is the safest exploit: it only ever *reduces* aggression (clamp floor 0, can
pull below the Phase 4 baseline), and it is built on directly observed calls, which
the censoring does not hide.

Profiles: both suppress; aggro may use a marginally lower threshold. Suppression is
the one place survival and aggro behave nearly identically.

---

## 4. v1 ACTIVE module 2 — Strict all-in-call relief (R2)

The highest-EV correction, but the highest-variance one — so it gets its **own
dedicated gate**, not the generic composition path (§5). It only ever changes the
**future-edge tax on a postflop +chipEV all-in CALL**; it never widens a range and
never initiates a jam.

**It fires only when ALL hold:**
1. hero faces an **all-in call** (`to_call ≥ hero_stack`) — *postflop* (preflop
   call-offs already hardcode `future_edge_tax = 0`; widening their range is range
   invention and is **deferred**, §9);
2. **heads-up** in the pot (exactly one non-folded, non-all-in villain besides the
   shover);
3. the **sole current aggressor's pid == a confirmed-spewy pid**;
4. **confirmed spewy** under the strict rule below;
5. hero's call is **non-negative chipEV** under the Phase 3 MC-equity estimate.

**Confirmed-spewy (strict):**
```
confirmed_spewy =
    jam_opportunity_n ≥ 3                       # real denominator, not raw count
    AND looseness_conjunction                   # high vpip / preflop_aggr / postflop_aggr
    AND ( jam_like_count ≥ 2 OR large_bet_count ≥ 3 )
    AND none of the counted jams were < ~12bb short-stack shoves
```
**Effect:** apply **partial** tax relief first (`spewy_tax_relief` < 1.0, e.g.
halve the tax), **not** an automatic ×0 — eval can push it to full later. Because
this is its own gate it is **exempt** from the §5.2 "no positive widening when
all-in-adjacent" rule (that rule is what would otherwise suppress R2 in exactly the
spot it targets — the contradiction §17 caught).

---

## 5. Composition, precedence, determinism

### 5.1 Compose by clamped addition, never multiply
For the postflop frequency lever (`_apply_freq_nudges` is multiplicative-then-
clamped to one gear cap), P5's contribution composes as
`final = clamp(p4_delta + p5_delta, …)`. In v1 the only active P5 frequency effect
is **suppression** (negative), which is always safe (clamp floor 0). Positive
widening is log-only, so the "positive widening shares P4's single one-gear cap"
subtlety doesn't bite in v1 — but it is recorded for §9.

### 5.2 Precedence (safety dominates; R2 is a separate gate)
1. Phase 1 legal sanitizer
2. Phase 4 stack-risk gate (applies to *widening* only)
3. station suppression (active)
4. **R2 all-in-call relief — its own gate (§4), exempt from step 2**
5. *(log-only in v1: multiway-contamination gate, passive/tight respect, overfolder
   widening — computed, not applied)*

### 5.3 Determinism
- Any reduction over opponents (jam aggressor, looseness check) sorts by
  `seat_indices` (fallback: pid string) — never dict/set insertion order.
- `p_hat`/`read_strength` are floats: require `band > 0`; route any non-finite value
  to the fail-closed path (`_clamp` does **not** catch NaN). Quantize/epsilon at
  thresholds.
- Profiles keyed by pid; `(hand_id, index)` dedup; reset + regression backstop (§2.2).

---

## 6. Testing (`sanity_tournament_hybrid.py`, new CHECK 20–34)

Reuse the `_view` / `_table_overrides` / `_postflop_view` helpers. CHECK 1–19 must
still pass.

**Substrate (active in v1):**
- **20** accumulation + dedup (overlapping prefixes counted once).
- **21** once-per-hand VPIP (limp→call→raise = 1).
- **22** facing-pressure state machine (limp/SB-complete not pressure; post-open is; BB check neither).
- **23** censoring discipline, sub-cases: (a) hero bets last + villain response absent → `fold_to_pressure`/`station` counters unchanged; (b) only-folded villain is never a station; (c) only-called villain is never an overfolder; (d) replay never increments an absent fold.
- **25** neutral default + reset (zero data == Phase 4; `reset_memory()` clears profiles).
- **26** idempotent prefix replay — assert **counter byte-equality** (the action clause is insufficient: `_stable_hash` ignores history); include an intermediate view with an extra action to prove `(hand_id, index)` keying.
- **27** P5-off / log-only == **byte-for-byte Phase 4 parity** over CHECK 1–19: scope to Action + the pre-existing Phase-4 trace key set; new keys use a `p5_` namespace; include a malformed-history view; assert the Phase-4 key set is unchanged.
- **31** reset across back-to-back tournaments (fresh-bot contract; no leakage).
- **33** fail-closed: a subclass raising inside `_p5_*` returns the exact `p5_enabled=False` action and increments `p5_error_count` by one.
- **34** every stat is neutral at zero data (`read_strength == 0` explicitly).

**Active modules:**
- **24a** station suppression direction + magnitude (bluff/cbet down; semibluffs spared; never bluff-raise; value unchanged in v1).
- **30** R2 scope: tax relief fires **only** on a heads-up postflop +chipEV all-in CALL vs a confirmed-spewy *aggressor* pid; **not** on hero open-jam, non-all-in raise, multiway, thin call vs a tight villain, a `<12bb` short-shove read, or a negative-chipEV call. Assert partial (not ×0) relief.

**Log-only guards (prove they don't move actions in v1):**
- **28/29/32** multiway contamination / passive-tight flip / multi-read precedence — assert the proposed deltas are **traced but not applied** while their modules are log-only (these become active-behavior tests in §9).

---

## 7. Evaluation / A-B

Harness: `eval_final_bot.py` and `run_eval.py --mode multiway` (locked params:
1000 chips, 5/10, +1.5× every 50 hands, ante off, 6-player WTA).

**Module-level ablation arms** (so a harmful module cannot hide behind a helpful one):
`P4`; `P4 + telemetry-only`; `P4 + station-only`; `P4 + strict-R2-only`;
`P4 + station + R2` *(the v1 candidate)*. (The §9 offensive modules get their own
arms only when promoted.)

**Method & rigor:**
- **Log-only pre-gate first:** before any win-rate arm, run telemetry-only and
  confirm the active reads fire on the intended villains at a sane rate and that
  `read_strength > 0` actually occurs in deep/medium spots (else the modules are
  near no-ops and the next step is tuning, not shipping).
- **Stress field:** nit/folder, maniac/jammer, calling-station, loose-passive,
  min-raiser (master-plan Phase 7).
- **Paired seeds / common random numbers** (same table composition, seat schedule,
  deck seed across arms) + bootstrap by seed block. ~5k/arm → ±1.6pp, 10k → ±1.1pp,
  20k → ±0.8pp. **Primary metric = tournament win rate.**
- **Locked holdout:** tune thresholds on *dev* seeds; make the ship decision **once**
  on separate *locked* seeds with seat rotation (avoid harness overfit).

**Ship gates (all required):**
- `p5_error_count == 0` across the full eval.
- **Station suppression:** non-inferior overall (CI lower bound > −0.5pp) and
  reduces bluff-into-station frequency materially.
- **R2:** the targeted metric improves (count of +chipEV all-in calls previously
  folded; realized all-in EV-when-called up) with no archetype regression beyond
  tolerance.
- **aggro** default-ON only if the combined arm's win-rate-delta CI lower bound > 0.

---

## 8. Implementation order (v1)

1. `OpponentProfiles`: ingest + `(hand_id, index)` dedup + per-`(hand,pid)` flags +
   facing-pressure state machine + reset + regression backstop; wire `_p5_ingest`
   into `act()` (always, even disabled) and `_profiles.reset()` into
   `reset_memory()`. **No behavior change** (`p5_enabled=False`). CHECK 20–23, 25–27,
   31, 33, 34.
2. Confidence layer (`p_hat`, read strength, opportunity guards, NaN/`band>0` guard).
3. **Station suppression** active (direct pressure-call evidence; no fallback). CHECK 24a.
4. **Strict R2** all-in-call relief as its own gate (§4), **partial** relief. CHECK 30.
5. Compute all §9 reads as **log-only** (traced, not applied). CHECK 28/29/32 in their not-applied form.
6. Telemetry + full regression (CHECK 1–19).
7. Eval: log-only pre-gate → ablation arms on locked holdout seeds → decide default-ON.

Each step lands behind passing tests before the next; the engine stays frozen.

---

## 9. Deferred design (Phase 5.5+) — the full exploit set, log-only until proven

These were fully designed in the interview and remain the **target**; they ship
only after their log-only telemetry (§7) shows the reads are real, stable across
stack-depth bands, and not tournament-phase artifacts. Promoting any of them
requires **stack-depth context separation** (§1.3) and its own ablation arm.

- **Overfolder offensive widening (R8).** Read: `fold_to_pressure_hat ≥ 0.58`
  (full ~0.75) with **direct visible folds** required. Action: widen
  steal/cbet/bluff **frequency** (not range-width); survival ≤ ×1.25 in-gear, aggro
  ≤ ×1.50 with a bounded second gear. Gated by spot-risk: low-risk steal → normal
  cap; medium → one gear; high-risk/all-in-adjacent → none. Hard stack-risk gates
  (zero widening when eff. villain ≤10bb, reshove-friendly ≤15bb, multiway-
  contaminated, or all-in-adjacent unsupported by chipEV). Target medium/deep
  stacks that can fold without being forced all-in. **Reason deferred:** fold-based
  → heavily censored; adds variance in a WTA.
- **Passive/tight action-conditional widening.** Steal more *before* the villain
  resists; when this villain bets/raises, **flip** — reduce marginal bluff-catching,
  fold marginal bluff-catchers more, value-jam less thinly. Never let "tight folds
  a lot" leak into call-downs. **Reason deferred:** same censoring + the flip must
  be proven not to misfire.
- **Thinner/bigger value vs stations.** Allowed only as a small modifier on existing
  Phase 3 value classes (if P3 already says value → modestly raise freq/sizing; if
  P3 thin-value near threshold → lower the threshold slightly). Must never infer
  villain ranges or create new value classes. **Reason deferred:** small EV, real
  overfit/creep risk.
- **Preflop R2 call-range widening.** Widening `_CALL_SHOVE_RANGES` for confirmed
  spewy villains is **range invention** without explicit per-spot equity; it risks
  tournament life on the noisiest label. Ship only with explicit preflop chipEV/range
  logic. **Reason deferred:** highest variance, lowest reliability.
- **Pooled / table-level field stats** (e.g. `field_is_overfolding_blinds`),
  **board-texture-conditioned reads**, `fold_to_cbet`/`3bet`/`4bet` as active stats,
  **min-raiser** archetype exploitation, and any **CFR/Deep-CFR advisor** (Phase 6).

**Magnitude reference (first-draft, for the deferred modules):** `p0` VPIP 0.32,
fold_to_pressure 0.45, postflop_aggression 0.33, station_score 0.43; overfolder band
0.58→0.75. All eval-tuned.

**Final guardrail (all phases):** Phase 5 may adjust frequencies/thresholds but must
**never** bypass Phase 1 legality, Phase 3 chipEV sanity, or Phase 4 stack-risk gates.

---

## 10. Open items for review

- **O1 — Partial-relief magnitude.** v1 R2 uses `spewy_tax_relief` < 1.0 (e.g. 0.5).
  Start value vs let eval pick? (Recommendation: 0.5, eval to 0/1.)
- **O2 — Default-ON policy.** Ship v1 off-by-default and enable per §7 gates, or
  survival-on/aggro-off? (Recommendation: off until the gate is met, then survival-first.)
- **O3 — `jam_like` reconstruction.** v1 prefers `all_in_opponents`; confirm the
  per-street `incremental_raise` reconstruction is worth adding for non-all-in
  overbets, or rely on `all_in_opponents` + a coarse overbet proxy only.
- **O4 — Stack-depth banding granularity** for §9 promotion (deep/medium/short vs finer).

---

## Appendix A — Design provenance & review history

1. **Design interview** (`grill-me` vs ChatGPT Pro, in-browser), branches A–J:
   stat schema, accumulation/dedup, confidence, exploit mapping, profile
   differentiation, magnitudes, safety, testing, eval, scope. Key corrections it
   produced over the first draft: AF must include checks; split fold-vs-station;
   no double-damping; rebuild-prefix-then-dedup; separate `OpponentProfiles`;
   spewy trigger is the most dangerous number; clamp-add (never multiply); fixed
   priors (no self-calibration); survival is variance-aware chipEV (still steals cheap).
2. **Internal code-grounded review** (multi-agent; code-accuracy lens found **0**
   reference mismatches). Fixes folded in: drop non-existent `session_id` (key on
   `(hand_id, index)` + reset + regression backstop); fresh-bot-per-tournament reset
   contract; R2 heads-up + aggressor-matched; preflop R2 ≠ tax removal; NaN/`band>0`
   guard; canonical villain ordering; demote `blind_fold_to_steal` to log-only; the
   eval/test rigor (CHECK 26/27/30/31/32/33/34).
3. **Fresh-context ChatGPT second opinion** (no prior involvement). Drove the **v1
   scope cut** (this document's spine): ship only telemetry + station suppression +
   strict R2; demote the offensive modules to log-only. New catches beyond review 2:
   R2 contradicts its own all-in-adjacent stack-risk gate → dedicated gate (§4/§5.2);
   `station_score` fallback conflates passivity with calling → removed (§2.1);
   shrinkage alone fires too eagerly → opportunity guards, eager-OK only for safe
   suppression (§2.5); spewy must exclude `<12bb` shoves + need a jam-opportunity
   denominator (§4); jam detection via incremental amount / `all_in_opponents` (§2.4);
   **state-confounding** is the biggest risk → context separation + telemetry (§1.3);
   module ablations + locked holdout + `p5_error_count==0` ship gate (§7).
