# Phase 4 — Tournament-Specific Logic (Implementation Plan)

> Status: **PLAN ONLY — not implemented.** For review.
> Bot: `bots/tournament_hybrid_bot.py` (`TournamentHybridBot`, profiles `survival` / `aggro`).
> Phases 1–3 (legal-safety, preflop ranges, postflop MC-equity vs pot odds) are complete.
> Design decisions below were resolved in a structured design interview; the
> game-theory framing and concrete magnitudes were pressure-tested against an
> external model (ChatGPT Pro). All numeric constants are **proposed defaults,
> subject to eval tuning** — the structure is the commitment, the numbers are knobs.
>
> ⚠ **An independent adversarial review (§12) revises several values below.**
> §12 is authoritative where it conflicts with §1–§11; items it supersedes are
> marked `[see §12]`.

---

## 0. Objective (from `CLASS_TOURNAMENT_BOT_PLAN.md`, Phase 4)

Add:
- Stack-rank detection
- Chip-leader pressure mode
- Short-stack desperation mode
- Heads-up aggression mode
- Early-bust avoidance for the survival profile

**Acceptance (from the master plan):**
- Survival avoids marginal tournament-life calls early.
- Aggro pressures when chip leader / top-2 stack.
- Both profiles loosen heads-up.
- Short-stack mode does not blind down forever.

---

## 1. Foundational design decisions (LOCKED)

### D1 — Risk model: winner-take-all ⇒ NO ICM ladder logic
The format is winner-take-all (2nd pays the same as 6th: zero). Therefore
`tournament_value ≈ hero_chips / total_chips`, so **+chipEV ≈ +tournamentEV**.
There is **no** ICM/ladder reason to fold a +chipEV spot.

The *only* legitimate reason to fold a thin +chipEV stack-off is a
**future-edge tax**: we expect to out-play simple class bots over time, so we
decline razor-thin gambles that risk our tournament life while that edge is
still ahead of us. This **replaces** the current `_commit_tax` survival overlay
with a properly-gated version.

The future-edge tax is applied **only** when ALL of these hold:
1. Many players remain, AND
2. stacks are deep enough that future edge matters, AND
3. the decision risks a large fraction of hero's stack, AND
4. the edge is thin, AND
5. hero is **not** already desperate.

It goes to **zero** heads-up, **zero** when `hero_bb ≤ 10`, and fades to
near-zero 3-handed. `[see §12: R1 lowers the cap and makes it A/B-gated; R3
revises the blanket zero-tax-heads-up rule; R2 caps the tax on all-in CALLS.]`

### D2 — Two independent axes: absolute depth vs relative rank
- **Absolute BB band** (existing: short ≤ ~12bb / medium ≤ 25bb / deep > 25bb)
  chooses the **strategic mode** (shove-fold vs open/raise vs postflop). This
  remains the dominant mechanical driver — **unchanged**.
- **Relative stack rank** (`chip_leader` / `top_2` / `middle` / `short_stack`)
  only **nudges frequencies inside** the mode the band already chose.

**Critical rule:** "short by rank" ≠ "short in BB". A 40bb last-place stack is
*behind, not desperate*; a 6bb 2nd-place stack is *desperate regardless of
rank*. **Desperation (`hero_bb ≤ ~10–12`) overrides rank.**

### D3 — Nudges never exceed "one tactical gear"
Every mode adjustment is bounded so the bot can never be pushed more than one
gear from its Phase 1–3 baseline. All adjustments are **deterministic** (no new
RNG; reuse the existing `_freq_gate` / `_stable_hash` mechanism) and every
action still passes the Phase 1 legal-action sanitizer.

### D4 — Correction: gate each mode with the RIGHT weight
- **Rank-based effects** (chip-leader pressure) gate on
  `rank_weight = clamp((min(hero_bb, eff_bb) − 10) / 20)` → 0 at ≤10bb, ~0.5 at
  20bb, 1.0 at ≥30bb. (Rank only matters when deep.)
- **Short-stack desperation** must **NOT** use `rank_weight` (it is 0 exactly
  when desperation must be strongest). It gates on `hero_bb` directly.
- **Heads-up postflop** gates on `heads_up_weight = clamp((eff_bb − 8) / 17)`.

---

## 2. New state: stack-rank context

### 2.1 Metric — hybrid (ordinal rank + magnitude guards)
Primary signal is **ordinal rank by stack**, with magnitude guards so "1st by 1
chip" ≠ "dominant chip leader". **Not** percentile (in a 6-field it is just rank
with worse readability and ignores magnitude).

Inputs, all derivable from `PlayerView.stacks` (a `Dict[str,int]` of every
seated player — see `core/bot_api.py:11`):
- `rank` — ordinal position by stack among **all tournament-alive players**
  (`stack > 0`), with a tie tolerance so near-equal stacks don't thrash buckets.
- `avg_ratio = hero_stack / average_stack`
- `leader_ratio = hero_stack / chip_leader_stack`
- `cover_count` — number of opponents hero's stack covers.

### 2.2 Buckets (6-handed thresholds)
| Bucket | Rule |
|---|---|
| `chip_leader` | rank 1 **AND** ≥120% avg **AND** ≥5% ahead of 2nd |
| `top_2` | rank ≤2 **AND** (≥95% avg **OR** covers ≥ half the opponents) |
| `short_stack` | rank 5–6 **AND** (≤75% avg **OR** ≤70% median **OR** ≤35% of leader) |
| `middle` | everything else (default) |

3-handed: looser cuts (`chip_leader` ≥110% avg; `short_stack` = rank 3 & ≤80%
avg or far behind leader). **Flat tables legitimately have no leader/short** —
buckets are not forced to be populated.

**Heads-up uses separate labels** `hu_leader` / `hu_trailer` / `hu_even` and
never the 4 buckets.

### 2.3 Pool & recomputation
- Compute the **global** bucket over **all tournament-alive players**, not just
  those live in the current hand (a button + SB folding must not make the BB the
  "chip leader of 1").
- **Recompute at the start of each hand; hold stable during the hand.** Mirror
  the existing per-`hand_id` cache pattern (`_bb_cache` at line 1350) with a new
  `_rank_cache: dict[hand_id, RankCtx]`. Cleared in `reset_memory()`.
- Recompute **hand-local** cover/commit facts (covers_villain, commit_fraction,
  stack_after_loss) **every decision** from current `view` fields.

### 2.4 Engine-field caveat (must verify before coding)
ChatGPT recommends ranking on `stack_behind + committed_this_hand` to avoid
blind-post jitter. `PlayerView` exposes **no `committed_this_hand` field**.
Mitigation: compute rank **once per `hand_id`** from `view.stacks` and cache it
(the small blind-post distortion is absorbed by ratio cuts + tie tolerance).
If jitter proves material in eval, reconstruct committed-this-hand from
`view.history`. **Open item O1.**

---

## 3. Code changes (file: `bots/tournament_hybrid_bot.py`)

### 3.1 Module-level tables (new constants near the Phase 3 tables, ~line 312–400)
Add, all keyed `profile → …`:
- `_RANK_BUCKET_WEIGHT = {"chip_leader":1.00, "top_2":0.55, "middle":0.0, "short_stack":0.0}`
- `_CHIPLEADER_PREFLOP` — per-spot `(relative_expansion, absolute_cap_pp)` for
  open/steal/iso/3bet/resteal, separate survival vs aggro (table A1 below).
- `_CHIPLEADER_POSTFLOP_MULT` — per profile/way/street multipliers for
  `CBET_FREQ` / `BLUFF_FREQ` / `SEMI_FREQ` (table A2 below).
- `_SHORT_DESPERATION` — per profile, `hero_bb`-band `(relative_expansion, abs_cap_pp)` (table B).
- `_SHOVE_CEILINGS` — per profile/position max first-in shove width.
- `_HU_POSTFLOP` — per profile/street multipliers + cushion/value-threshold deltas (table C).
- One-gear **clamp caps** per profile for each knob family (Section 6).

### 3.2 New helper methods
```
_rank_context(view) -> dict           # bucket, rank, avg_ratio, leader_ratio, cover_count; cached per hand_id
_rank_weight(hero_bb, eff_bb) -> float
_pressure_weight(view, rank_ctx, hero_bb, eff_bb, relevant_villains) -> float   # rank_weight * bucket_weight * cover_weight
_cover_weight(hero_stack, villain_stacks) -> float    # 1.0 cover-all, 0.7 ≥⅔, 0.4 ≥½, else 0
_short_weight(hero_bb) -> float       # desperation gate, NOT rank_weight
_heads_up_weight(eff_bb) -> float
_urgency_bonus(hero_bb) -> float      # equity-pp reduction for PROACTIVE shoves only
_future_edge_tax(view, ctx, hero_bb, eff_bb, commit_frac, players_left, rank) -> float   # replaces _commit_tax

# one-gear-bounded combinators (Section 6)
_apply_range_nudges(base_pct, nudges, profile, ceiling) -> float
_apply_freq_nudges(base_freq, mults, profile, knob_name) -> float
_apply_call_cushion(base, deltas, profile) -> float
_apply_value_threshold(base, deltas, profile, is_raise) -> float
```

### 3.3 Wiring into existing decision paths
- **`_context` (line 1282):** also attach `rank_ctx` (or compute lazily and
  cache). Keep `effective_stack` as-is.
- **`_commit_tax` (line 1050) → `_future_edge_tax`:** swap the call site in
  `_postflop_vs_bet` (line 933). New formula gated by D1's five conditions and
  capped at survival 4.0pp / aggro 1.2pp; zero HU; zero `hero_bb ≤ 10`;
  near-zero 3-handed (see table D1 below). `_COMMIT_TAX` table is removed/retired.
- **Chip-leader pressure (preflop):** in `_deep_stack_action` / `_medium_stack_action`
  RFI/steal/3bet/iso branches, widen the active range via `_apply_range_nudges`
  using `_CHIPLEADER_PREFLOP` × `pressure_weight`. Range widening = **union with
  a deterministic expansion hand-set, admitted via `_freq_gate(view, salt,
  pressure_weight)`** so partial weights add hands deterministically (keeps the
  current frozenset-membership design; no global hand-ranking needed).
- **Chip-leader pressure (postflop):** in `_postflop_open` (line 969) multiply
  `CBET_FREQ` / `BLUFF_FREQ` / `SEMI_FREQ` by `_CHIPLEADER_POSTFLOP_MULT` via
  `_apply_freq_nudges`. **Do NOT lower value thresholds for chip-leader
  pressure** (rank does not make worse hands get called more). Pressure shows up
  as *more* bets/bluffs, not *thinner value*.
- **Short-stack desperation (preflop):** in `_short_stack_action` (line 638),
  widen the active first-in shove range via `_SHORT_DESPERATION` (gated by
  `hero_bb`, capped by `_SHOVE_CEILINGS`), and apply `_urgency_bonus` to
  proactive open-shoves / resteal-jams **only** (never to all-in *calls* — those
  stay pure range/pot-odds). Existing ~12bb shove/fold trigger is unchanged; add
  a mild 12–16bb transition nudge + `future_edge_tax *= 0.25` in that band.
- **Heads-up aggression (postflop):** when `players_left == 2`, set
  `future_edge_tax = 0`, then in `_postflop_open` / `_postflop_vs_bet` apply
  `_HU_POSTFLOP` multipliers (cbet/bluff/semi) and cushion/value-threshold
  reductions, scaled by `heads_up_weight`. HU is the one place value thresholds
  **are** lowered (wider ranges ⇒ thinner value is structurally correct).
  Preflop HU already uses `heads_up` range keys (Phase 2) — keep those.

---

## 4. Proposed default magnitudes (tuning knobs)

### Table A1 — Chip-leader preflop widening (max, at full pressure_weight=1, bucket=chip_leader)
`top_2` = ×0.55 of these. Format `(relative %, absolute cap pp)`.
| Spot | survival | aggro |
|---|---|---|
| HJ open | +3% / +1pp | +5% / +2pp |
| CO open/steal | +8% / +3pp | +14% / +5pp |
| BTN steal | +14% / +5pp | +22% / +8pp |
| SB steal | +16% / +6pp | +25% / +9pp |
| Iso vs limper (hero covers) | +10% / +4pp | +16% / +6pp |
| 3bet vs normal open | +8% / +2pp | +14% / +4pp |
| 3bet vs CO/BTN/SB steal | +14% / +4pp | +24% / +7pp |
| Resteal jam (eff medium/short) | +12% / +4pp | +20% / +7pp |

### Table A2 — Chip-leader postflop multipliers (full pressure)
survival: CBET ×1.08/1.05/1.00 (flop/turn/river); BLUFF ×1.12/1.08/1.04; SEMI ×1.10/1.08 (flop/turn).
aggro: CBET ×1.15/1.10/1.03; BLUFF ×1.25/1.18/1.10; SEMI ×1.18/1.14.
(3way ≈ half these deltas; 4plus ≈ near 1.0 — see interview transcript for the full per-way grid.)

### Table B — Short-stack desperation shove widening (by hero_bb) `[see §12: R4 shrinks these for the no-ante structure; R5 makes preflop desperation range-widening only]`
| hero_bb | survival | aggro |
|---|---|---|
| 10–12 | +5% / +2pp | +8% / +3pp |
| 8–10 | +10% / +4pp | +16% / +6pp |
| 6–8 | +18% / +7pp | +28% / +10pp |
| 4–6 | +30% / +12pp | +45% / +16pp |
| ≤4 | +45% / +18pp | +65% / +25pp |

Shove ceilings (max first-in shove width): survival HJ35/CO50/BTN75/SB90;
aggro HJ45/CO60/BTN90/SB100.

**Urgency bonus** (equity-pp reduction, proactive shoves only): 0 at 12bb →
linear → max 3.5pp (survival) / 5.0pp (aggro) at ≤4bb. Resteal/iso jams get
×0.75 of it. All-in *calls* get 0.

### Table C — Heads-up postflop (full heads_up_weight)
survival: CBET ×1.12/1.08/1.03; BLUFF ×1.18/1.12/1.08; SEMI ×1.15/1.12;
CALL_CUSHION −1.5/−2.0/−2.5pp; VALUE_BET −1.0pp.
aggro: CBET ×1.20/1.15/1.08; BLUFF ×1.30/1.22/1.15; SEMI ×1.25/1.18;
CALL_CUSHION −2.5/−3.0/−3.5pp; VALUE_BET −1.5/−1.5/−2.0pp.

### Table D1 — Future-edge tax (replaces `_commit_tax`), max equity-pp by stage `[see §12: R1 lowers the survival cap to ≈2.0pp; R2 caps the all-in-call tax ≤1pp; R3 keeps a small HU premium]`
| Players | survival | aggro |
|---|---|---|
| 6 (early/deep) | 3.0–4.0pp | 0.0–1.2pp |
| 5 | 2.0–3.0pp | 0.0–0.8pp |
| 4 | 0.5–1.5pp | ~0 |
| 3 | 0.0–0.75pp | 0 |
| HU | 0 | 0 |
Multiplied down by depth (`(min_bb−12)/28`, 0 at ≤12bb), commit fraction
(0 below 25% risk, full at 75%), and bust severity. Hard 0 when `hero_bb ≤ 10`.

---

## 5. Precedence / application order (per decision)
1. Player-count mode (heads-up? 3+?).
2. Absolute stack band (`hero_bb`/`eff_bb` → short/medium/deep) → base strategy.
3. Global rank bucket (cached per hand).
4. `rank_weight` (gates rank effects only).
5. `pressure_weight = rank_weight × bucket_weight × cover_weight`.
6. Preflop: base range → at most one gear of range expansion.
7. Postflop: base freq → multiplicative nudges with one-gear caps.
8. Additive equity adjustments: `CALL_CUSHION` + `future_edge_tax` − `urgency_bonus` (+ HU cushion reduction).
9. Clamp everything; Phase 1 sanitizer has the final word.

**Override:** desperation (`hero_bb ≤ ~10–12`) zeroes `future_edge_tax` and
disables rank-pressure logic; heads-up zeroes `future_edge_tax`.

---

## 6. Combination rules (how nudges compose) + one-gear clamps
- **Preflop range widths:** additive deltas, then cap total. Survival total cap
  ≈ +25% rel / +8pp abs (3bet/resteal tighter: +25%/+5pp); aggro ≈ +35%/+12pp
  (3bet +40%/+8pp).
- **Postflop frequencies:** multiplicative, then cap. Survival ≤ ×1.25 and
  ≤ +15pp absolute; aggro ≤ ×1.40 and ≤ +22pp. Knob sanity ceilings:
  CBET ≤0.92/0.96, BLUFF ≤0.45/0.55, SEMI ≤0.70/0.80 (survival/aggro).
- **Equity thresholds / cushions:** additive pp, then clamp to [base−maxReduction, base+maxIncrease].
- **Future-edge tax / urgency:** additive pp, then clamp.
- **Sizing:** essentially unchanged (all-in when short; optional −0.05 pot HU
  high-freq flop cbet). Chip-leader pressure = *more frequent*, not *bigger*.

---

## 7. Config (`ProfileConfig`, line 403) — reuse existing fields where possible
`SURVIVAL_CONFIG` / `AGGRO_CONFIG` already carry `survival_pressure`,
`max_risk_fraction`, `short_stack_bb_thresholds`, `medium_band_top_bb`, and an
unused `bubble_factor`. Plan:
- Replace the *meaning* of the commit-tax overlay with the future-edge tax;
  expose its per-stage caps as config (e.g. `future_edge_tax_caps`).
- Add `chipleader_pressure_scale`, `desperation_scale`, `hu_aggression_scale`
  (defaults 1.0 survival; aggro tuned up) so both profiles share one code path
  with different intensities. Retire/repurpose `bubble_factor`.

---

## 8. Testing — extend `sanity_tournament_hybrid.py` (new CHECK 15–19)
Follow the existing `_view` / `_table_overrides` / `_postflop_view` helpers.
- **CHECK 15 — stack-rank classification:** unit-test `_rank_context` over crafted
  `stacks` dicts: dominant leader → `chip_leader`; flat table (1020…980) → all
  `top_2`/`middle`, **no** forced leader/short; bottom-of-field but 40bb →
  `short_stack` bucket yet **not** desperate behavior; tie tolerance.
- **CHECK 16 — desperation ≠ rank:** 6bb 2nd-in-chips uses shove/fold + widened
  range + urgency, `future_edge_tax==0`; 40bb last-in-chips uses normal
  deep/medium ranges (no panic jam). Assert via `last_decision` branch/trace.
- **CHECK 17 — chip-leader pressure:** deep `chip_leader` covering field opens/
  steals/cbets/bluffs **strictly more often** than the Phase 3 baseline over a
  seeded sample; aggro > survival; value-bet thresholds **unchanged**; never
  exceeds the one-gear caps.
- **CHECK 18 — heads-up loosening:** `players_left==2` → `future_edge_tax==0`,
  cbet/bluff up, cushion down vs 6-handed baseline, for both profiles.
- **CHECK 19 — invariants preserved:** rerun a determinism check (same spot →
  same action) and a legality fuzz with rank context populated; assert no
  illegal actions and `path ∈ {preflop, postflop, passive, fallback}`. Confirm
  `reset_memory()` clears `_rank_cache`.
- **Regression:** CHECK 1–14 must still pass unchanged.

Run: `.venv/bin/python sanity_tournament_hybrid.py`

## 9. Evaluation (per `CLASS_TOURNAMENT_BOT_PLAN.md`)
Locked params: 1000 chips, 5/10 blinds, +1.5× every 50 hands, ante off,
6-player WTA. Use `eval_final_bot.py` (and `run_eval.py --mode multiway` where
CFR/Deep-CFR seats are available).
- Compare **Phase 4 vs Phase 3** survival and aggro on: win rate (+Wilson CI),
  avg finish, **early-bust rate**, heads-up-reached rate, HU conversion, all-in
  equity-when-called, decision time in heavy spots.
- **Go criteria:** Phase 4 survival's early-bust rate ↓ or win rate ↑ with no
  regression; aggro's chip-leader pressure does not raise its bust rate beyond
  the master plan's tolerance. Tune the Section 4 constants from these results.
- **A/B telemetry is mandatory — see §12 R7.** Tournament win rate (not per-hand
  chip delta) is the metric; the future-edge tax must beat a pure-chipEV arm
  before its cap is trusted.

---

## 10. Implementation order
1. `_rank_context` + caching + `reset_memory` clear + CHECK 15. *(no behavior change yet)*
2. Weight helpers (`_rank_weight`, `_pressure_weight`, `_cover_weight`, `_short_weight`, `_heads_up_weight`, `_urgency_bonus`) + one-gear combinators (unit-tested in isolation).
3. `_future_edge_tax` replacing `_commit_tax` (retire `_COMMIT_TAX`) → re-run CHECK 14.
4. Short-stack desperation (preflop) + CHECK 16.
5. Chip-leader pressure (preflop + postflop) + CHECK 17.
6. Heads-up aggression (postflop) + CHECK 18.
7. CHECK 19 + full regression (CHECK 1–14).
8. Eval pass (Section 9); tune constants.

Each step lands behind passing tests before the next; engine treated as frozen.

---

## 11. Open items for review
- **O1 — `committed_this_hand`** absent from `PlayerView`; plan caches rank per
  hand to mask blind jitter. Accept, or reconstruct from `history`?
- **O2 — Profile scope of chip-leader pressure:** master plan says aggro
  pressures as chip leader; survival gets a *small* version (×~0.4 intensity) so
  it still applies fold-equity pressure when it covers the table. Confirm
  survival should pressure at all, or stay neutral.
- **O3 — Preflop early-bust avoidance** stays **range-encoded** (the existing
  conservative `_CALL_SHOVE_RANGES` already decline thin all-in calls); the
  equity-based future-edge tax is a **postflop-only** overlay. Confirm we do NOT
  add preflop equity computation in Phase 4.
- **O4 — Constants** in Section 4 are first-draft defaults; final values come
  from the Section 9 eval, not from this document.

---

## 12. Second-opinion revisions (adversarial review — AUTHORITATIVE)

Source: an independent adversarial review in a **fresh** ChatGPT Pro chat (no
prior context), 2026-06-19. Verdict: *"right to kill ladder ICM, but too eager
to replace it with a generic survival tax."* These revisions move the design
**chipEV-first** and push exploit-heavy machinery to Phase 5. Where §12 conflicts
with §1–§11, **§12 wins.**

**R1 — chipEV-first default; lower + A/B-gate the future-edge tax.**
- The default decision rule is **pure chipEV**. The future-edge tax is a *small*
  premium applied only to close, high-variance, high-stack-risk spots, and it
  **must prove itself** in A/B eval (R7) before its cap is trusted.
- Lower the survival cap from 3.5–4.0pp to **≈2.0pp** (aggro ≈0.5–0.8pp) for v1.
  The burden of proof is on every *fold* of a +chipEV stack-off, not every call.
- Keep scaling by stack-fraction (`commit_factor`/`bust_factor`): a flat
  equity-pp premium is incoherent across pot sizes (3.5pp of a 200bb pot ≈ 7bb;
  of a 20bb pot ≈ 0.7bb). The premium must scale with chips actually at risk.
- The principled form is a state-value premium
  `p_required = (V(fold) − V(lose)) / (V(win) − V(lose))`, where the tax can be
  positive, zero, **or negative**. A real state-value estimator is **deferred
  (Phase 6+)**; v1 keeps the bounded heuristic but treats it as A/B-tunable.

**R2 — biggest-leak guard: do NOT fold +chipEV all-in CALLS vs likely punters.**
The #1 EV risk is passing 52–55% all-in *calls* against bots that overjam — in
WTA there is no prize for letting bad bots punt chips to someone else. In Phase 4
(pre-opponent-model) the tax on **all-in calls** is capped very low (**≤1.0pp**
survival) and is **removed entirely in Phase 5** for any opponent flagged
loose/spewy.

**R3 — decouple heads-up (resolves the inter-opinion conflict).**
HU still **loosens aggression** (steals/cbets/bluffs, optional smaller flop cbet)
per Table C. But do **not** blanket-zero the stack-off premium: against a weak HU
opponent, declining a ~51% flip *for the title* to keep grinding a postflop edge
can be correct. Revised rule:
- Keep a **small** stack-off premium HU (**≈0.5–1.0pp**) while `eff_bb` is deep
  enough to realize postflop edge.
- The Table C `CALL_CUSHION` reductions apply to **non-all-in streets**, not to
  coinflip-for-the-title all-ins.
- Premium → 0 only when HU stacks are shallow (skill can't be realized).

**R4 — no-ante urgency (shrink Table B; add blind-clock / M).**
No ante ⇒ orbit cost ≈1.5bb, so waiting is cheap and shove ranges should stay
**tighter** than ante-MTT charts. For v1: roughly **halve** the ≤8bb tiers of
Table B and cut the urgency magnitudes. Add **blind-clock awareness** — track
hands-until-next-level (schedule +1.5× every 50 hands; derive from `hand_id`) and
use **M = stack / (SB + BB)** alongside bb. 15bb two hands before a jump ≠ 15bb
with 45 hands left.

**R5 — fix the urgency *mechanism*.**
Preflop is **range-membership based**, so desperation = **range widening only**
(Table B). Drop the "urgency_bonus as required-equity reduction" framing for
preflop (no equity is computed there). Reserve any equity/EV-discount form for
spots where equity is actually computed.

**R6 — rank is a weak WTA signal; fix the gate.**
Keep the 4 buckets only as a coarse layer. Drive actual pressure from **chip
share, per-opponent effective stack, and cover ratio**, computed
per-opponent/per-action — not a single global `min(hero_bb, eff_bb)` gate, which
wrongly suppresses pressure against *short* stacks that overfold. Recompute
pressure against the specific villain(s) contesting the pot.

**R7 — empirical A/B telemetry is mandatory (extends §9).**
Log per decision: spot type (all-in call / shove / steal / 3bet / cbet / bluff),
the pure-chipEV action, the Phase-4 action, tax applied, equity threshold
before/after, stack state, players left. A/B arms: pure chipEV; tax cap ∈
{1, 2, 3.5}pp; HU tax 0 vs small; rank-pressure on/off. **Metric = tournament
win rate over a large sample** (per-hand chip deltas are too noisy in WTA).

**R8 — scope reconciliation with later phases.**
- The review's "biggest missing piece" — **opponent modeling** (fold-to-steal,
  call-all-in freq, fold-to-cbet, etc.) — **is Phase 5** by design. Phase 4
  therefore keeps chip-leader pressure and multiplied bluff frequency **small and
  one-gear-capped**: they profit only against overfolders and backfire vs calling
  stations. Phase 5 reads turn these from blind nudges into measured exploits
  (and may *raise* the one-gear cap for confirmed overfolders).
- Multiway / overcall / squeeze-risk on preflop all-in calls, and a full
  dedicated heads-up module, are noted as **refinements** layered with Phase 5.
- Framing: "survival" = **edge-preservation**, not chip-preservation (2nd pays
  zero). No rename of the `final_survival` alias, but bucket meanings and the tax
  are written to that principle.

### Net effect on the implementation order (§10)
Unchanged sequence, but: step 3 (`_future_edge_tax`) ships at the **R1/R2/R3
caps** with the **R7 telemetry hooks** built in from the start; step 4
(desperation) uses the **R4-shrunk** Table B as **range widening only** (R5);
steps 5–6 stay deliberately small per **R8** pending Phase 5.
