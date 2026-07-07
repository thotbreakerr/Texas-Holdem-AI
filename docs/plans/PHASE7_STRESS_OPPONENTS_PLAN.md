# Phase 7 — Stress Opponents (Implementation Plan)

> Status: **PLAN ONLY — design complete, not implemented.** Engine frozen.
> Bots to build live in `bots/` and register through `create_bot` (`bots/__init__.py`).
> They are **opponent** bots; they do **not** touch `TournamentHybridBot`'s Phase-4/Phase-5
> ablation arms.
>
> **Provenance.** Designed in three passes: (1) a code-grounded mapping workflow over the
> engine, the existing bots, the eval harness, and the Phase-5 read machinery; (2) a
> `grill-me` design interview conducted **against ChatGPT in the browser** (4 rounds,
> branches A–J + 15 open questions), the same methodology used for Phase 5; (3) direct
> code verification of the two load-bearing claims the interview surfaced. Every numeric
> constant below is a **proposed default subject to eval tuning** — the structure is the
> commitment. Key interview catches and their resolutions are in **Appendix A**.

---

## 0. Objective and scope

### 0.1 The problem (from eval, code-confirmed)
The hybrid bot's Phase-5 layer ships two **active** exploits — station-suppression and R2
(stop folding +chipEV all-in CALLS to confirmed-spewy jammers) — but in eval against the
current pool (`smart, mc200, gto, icm, exploitative`) **`jam_like_count = 0` and
`large_bet_count = 0` for every opponent, so R2 never fires and the station read is weak**.
Root cause (verified): every pool bot caps postflop sizing at ~25–30% of stack
(`bots/exploitative_bot.py:457,468,477`), so no opponent ever enters `all_in_opponents`
off a postflop bet, and the jam/large-bet classifiers (`tournament_hybrid_bot.py:894–909`)
stay pinned at zero. Phase 5 is therefore **untested**, not proven good or bad.

### 0.2 What Phase 7 is — and is not
Phase 7 builds a deliberate archetype opponent field so those reads fire and Phase 5 can be
A/B-validated. **v1 is a STRESS HARNESS, not a plausible proxy for the real class field.**
It answers: *"When the world contains the observable mistakes Phase 5 claims to exploit,
does the bot detect and exploit them without breaking baseline play?"* It does **not**
answer: *"Will Phase 5 improve performance against the real class field?"* — that is the
non-gating transfer question (Tier 3).

### 0.3 The 3-tier scope (gating order)
| Tier | Field | Role | Gating? |
|---|---|---|---|
| **T1** | targeted single-read cells (hero + the target archetype + fillers, seat-controlled) | **trigger harness** — prove each read fires, exploit direction correct | **YES** |
| **T2** | blended archetypes + a sane null | **diagnostic** — does P5 help without collateral damage when the bot can't farm one cartoon leak? | partial |
| **T3** | legacy pool ± tuned-down archetypes | **realism / transfer** — does the result carry to natural opponents? | **NO (v1)** |

### 0.4 Governing guardrail (the anti-caricature gate)
A Phase 7 **PASS is not "better tournament finish."** It requires evidence that
**(a)** the intended read *fired and was hero-observed*, **(b)** the exploit *changed a
decision*, and **(c)** the changed decision had **non-negative ex-ante EV support and no
realized harm signal**. Raw win-rate against caricatured archetypes is insufficient — a
cartoon maniac that makes R2 fire can produce a *misleading PASS from caricature
overfitting*. (Interview catch #1, Appendix A.)

---

## 1. Engine contract (LOCKED — the only legal surface)

A bot implements `act(view: PlayerView) -> Action`. Verified facts:

- **Read state** from `PlayerView` (`core/bot_api.py:10-29`, populated `core/engine.py:720-739`):
  `hole_cards, board, street, position, pot, to_call, stacks, history, hand_id`, and
  `legal_actions` (`engine.py:730`) — **the authoritative move list**; `bet`/`raise`
  entries carry `{"min","max"}`. Also `all_in_opponents` (`engine.py:715-718`),
  `acting_opponents` (`engine.py:711-714`), `seat_indices`. **There is no `session_id`.**
- **Do NOT** size from `min_raise`/`max_raise` (`engine.py:728-729`) — increment-style,
  partly stale. Size **only** from `a["min"]`/`a["max"]` on the legal `bet`/`raise` dict;
  always `int()` + clamp.
- **Return `Action(type, amount=None)`** (`core/bot_api.py:5-8`). `fold`/`check`/`call` →
  no amount; **never `fold` when `to_call == 0`** (use `check`). `bet`/`raise` `amount`
  is the **raise-TO total street contribution**, not an increment (`engine.py:830-837`).
  **There is no `"all_in"` type** — a shove is a `bet`/`raise` to the full stack.
- **History entry** (`engine.py:804-811`): `{street, pid, type∈{fold,check,call,bet,raise},
  amount, to_call_before, pot_before}`; for `bet`/`raise`, `amount` is raise-TO total; for
  `call`, chips actually paid. Phase 5 reconstructs `incremental = amount − prior_street_contrib`.
- **Lifecycle:** no callbacks. Stateless archetypes need no `reset_memory`. If a bot ever
  keeps cross-hand state, dedup on `(hand_id, pid, type, street, seq)` and **never on
  `id(entry)`** (that broke cross-process determinism here; guarded by sanity CHECK 35).

---

## 2. The censoring constraint (why this is genuinely hard)

The bot sees only the actions that occur **before its own turn**. Folds, calls, and all-in
calls that **end the hand to the hero's own bet are never observed.** Therefore every
"make a read fire" goal is really **"make a *hero-observable* read fire."** This reframes
the whole harness:

- Reads built on **folds** (nit `fold_to_pressure`) are the most censored — the canonical
  event (hero bets → nit folds → hand ends) is invisible. Such reads fire reliably only
  via **third-party pressure with the hero still live** (another bot bets, nit folds, hero
  acts after).
- Reads built on **observed calls** (`station_score`) are less censored but still miss the
  final river call that closes the action.
- **Every read's telemetry must split `happened-in-world` from `hero-observed-before-a-decision`.**
  Otherwise a censored read looks identical to a broken archetype spec.

---

## 3. Determinism design (LOCKED)

Archetype randomness must be **byte-reproducible across processes** (the project has been
burned by non-determinism before — sanity CHECK 35). Rules:

- **Stateless hash-to-uniform Bernoulli.** Each stochastic choice draws a uniform from a
  stable hash of the key, never from a mutable PRNG advanced inside `act()`.
- **Seed key (every field comes from `PlayerView` — the bot has no other channel):**
  `(hand_id, hole_cards, board, street, pot, to_call, pid, actor_decision_count_this_hand, decision_kind)`
  where `decision_kind ∈ {"preflop_open","postflop_jam","station_call","minraise_raise",
  "overbet","nit_fold",...}`. Hash with a stable hash (mirror `_stable_hash`,
  `tournament_hybrid_bot.py:3089-3104`); **never** Python builtin `hash()`/`id()`.
- **No `global_eval_seed` — there is no way to inject one (corrects an earlier interview
  conclusion).** `PlayerView` has no seed field (`core/bot_api.py:10-29`) and
  `create_bot(btype)` takes only a type string (`bots/__init__.py:17`); the eval never hands
  a bot per-tournament state (`eval_final_bot.py:149`). Per-tournament variation is **already**
  guaranteed by the fresh per-tournament deck RNG (`eval_final_bot.py:166`): different cards ⇒
  different `hole_cards`/`board`/`pot`/`to_call` ⇒ a spot-keyed hash differs every tournament.
  The "N trials collapse to one script" worry does **not** apply here. (True seed isolation
  would require an engine change to pass a seed into bots — explicitly out-of-scope.)
- `actor_decision_count_this_hand` = this bot's *n*th decision this hand (**not** an index
  into `legal_actions`, whose ordering is engine-owned and could change).

---

## 4. Architecture (LOCKED)

**One `ArchetypeBot` wrapper + separate policy classes — not a flat knob table.**
Archetypes differ *structurally*, not just in frequency; a flat table pollutes the reads
(a "station" that accidentally carries maniac aggression, a "maniac" that folds too much).

```
ArchetypeBot                       # bots/archetype_bot.py
  - shared observation helpers (stack/pot/bb, effective-stack, players-live)
  - shared legality/sizing helpers (parse legal_actions, min/max, int+clamp, shove-to-max)
  - shared seeded hash-to-uniform Bernoulli (Section 3)
  - safe passive fallback (check>call>fold, never illegal)
  - preset config (frozen knob dict) selects ONE policy:
      NitPolicy  ManiacPolicy  StationPolicy  LoosePassivePolicy
      MinRaisePolicy  OverbetPolicy  BaselineSanePolicy  PressureFillerPolicy
```

The frozen knob table **configures** the policies; it does not replace them.

**Registration** (`bots/__init__.py`, the `create_bot` if/elif chain, single source of
truth — `parse_players` and both eval harnesses funnel through it): add one branch per
alias before the final `raise`, and update the docstring + the `raise ValueError`
enumerated list. Use **`calling_station`**, not `station` (collides with the Phase-5
`p4_station_only`/`station_r2` arm tokens). Suggested aliases: `nit`/`folder`,
`maniac`/`maniac_trigger`/`maniac_mixed`, `calling_station`, `loose_passive`,
`minraise`/`minraiser`, `overbet_merchant`, `baseline_sane`, `pressure_filler`.

**Pool wiring:** these go into an **opt-in named `--stress-pool`**, **not** `DEFAULT_POOL`
(`eval_final_bot.py:212`, `run_eval.py:30`) — changing the default silently moves every
historical win-rate baseline.

---

## 5. The roster and per-archetype specs

Eight presets + one infra bot. Each is a **public-state frequency script** for v1 (it
conditions only on cheap public state — effective stack, players-live, facing-bet — and
**never** on hole-card equity; so call it a "frequency stressor," not a "range"). All
numeric values are eval-tunable defaults. Stat machinery references:
`tournament_hybrid_bot.py:870-942` (classify), `894-909` (jam/large-bet thresholds),
`1424-1426` (station read), `1491-1518` (spewy gate), `1530-1556` (R2 fire), `2270`
(all-in-call definition).

### 5.1 `maniac_trigger` / `maniac_mixed` → `jam_like_count` ⇒ **R2** (headline)
- **Behavior:** loose preflop; postflop, in **eligible jam spots**, shove to `a["max"]`
  (`bet`/`raise` total = stack) so the engine flags it all-in and it appears in a later
  `all_in_opponents` view.
- **Eligibility (the denominator that keeps it from being a cartoon):** postflop **only**;
  effective stack **12–35bb**; shove legal to max; **incremental ≥ 12bb** (so it counts as
  `jam_like`, not `short_jam_like` — a `<12bb` shove disqualifies R2 via
  `short_jam_like_count == 0`); prefer **heads-up / 3-way**; wide-but-not-any-two range.
- **Frequency:** `maniac_trigger` jams **65%** of *eligible* spots (T1);
  `maniac_mixed` **40–45%** (T2). **Never 100%; avoid 80%** except debug.
- **Anti-bust / observability (critical):** below ~**8–10bb** it reverts to normal
  short-stack shove/fold (no longer the intended postflop jammer). Run **multiple maniacs**
  in some T1 cells and seat them in the hero's orbit, or the spew is observed by the wrong
  opponents / after the hero folded. Log **missed-jam-opportunity** events.
- **R2 path:** must also clear the looseness conjunction (Section 5.9) and accumulate
  `jam_opportunity_n ≥ 3` with `jam_like_count ≥ 2` *while hero observes it.*

### 5.2 `overbet_merchant` → `large_bet_count ≥ 3` ⇒ **R2 (2nd OR-branch)** — keep SEPARATE
- **Behavior:** postflop **non-all-in** bets/raises sized to
  `incremental ≥ max(8bb, 0.75·pot_before)` while **keeping ≥2–4bb behind** so it is **not**
  reclassified as a jam (the `elif` precedence at `_classify_entry` makes jam win).
- **Eligibility:** postflop; eff stack **18–60bb**; target **0.8–1.25 pot**, capped to
  leave chips behind; **55–70%** of eligible spots (T1).
- **Why separate (not folded into the maniac):** if combined, any R2 improvement attributes
  to the *jam* branch and the large-bet branch is never isolated. It also catches
  all-in-vs-non-all-in misclassification, raise-TO-vs-incremental confusion, and the
  max-clamp accidentally converting an overbet into a shove.
- **Read-vs-exploit gap (subtle, important):** `large_bet_count` fires the **read**, but R2
  only changes a decision at an **all-in CALL**. So the overbet merchant must **create a
  rare late all-in (or a bad short call-off) AFTER `large_bet_count ≥ 3`**, else it fires
  the read but never exercises the exploit. (Interview catch #6.)

### 5.3 `calling_station` → `station_score ≥ 0.60`
- **Behavior:** loose entry (VPIP 45–60%, PFR 0–5%); facing non-all-in postflop pressure
  **call 75–90% / raise 0–2% / fold the rest**, across multiple streets; facing an all-in,
  call lower (**35–55%**) so it survives long enough to be sampled. Do **not** rely on river
  calls for samples (final river call is often unobserved).
- **Note the defect it exposes:** see Section 6.

### 5.4 `nit` / `folder` → `fold_to_pressure` (telemetry only; **not a hard T1 gate**)
- **Behavior:** premium-only VPIP; facing a bet/raise (preflop after a raise, or any
  postflop) **fold the large majority**.
- **Highest censoring risk:** its canonical fold ends the hand unseen. It produces
  hero-observable folds **only via third-party pressure with the hero still live** — which
  is what `pressure_filler` (5.8) and seat geometry (Section 7) are for. Because of this,
  `fold_to_pressure` is a **telemetry archetype**, not a Phase-5-benefit gate.

### 5.5 `loose_passive` → R2-**looseness negative control ONLY**
- **Behavior:** wide entry via **limp/call, not raise** (open-limp 45–55%, raise-first-in
  3–8%, call preflop raises 30–45% size-capped); postflop check 80–95% when checked to,
  facing a bet call 45–65% / fold 35–55% / raise ~0%; `large_bet`/`jam` = 0.
- **Claim narrowly:** it proves **high VPIP alone does not satisfy the R2 spewy pathway**
  (it fails the PFR and postflop-aggression gates and produces no jam/large-bet evidence).
  It is **not** a clean station negative control — under the current `station_score` it can
  read ≈1.0 (Section 6).

### 5.6 `minraise` → PFR-without-large-bet control + read-classification regression guard
- **Behavior:** when aggressive, raise to **exactly `a["min"]`** (legal min raise-TO);
  VPIP 35–50%, PFR 20–35%; `jam_like`/`large_bet` must stay **0**. Stop min-raise testing
  below ~8–10bb (a min-raise can become functionally all-in).
- **Value:** confirms PFR fires without big sizing, exercises the min raise-TO sizing path,
  and regression-guards the incremental-vs-raise-TO classification (verified correct today,
  Section 6).

### 5.7 `baseline_sane` → T2 null control (no exploit should fire)
A plain, capped-bet, roughly-sane bot. A mixed field of **only** stress archetypes is too
artificial; T2 needs at least one opponent that should trigger **no** exploit, to detect
**false exploit activation / collateral damage.**

### 5.8 `pressure_filler` → infra bot (not tied to a read)
A bot that **reliably bets** into spots so that a downstream nit/station fold or call
becomes **hero-observable** (`pressure_filler` acts → archetype responds → hero acts later,
still live). Without it, nit/station reads are dominated by censoring.

### 5.9 The looseness conjunction (shared R2 gate for maniac/overbet)
`_p5_confirmed_spewy` (`tournament_hybrid_bot.py:1493-1496`) is a 3-way AND on shrunk
ratios: `vpip_hat ≥ 0.36` AND `preflop_aggression_rate_hat ≥ 0.18` AND
`postflop_aggression_freq_hat ≥ 0.38`. With shrink weight `w ≈ 10`, the jammer needs real
hand **volume** to clear the priors before it busts — another reason for anti-bust guards
and multiple maniacs.

---

## 6. Confirmed code findings (verified during the interview)

1. **`large_bet`/`jam` sizing is incremental — SAFE.** `tournament_hybrid_bot.py:885`
   computes `incremental = self._incremental_amount(...)`; both the jam `risk_bb` (898) and
   the `large_bet` threshold (908) use *incremental*, not the raw raise-TO total. So a
   min-raise is **not** misclassified as a large bet. `minraise` stays as a regression guard.

2. **`station_score` denominator excludes folds — CONFIRMED Phase-5 limitation.**
   `station_score_hat = p_hat(postflop_pressure_call, station_n, …)` where
   `station_n = postflop_pressure_call + postflop_pressure_raise`
   (`tournament_hybrid_bot.py:690-693, 731-735`), and the `pressure_response_n ≥ 4` guard
   (`704`) uses that **same** fold-excluding `station_n`. Consequence: a `loose_passive`
   (high postflop call, ~0 raise) — and an occasionally-calling nit — that accumulates ≥4
   observed postflop pressure-*calls* scores `station_score ≈ 1.0` and fires
   **station-suppression** despite folding plenty.
   - **The fix is small (corrects an earlier note).** A `postflop_pressure_fold` counter
     **already exists** — declared `:607`, incremented `:838` — it is simply not in
     `station_n`. Adding `+ counters["postflop_pressure_fold"]` to `station_n` (`:690-693`)
     turns the ratio into `call/(call+raise+fold)`; alternatively add a secondary
     `fold_to_pressure_hat ≤ max` guard to the station read.
   - **Phase 7's job:** *expose* this empirically — measure the **station false-positive
     rate** vs `loose_passive`/`nit` against the true `calling_station`. The actual fix is a
     **Phase-5 follow-up, not Phase-7 scope** (O1).

---

## 7. Seat geometry and field compositions

Censoring makes **seat order load-bearing** — "2× target + fillers" alone is not enough.

- **T1 trigger cells** — cover these geometries (hero = `H`, target = `T`, filler = `F`):
  `[T, H, T, F, F, F]`, `[F, T, H, T, F, F]`, `[T, F, F, T, H, F]` — i.e.
  **target-before-hero**, **target-after-hero**, and **bracketing** layouts, plus a
  **`pressure_filler`-before-`nit`-before-`H`** layout for fold observability. Rotate seats:
  **6 rotations × 25 seeds = 150 tournaments per T1 cell minimum**; go to **300** for the
  R2 maniac/overbet cells if observed *changed* all-in decisions are `< 30`.
- **T2 mixed:** one each of `maniac_mixed, calling_station, loose_passive, minraise,
  overbet_merchant` + `baseline_sane`.
- **T3 realism (non-gating):** legacy pool ± tuned-down archetypes (transfer risk only).

---

## 8. Telemetry and metrics

For **every** read, log both `happened-in-world` and `hero-observed-before-a-decision`,
plus the censoring reasons:
```
true_<read>_count
hero_observed_<read>_count
missed_due_to_hand_end_count
missed_due_to_hero_already_folded_count
```
Plus:
- **missed-jam-opportunity** log: eligible jam spot existed, draw did/didn't jam, hero
  did/didn't observe.
- **R2 fire count** + **tax actually relieved** per tournament (must move 0 → >0).
- **`short_jam_like` contamination rate** (fraction of maniac shoves routed to short-jam,
  which zeroes R2 — if high, raise the maniac jam-stack floor).
- **station false-positive rate** vs `loose_passive`/`nit` (Section 6).
- **value layers** for changed decisions (Section 10).
- **sample sufficiency:** `pressure_response_n`, `jam_opportunity_n`, `preflop_action_seen`
  reached per archetype per tournament before elimination.

---

## 9. Eval integration

These are **opponent** bots; they do **not** touch the hybrid's P4/P5 arms
(`P5_ARM_SPECS`, `_configure_final_arm`). The experiment is: run the **same hybrid arm**
(e.g. `final_aggro:p5`, R2 on) against **(i) the legacy pool** and **(ii) the stress pool**,
and diff read-fire telemetry + win-rate.

```
# T1 trigger cell, no model files needed:
.venv/bin/python eval_final_bot.py --profile aggro --arm p5 --tournaments 150 \
  --stress-pool maniac_trigger,maniac_trigger,pressure_filler,calling_station,nit
# (introduce --stress-pool; do NOT alter --pool DEFAULT_POOL)
```

If curriculum Tier-3's hardcoded 7-bot field (`run_eval.py:566-580`, `1/7` break-even) is
ever extended with archetypes, re-derive the break-even for the new field size.

---

## 10. Acceptance criteria (event-count based, not tournament-count)

A Phase 7 **PASS** requires (win-rate alone is insufficient):
1. **Each target read fires vs its archetype**, both *in-world* AND *hero-observed* above a
   floor (`maniac → jam_like`, `overbet_merchant → large_bet`, `calling_station → station`,
   `minraise → pfr-only`, `loose_passive → vpip-only-not-spewy`, `nit → fold_to_pressure`
   telemetry).
2. **R2 fires `> 0`** (was 0) with the **tax actually relieved**.
3. **Station false-positive rate** vs `loose_passive`/`nit` is **measured and reported**
   (Section 6 finding).
4. **Value gate (honest wording under the no-showdown constraint):** *"the exploit changed
   decisions that have **non-negative ex-ante EV support** (the hybrid's own MC-equity
   estimate at decision time) and **no realized harm signal** (chip-delta on changed
   decisions is not net-negative)."* True counterfactual EV is **not measurable** without a
   showdown hook — do **not** claim "+value proven."
5. **`baseline_sane` triggers no exploit** (no false activation / collateral damage).

The bar is **event counts** — enough hero-observed events, threshold crossings, seat-bucket
coverage, and changed decisions — **not** tournament count alone.

---

## 11. Implementation order

1. `ArchetypeBot` wrapper + shared helpers + seeded hash-to-uniform Bernoulli (Section 3) +
   `create_bot` registration; default behavior = safe passive. **No read-firing yet.**
2. Policies one at a time. **Each lands with a deterministic unit test** that hand-builds a
   `PlayerView` history sequence and asserts `_classify_entry` sets the target read
   (`jam_like`/`large_bet`/`station`/`fold`) — CI-able, independent of stochastic play.
3. `pressure_filler` + the seat-geometry harness (Section 7).
4. Telemetry split (in-world / hero-observed + censoring reasons) wired into
   `eval_final_bot` (Section 8).
5. **T1 trigger cells**: confirm each read fires hero-observed; confirm **R2 > 0** with tax
   relieved; report station false-positive rate.
6. **T2 mixed**: value/no-harm gate; `baseline_sane` null check.
7. **(Separate, Phase-5 follow-up)** the `station_score` fold-aware fix (O1) — *not* Phase 7.

Each step lands behind passing tests before the next; the engine stays frozen.

---

## 12. Open items

- **O1 — `station_score` fix form** (add `postflop_pressure_fold` to the denominator vs a
  secondary `fold_to_pressure ≤ max` guard). **Phase-5 scope**, surfaced by Phase 7.
- **O2 — `pressure_filler` realism** — does an always-betting filler distort T2 enough to
  matter, or keep it T1-only?
- **O3 — class-field calibration (#9) / T3 archetype tuning** — deferred; T3 is non-gating
  in v1.
- **O4 — hero-observed-event floors per read** — set from T1 telemetry, not guessed.

---

## Appendix A — Design provenance and interview catches

**Process.** (1) Code-grounded mapping workflow → design brief (engine contract, bot
skeleton, the behavior→read mapping, eval integration). (2) `grill-me` interview vs ChatGPT
in the browser, 4 rounds. (3) Code verification of the two load-bearing claims (Section 6).

**Catches that changed the design:**
1. **Caricature-overfit → misleading PASS.** Drove the 3-tier scope + the value/no-harm
   gate (a cartoon maniac making R2 fire is not a validated exploit).
2. **Replay degeneracy.** No `session_id` ⇒ `global_eval_seed` is mandatory in the seed key,
   else "N trials" are one behavioral script run N×.
3. **Flat knob table blurs archetypes.** One wrapper + **separate policy classes**.
4. **Maniac eligibility + anti-bust.** Jam only in 12–35bb eligible spots at 65%/40–45%;
   revert <8–10bb; multiple maniacs + missed-opportunity telemetry so the hero observes the
   spew before the maniac busts.
5. **`station_score` excludes folds — confirmed defect.** `loose_passive`/`nit` can
   false-positive as stations; Phase 7 measures it, Phase 5 fixes it (O1).
6. **Overbet merchant fires the read but not the exploit** unless it later reaches an
   all-in-call decision — keep it separate and give it a late-all-in/short-call-off path.
7. **Censoring makes seat order load-bearing** — controlled seat geometry + `pressure_filler`
   for nit/station; split `in-world` vs `hero-observed` telemetry.
8. **Acceptance is event-count based**, with the honest "non-negative ex-ante EV support +
   no realized harm" wording instead of "+value proven."

---

## Appendix B — Handoff implementation prompt

> **Goal.** Implement Phase 7 stress-opponent archetype bots so the hybrid's Phase-5 reads
> (station-suppression, R2) fire and can be A/B-validated; today `jam_like=large_bet=0` and
> R2 never fires against the legacy pool.
>
> **Context.** Engine contract in §1 (legal_actions authoritative; `amount` = raise-TO
> total; shove = bet/raise to `a["max"]`; no showdown hook → action-frequency, censored
> reads). Phase-5 read machinery: `bots/tournament_hybrid_bot.py:870-942, 894-909,
> 1424-1426, 1491-1556`. Registration: `create_bot` if-chain in `bots/__init__.py`.
>
> **Constraints.** One `ArchetypeBot` wrapper + separate policy classes (§4); stateless
> hash-to-uniform Bernoulli seeded per §3 (never builtin `hash()`/`id()`, never a mutable
> PRNG in `act()`); opt-in `--stress-pool` (do **not** change `DEFAULT_POOL`); use
> `calling_station` not `station`; every policy ships with a classifier unit test (§11.2);
> telemetry splits in-world vs hero-observed (§8); engine frozen.
>
> **Done when.** §10 acceptance holds: each read fires hero-observed above its floor; R2
> fires `>0` with tax relieved; station false-positive rate vs `loose_passive`/`nit` is
> reported; the value/no-harm gate holds; `baseline_sane` fires nothing. The `station_score`
> fold-aware fix (O1) is a separate Phase-5 task, not part of this.
