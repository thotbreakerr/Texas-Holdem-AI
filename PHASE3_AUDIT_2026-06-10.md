# Phase 3 Audit — MCCFR / Path A Correctness (2026-06-10)

Scope: `bots/cfr_bot.py`, `core/action_history.py`, `training/train_cfr_bot_multiway.py`.
All five suspected issues were verified directly in code. No code changed yet — this is the audit + fix plan, pending approval.

---

## Issue 1 — External-sampling regret update double-counts opponent reach

**Verdict: REAL. Severity: CRITICAL (blocks retrain).**

Where: `bots/cfr_bot.py`, `_cfr_recurse` (lines 1791–1858).

- Opponent nodes sample exactly one action on-policy (lines 1818–1825). In external sampling, that sampling frequency *already* carries the opponent reach π₋ᵢ in expectation.
- The regret increment then multiplies by `cf_reach` = ∏ opponent reach **again** (computed 1846–1850, applied 1854). Expected update weight becomes ∝ π₋ᵢ², not π₋ᵢ.
- Effect: root-level updates are unbiased (cf_reach = 1 at the root, since `_run_iterations` line 1571 starts all reaches at 1.0), but every deeper hero node is biased toward high-probability opponent lines and geometrically down-weighted with depth. Textbook ES (Lanctot et al. 2009) uses no reach factor at all in the traverser's regret update.
- **Answer to "should `cf_reach` be removed from regret increments": yes — remove it entirely** (the alternative, importance-weighting by 1/q, is algebraically the same thing).

**Strategy-sum averaging is also not correct for this sampler.**

- Current code updates `strategy_sum` at *hero* nodes weighted by `self_reach` (line 1856). Under ES sampling, a hero node is visited with probability π₋ᵢ·π_c, so the expected increment is π₋ᵢ·π_c·πᵢ·σ(a) — vanilla CFR wants πᵢ·σ(a). Opponent reach leaks into the average strategy too.
- Textbook ES: update `strategy_sum` at **opponent** nodes, unweighted (`strategy_sum[a] += strategy[a]` for all legal a, before sampling); the opponent's own sampled actions supply the πⱼ weighting in expectation. Remove the hero-node `strategy_sum` update.
- This works in our training setup because `train_cfr_bot_multiway.py` (lines 143–160, 184) shares **one** CFRBot/table across all 6 seats: every seat's traversals deposit average-strategy mass at the infosets the other seats hit, and info-set keys are seat-agnostic.
- Trade-off to be aware of: in single-bot *online* learning against external opponents (e.g., `train_multi_deep_rl_bot.py` uses CFRBot as a learning opponent), opponent-node-only updates still accumulate (the tree's opponent seats use the same shared table), but coverage of hero-perspective infosets is weaker. Acceptable; flagged for the implementation step.

**Minimal fix (~6 lines):** drop `cf_reach` from line 1854; move the `strategy_sum` update from line 1856 into the opponent branch (after `get_strategy`, before sampling); `reach_probs` bookkeeping can then be deleted entirely (no remaining consumer), simplifying the signature.

---

## Issue 2 — "MCCFR+" claim vs. vanilla regret matching

**Verdict: REAL mismatch. Severity: MINOR (doc/claim issue; code is internally consistent).**

- `_CFRNode.get_strategy` (lines 1142–1163) is plain regret matching: `max(0, regret_sum)` at *read* time only. `regret_sum` itself is never floored after the update (line 1854) — negative regret accumulates, which CFR+ forbids (CFR+ stores Q_t = max(0, Q_{t-1} + Δ)).
- No iteration index anywhere in accumulation — `strategy_sum` (line 1856) is uniformly weighted, not linear/quadratic as CFR+ prescribes.
- Claims live in `TRAINING_PLAN.md:246` ("MCCFR+ … linear iteration weighting + positive-regret-only matching"), `TRAINING_PLAN.md:285`, and `SESSION_LOG_2026-04-26.md:142–146`. The `cfr_bot.py` module docstring honestly says "External-sampling MCCFR with regret-matching" — only the planning docs over-claim. (Path B *does* have the linear weighting: `deep_cfr_bot.py:1955–1961`.)

**Recommendation: update the docs to "vanilla external-sampling MCCFR", don't implement MCCFR+ now.** Rationale: regret flooring + linear averaging under *sampling* has weaker/shakier guarantees than exact CFR+ (known result; CFR+ was designed for the unsampled setting), and Phase 3's goal is a provably-correct base sampler before a long retrain. MCCFR+ can be added later behind a flag and A/B'd. If you prefer to implement instead, it's ~10 lines (floor `regret_sum` at 0 after each update; weight `strategy_sum` increments by an iteration counter).

---

## Issue 3 — Tree-training vs live-play history token mismatch

**Verdict: REAL. Severity: CRITICAL (blocks retrain).**

Where: `_GameState.apply_action` token emission (`cfr_bot.py:925–929`) vs canonical tokenizer (`core/action_history.py:122–176`).

- The tree appends a **fixed token per chosen bucket** (bet_33→"S", bet_50→"Q", … all_in→"A") regardless of the realized amount.
- Live play tokenizes by **realized ratio** `amount / pot_before` with thresholds 0.40 / 0.55 / 0.70 / 0.85 / 1.2 (`action_history.py:160–172`), where `amount` is the engine's *total street contribution* target.
- These coincide only for opening bets. Divergence cases, all confirmed in code:
  - **Raises**: target = `current_bet + frac·(pot + to_call)` (`_raise_sizing_target`, lines 485–502), so the realized ratio lands 1–2 buckets higher. Example: pot 100, current_bet 50, to_call 50, bet_33 → target ≈ 100 → ratio 1.0 → live "P", tree "S". Preflop raises almost always tokenize live as "P"/"A".
  - **Min-raise clamps**: `apply_action` clamps to `lo` (line 911); the chosen bucket token is kept even though the realized ratio moved.
  - **All-in clamps**: a sizing bucket clamped to `hi` keeps its sizing token in the tree, but tokenizes by ratio live (could be "A" or, for a short stack, "P"/"M"). Conversely the tree's all_in token "A" requires ratio ≥ 1.2 live — a short-stack shove tokenizes lower.
- Consequence (answers the orphaning question): yes — every tree node deeper than the traversal root is keyed with fixed-token suffixes. When the real hand reaches that same situation, `act()` rebuilds the key from canonical tokens of the realized history (line 1353), which differ whenever any intervening bet/raise was a raise or got clamped. The trained subtree mass never transfers to the next real decision's root; deep training work is largely wasted, and subgame-search opponent lookups hit partially phantom keys. Root-level nodes are fine (both paths tokenize the realized engine history canonically).

**Minimal fix — canonical tokenization shared by tree and live play:**

1. Add `sizing_token(amount_total: int, pot_before: int) -> str` to `core/action_history.py`, extracted from the existing threshold ladder in `tokenize()` (single source of truth; `tokenize()` calls it).
2. In `apply_action`, replace the fixed `token_map` for indices 2–7 with `sizing_token(bet_total, pot_at_decision)` where `pot_at_decision` is the pot before chips are added (matches engine `pot_before` semantics; engine `amount` is the same total-contribution target as `bet_total`). F/K/C tokens stay as-is — already consistent.
3. Side effect: existing `cfr_regret_deep_v2.pkl` keys become stale (they're already partially phantom). Loader stays unchanged; profile is superseded by the planned retrain.

---

## Issue 4 — Path A inference search instability

**Verdict: REAL. Severity: CRITICAL for live play (does not affect training).**

Where: `_subgame_search` (`cfr_bot.py:1864–1906`), `_search_subtree` (1908–1951).

- Refined weight = `max(prior, 1e-6) · exp(ev − max_ev)` with `ev` in **raw chips** (line 1894). A 20-chip EV gap → e⁻²⁰ ≈ 2·10⁻⁹: the trained average strategy is annihilated and the bot plays argmax over the EVs.
- Those EVs are extremely noisy: one sampled opponent action per node (1936–1938), hero subnodes take `max` (1941–1951, compounding optimism), and opponents have **no hole cards** at inference (`_build_game_state_from_view` line 1085 only fills hero), so their info keys get card bucket 0 → usually unseen → uniform play. Net effect: single-sample noisy argmax overrides the trained table. Confirms the suspicion exactly.
- Path B already solved this (`deep_cfr_bot.py:1140–1142, 1731–1762`): costs normalized to **BB units**, baseline = prior-weighted mean value, advantage / temperature 20, clip ±2, `w = prior · exp(adv)`, then **blend** `0.75·prior + 0.25·refined`.

**Minimal fix: port Path B's shaping to `CFRBot._subgame_search`** (~15 lines):
normalize `ev` and `added_cost` by `gs.big_blind`; baseline = Σ prior·ev; advantage = (ev − baseline)/T with T=20; clip to ±2; `w = prior·exp(adv)`; normalize; blend 0.25 toward refined. Optional (separate knob, not required for the patch): average k>1 sampled opponent lines per action to cut variance, and replace hero-node `max` with prior-weighted value below the root.

---

## Issue 5 — Tests to add (new sanity scripts, ladder-style)

1. **`sanity_mccfr_es_update.py`** — estimator correctness. Freeze node strategies on a tiny hand-built 2-seat `_GameState` with one opponent decision (probs p, 1−p) and known leaf values; run N traversals; assert E[regret increment] ≈ π₋ᵢ·(u(a)−ū) (single power of p, not p²) within MC tolerance. Also assert: after one traversal, `strategy_sum` incremented at the opponent node (unweighted) and not at hero nodes.
2. **`sanity_mccfr_tiny_game.py`** (stretch, "if practical") — mini-game convergence: a 1-street toy where the equilibrium is computable by hand (or Kuhn-style); assert average strategy approaches it / exploitability decreases. Useful but the estimator test above is the load-bearing check.
3. **`sanity_cfr_token_parity.py`** — round-trip parity: for a grid of spots (opening bets at each bucket, raises facing bets, min-raise clamp, sizing-clamped-to-all-in, short-stack all-in-for-less, multi-raise sequences), apply the tree action, build the equivalent engine-style `ActionEvent(amount=bet_total, pot_before=pot)`, and assert `apply_action`'s appended token == canonical `tokenize()` token. Plus a randomized property pass.
4. **`sanity_cfr_search_shaping.py`** — shaping sanity: (a) with chip-scale EV gaps and a prior favoring another action, refined must not be one-hot (prior retains ≥ floor mass via blend); (b) scale invariance: multiply pot/stacks/blind by 10 → refined distribution unchanged; (c) two EVs within noise of each other → prior dominates the choice.
5. Wire all of the above into `sanity_validation_ladder.py`.

Existing `sanity_cfr_sizing_parity.py` covers concrete-amount parity (train vs deploy `current_bet`), not token parity — keep it, it's complementary.

---

## Fix order relative to MCCFR retrain

| Fix | Must precede retrain? | Why |
|---|---|---|
| 1a. Remove `cf_reach` from regret update | **Yes** | Shapes every regret in the trained table |
| 1b. Move strategy-sum to opponent nodes | **Yes** | Shapes the average strategy you'll deploy |
| 3. Canonical tokenization in `apply_action` | **Yes** | Changes the info-set key space; retraining on the old keys wastes the run |
| 2. Doc fix (or MCCFR+ flag) | Yes (trivial) | Docs should describe what the retrain actually runs |
| 4. Search shaping | No (inference-only) | But do it in the same patch so the next eval is meaningful |

Estimated total diff: ~35 lines of production code across `cfr_bot.py` + `core/action_history.py`, plus docs and 3–4 test scripts. No changes to MLBot, PPO/RL, or Deep CFR.

---

## Implementation record (approved option: fix code, rename docs)

All fixes implemented 2026-06-10 after approval:

1. **`bots/cfr_bot.py` `_cfr_recurse`** — `cf_reach` removed from regret increments (unweighted update); `strategy_sum` now accumulates at opponent nodes, unweighted (textbook Lanctot ES); `reach_probs` parameter and bookkeeping deleted (signature is now `_cfr_recurse(state, hero_seat, depth)`). Docstring documents why no reach weighting appears.
2. **`core/action_history.py`** — `sizing_token(amount, pot_before)` extracted from `tokenize()` (no behavior change to live tokenization). **`_GameState.apply_action`** now tokenizes sizing actions from the realized total street contribution and pre-action pot via `sizing_token`, instead of fixed per-bucket tokens. Existing `cfr_regret_deep_v2.pkl` deep-node keys are superseded; retrain planned anyway.
3. **`CFRBot._subgame_search`** — EVs converted to big-blind units; shaping extracted into `_shape_search_strategy()` mirroring Path B: prior-weighted baseline, temperature 20, advantage clip ±2, 25% blend into the trained prior.
4. **Docs** — `TRAINING_PLAN.md` Path A paragraph and `cfr_bot.py` module docstring now say vanilla external-sampling MCCFR (not MCCFR+). Session logs left untouched as historical record.

New gates (wired into `sanity_validation_ladder.py`, Tier 2 / cfr):

- `sanity_mccfr_es_update.py` — frozen-strategy MC estimator test: measured mean call-regret increment +7.35 vs expected +7.50 (buggy double-count would give +3.75); strategy-sum placement checks exact.
- `sanity_cfr_token_parity.py` — tree token == canonical token across opens, raises, min-raise clamps, all-in clamps, all-in-for-less calls + 200-spot random sweep (1333 pairs).
- `sanity_cfr_search_shaping.py` — anti-collapse on 50 BB EV gaps, 10× chip-scale invariance, prior dominance at noise-scale EV gaps.

Validation: `--path cfr` ladder — every torch-free gate PASS including all three new gates and `sanity_cfr_sizing_parity` / `sanity_cfr_equity` / `sanity_aivat`. (5 gates failed only with `ModuleNotFoundError: torch` in the sandbox — environment, not code; re-run the ladder locally where torch is installed.)

---

# Phase 3.1 addendum — second-audit blockers verified & fixed (2026-06-11)

A second audit pass raised two candidate retrain blockers on top of the Phase-3 fixes. Both were verified REAL by direct probes, then fixed.

## Blocker A — decision-root infosets had regret mass but zero strategy_sum. REAL.

- Traversals are decision-rooted with the hero acting first (`_build_training_game_state` puts hero at `seat_order[0]`), and Phase 3 moved `strategy_sum` accumulation to opponent nodes only. An infoset that only ever occurs as a traversal root therefore never receives average-strategy mass. Guaranteed-zero case: the hand's FIRST voluntary action — blinds are not history entries (engine.py `post_blind`), so its token history is empty, and an empty-history node is by construction always the root with the actor as traverser. 6-max: every UTG opening infoset; heads-up: the BTN/SB first decision. `act()` then discarded the training (`strategy_sum == 0` → search/heuristic fallback).
- Probe (pre-fix): root key `preflop:5:early:high:TA-TA-TA-TA-TA:25:` → regret mass 119,422, strategy mass 0.0, inference `act()` took `_search_fallback`.
- Fix 1 (`_cfr_recurse`): `strategy_sum` also accumulates at the depth-0 hero node. Hero reach at the decision root is exactly 1, so the unweighted increment `+= strategy[a]` is the textbook average-strategy update there — deeper hero nodes still accumulate nothing.
- Fix 2 (`act()`): when a node has positive regret on the legal actions but no average mass (`strategy_sum == 0`), deploy the regret-matched current strategy instead of falling back. Covers profiles trained before Fix 1 and regret-only infosets generally.
- Gate: `sanity_cfr_root_coverage.py` (ladder Tier 2 / cfr) — root mass, no-fallback deployment, regret-only deployment. Fails pre-fix.

## Blocker B — short all-in CALL reconstructed as the full to_call. REAL.

- Engine recorded `amount=None` for every call (`action_amt = None` for non-bet/raise actions) while paying `min(stack, to_call)`. `_reconstruct_contributions_from_view` substitutes `to_call_before` when `amount is None` — correct for full calls, over-counted for call-for-less. The rebuilt contributions then exceeded the public pot, `is_real=False`, and `_build_training_game_state()` returned None: every MCCFR traversal at such decisions was silently skipped.
- Probe (pre-fix, real engine hand — UTG raises to 200, BTN with 60 calls all-in): call entry `amount=None, to_call_before=200`; reconstructed 415 vs `view.pot` 275; training state None.
- Fix (`core/engine.py`): history records the chips actually paid for calls (`min(seat.chips, to_call)`). The `to_call_before` fallback remains for legacy/synthetic histories. Call tokens ("C") ignore amount, so info-set keys are unchanged; Path B tensors get the real call amount with train/serve parity preserved (both read the same engine history).
- Gate: `sanity_cfr_allin_call_reconstruction.py` (ladder Tier 2 / all) — recorded amount, pot parity, state built, full-call regression, legacy-None fallback. Fails pre-fix (verified by stash-revert run).

## Profile safety — format_version gate (pre-retrain requirement)

- `save()` stamps `format_version = PROFILE_FORMAT_VERSION` (= 3). `load()` REJECTS missing/lower versions when `inference_mode=False` unless `allow_stale_profile=True`. Inference-mode loads stay permitted (eval/play vs old profiles). Landed before the retrain on purpose: checkpoints written mid-run are stamped v3 and resume cleanly; the pre-Phase-3 `cfr_regret_deep.PRE_MULTIWAY.bak` is rejected for training (verified).
- `models/cfr_regret_deep_v2.pkl` does not currently exist — the default retrain path starts fresh; the gate guards against pointing at any stale file.

## Docs

- Nash-convergence claims removed/softened: `cfr_bot.py` module docstring, `_CFRNode.get_average_strategy`, `use_average` param doc, `README.md` CFR section, `TRAINING_PLAN.md` Path A (now also records root-averaging). The setup is vanilla external-sampling MCCFR + abstraction + decision-rooted engineering approximations: a regret-minimising approximation validated by eval, not a guaranteed-Nash solver.
