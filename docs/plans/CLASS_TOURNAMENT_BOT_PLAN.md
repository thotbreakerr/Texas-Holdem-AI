# Class Tournament Bot Execution Plan

## Objective

Build a one-submission poker bot with two selectable final variants:

- `final_survival`: default if we have no useful opponent intel.
- `final_aggro`: optional match-day choice if we learn the field is passive, tight, or overly simple.

The goal is not theoretical poker purity. The goal is to maximize the chance of winning a single graded 6+ player winner-take-all class tournament.

## Locked Tournament Assumptions

- Engine: this repository's engine unless the class adopts another shared engine.
- Players: 6+ players.
- Starting stack: `1000` chips.
- Initial blinds: `5 / 10`.
- Blind schedule: existing project schedule, `1.5x` every `50` hands.
- Ante: off unless the class explicitly agrees otherwise.
- Winner: last bot standing.
- Saved checkpoints/models: allowed.
- Per-move time limit: none expected, so heavier compute is allowed in important spots.

## Strategy Decision

Build one shared hybrid bot implementation with two profiles:

```text
bots/tournament_hybrid_bot.py
  - TournamentHybridBot(profile="survival")
  - TournamentHybridBot(profile="aggro")
```

Register aliases:

```text
final            -> survival by default
final_survival   -> survival profile
final_aggro      -> aggro profile
```

This keeps the submission simple while preserving a match-day choice.

## Core Bot Architecture

The final bot should use only the public `PlayerView` contract and return legal `Action` objects.

Main components:

1. Legal-action safety layer
2. Stack/tournament state detector
3. Preflop hand evaluator and range tables
4. Monte Carlo postflop equity evaluator
5. Survival profile
6. Aggressive profile
7. Optional checkpoint advisors
8. Deterministic fallback path

Checkpoint advisors are useful only if they improve tournament win rate in evaluation. The final bot must not crash or become useless if a checkpoint is missing.

## Profile Behavior

### `final_survival`

Default choice when opponent styles are unknown.

Behavior:

- Tight-aggressive early full-table play.
- Avoid marginal early all-ins.
- Value bet strong hands because class bots may overcall.
- Fold dominated hands facing large multiway aggression.
- Avoid huge flips against bigger stacks unless equity is clearly favorable.
- Become more aggressive when short-stacked or short-handed.

Best against:

- Unknown fields.
- Maniac-heavy fields.
- Random or over-aggressive bots.
- Multiway pots with high variance.

### `final_aggro`

Backup choice if we learn the class field is passive, tight, or timid.

Behavior:

- Wider late-position opens.
- More blind stealing.
- More pressure on limpers.
- More continuation betting on favorable boards.
- More pressure from chip-leader or top-2 stack positions.
- Still avoid pure punt all-ins without equity or fold equity.

Best against:

- Tight/passive bots.
- Bots that overfold blinds.
- Simple rule bots that do not adapt.
- Tables where early chip accumulation is unusually easy.

## Tournament State Modes

Each decision should classify the current state.

Stack depth:

```text
desperate: <= 6 BB
short:     <= 12 BB
medium:    <= 30 BB
deep:      > 30 BB
```

Table size:

```text
full_table: 5+ players alive
short_handed: 3-4 players alive
heads_up: 2 players alive
```

Stack rank:

```text
chip_leader
top_2
middle
short_stack
```

Use these modes to route between cautious survival, pressure, and shove/fold behavior.

## Compute Policy

Because no per-move time limit is expected, use more compute in important spots.

Suggested Monte Carlo budgets:

```text
obvious preflop fold/check:       0 sims
normal postflop decision:         500-2,000 sims
large pot:                        5,000-20,000 sims
river/all-in/tournament life:     20,000-100,000 sims if practical
```

Always return an action. Heavy compute is allowed; crashing or hanging forever is not.

## Implementation Phases

### Phase 1: Bot Skeleton

Files:

- `bots/tournament_hybrid_bot.py`
- `bots/__init__.py`

Tasks:

- Create `TournamentHybridBot`.
- Add `profile` argument with `"survival"` and `"aggro"`.
- Implement legal-action helper methods:
  - choose fold/check/call safely
  - choose bet/raise within legal min/max
  - detect all-in raise
- Register aliases:
  - `final`
  - `final_survival`
  - `final_aggro`

Acceptance:

- `create_bot("final")` succeeds.
- `create_bot("final_survival")` succeeds.
- `create_bot("final_aggro")` succeeds.
- Bot never returns illegal action in smoke tests.

### Phase 2: Preflop Strategy

Tasks:

- Implement hand classification:
  - pairs
  - suitedness
  - connectors
  - broadways
  - ace-high strength
  - position
  - stack depth
- Add survival preflop ranges.
- Add aggro preflop ranges.
- Add short-stack shove/fold behavior.

Acceptance:

- Survival folds obvious junk early full-table.
- Aggro opens wider from late position.
- Both profiles shove reasonable short-stack hands.
- Neither profile open-jams deep stacks with trash.

### Phase 3: Postflop Equity And Pot Odds

Tasks:

- Add Monte Carlo equity estimation using known hero cards, board, and random opponent ranges.
- Compare equity to pot odds.
- Adjust threshold by:
  - number of opponents
  - stack risk
  - all-in risk
  - profile
  - street
- Add value-bet sizing rules.
- Add bluff/semi-bluff rules for aggro profile.

Acceptance:

- Bot calls more often with strong equity.
- Bot folds weak equity to large bets.
- Bot value bets strong made hands.
- Aggro profile applies more pressure than survival.

### Phase 4: Tournament-Specific Logic

Tasks:

- Add stack-rank detection.
- Add chip-leader pressure mode.
- Add short-stack desperation mode.
- Add heads-up aggression mode.
- Add early-bust avoidance for survival profile.

Acceptance:

- Survival avoids marginal tournament-life calls early.
- Aggro pressures when chip leader/top-2 stack.
- Both profiles loosen heads-up.
- Short-stack mode does not blind down forever.

### Phase 5: Opponent Tendencies

Tasks:

- Track simple per-opponent stats from `PlayerView.history`:
  - aggression frequency
  - call frequency
  - fold/passivity clues
  - large bet frequency
- Use tendencies lightly:
  - bluff less into calling stations
  - steal more against folders
  - trap/call stronger against maniacs

Acceptance:

- Bot adapts within a match without needing external data.
- Tendencies reset cleanly between tournaments.

### Phase 6: Optional Checkpoint Advisors

Tasks:

- Optionally consult CFR or Deep CFR if available.
- Do not let advisor output override safety/tournament-life rules blindly.
- Ignore advisor if missing, slow to initialize, or clearly illegal.

Acceptance:

- Bot works with no model files.
- Bot works with model files.
- Advisor improves eval win rate before being enabled by default.

### Phase 7: Stress Opponents

Add simple opponent bots if not already present:

- nit/folder
- maniac/all-in pressure
- calling station
- loose passive
- min-raiser

Acceptance:

- They can be created by `create_bot` or local eval harness.
- They are used in tournament evaluation fields.

## Evaluation Plan

Use the exact locked tournament assumptions:

```text
1000 chips
5/10 blinds
blind increase every 50 hands
ante off
6-player winner-take-all
```

Run quick tests after each meaningful change:

```bash
.venv/bin/python run_eval.py --mode multiway --tournaments 200 --chips 1000 --sb 5 --bb 10 --blind-increase-every 50 --pool smart,mc200,gto,icm,exploitative
```

Run larger tests before trusting a profile:

```bash
.venv/bin/python run_eval.py --mode multiway --tournaments 1000 --chips 1000 --sb 5 --bb 10 --blind-increase-every 50 --pool smart,mc200,gto,icm,exploitative
```

If needed, add a dedicated final-bot eval mode or temporary script that puts `final_survival` or `final_aggro` in the Path B / candidate seat.

Track:

- tournament win rate
- Wilson confidence interval
- average finish position
- early bust rate
- heads-up reached rate
- heads-up conversion rate
- all-in equity when called
- average decision time in heavy spots

## Final Selection Rule

Default:

```text
submit/use final_survival
```

Use `final_aggro` only if one of these is true:

- Evaluation shows it clearly beats survival overall.
- We learn the class field is passive/tight.
- We learn opponents overfold blinds or avoid big pots.

No-intel match-day choice:

```text
final = final_survival
```

## Execution Trigger

When the user says to execute this plan, start with Phase 1 and proceed in order:

1. Implement `TournamentHybridBot`.
2. Register bot aliases.
3. Add smoke tests.
4. Run sanity/eval checks.
5. Iterate through later phases only after the current phase passes.

Do not rewrite the engine unless a concrete bug blocks the bot. Treat the engine rules as frozen.

