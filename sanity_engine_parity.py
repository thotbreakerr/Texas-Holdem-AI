"""Rules-parity harness: legacy core.engine.Table vs the Poker-Engine adapter.

Runs identical scenarios (same seats/blinds/dealer, same scripted bots, and —
for showdown scenarios — the SAME injected hole cards and board) through both
engines and asserts identical decision traces, histories, and net chip results.

Covers the rule-delta checklist from docs/plans/ENGINE_MIGRATION_PLAN.md §4:
  R1 lone-live all-in closure     R2 short all-in does not reopen
  R3 heads-up button order        R4 preflop UTG / postflop SB first
  R5 short-stack call amount       R7 uncalled bet returned (fold-win)
  R8 multiway side pots            R9 bet-vs-raise history labels
plus a large injected-card fuzz. R6 (odd-chip split) is checked with a
documented tolerance: the adapter intentionally adopts Poker-Engine's standard
"first winner left of the button" odd-chip rule, so totals + winner sets must
match but a single leftover chip may go to a different winner.

Run:  .venv/bin/python sanity_engine_parity.py
Exit code 0 = all parity checks passed.
"""

import random
import sys

from core.engine import Seat, Table, InProcessBot, _FULL_DECK
from core.pe_engine import PokerEngineTable, _legacy_to_pe, _positions

from engine.cards import Deck as PEDeck  # noqa: E402  (via pe_engine bootstrap)


# --------------------------------------------------------------------------
# scripted bot
# --------------------------------------------------------------------------

class ScriptedBot:
    """Acts from a fixed queue of step tokens, recording every decision into a
    shared trace. The same instance-config drives both engines so any rules
    divergence shows up as a trace/net difference."""

    def __init__(self, pid, steps, trace):
        self.pid = pid
        self.steps = list(steps)
        self.i = 0
        self.trace = trace

    def _next_step(self):
        if self.i < len(self.steps):
            step = self.steps[self.i]
            self.i += 1
            return step
        return "check_call"

    @staticmethod
    def _spec(legal, *types):
        for want in types:
            for a in legal:
                if a["type"] == want:
                    return a
        return None

    def _choose(self, view):
        legal = view.legal_actions
        step = self._next_step()
        name = step[0] if isinstance(step, tuple) else step

        if name == "fold" and self._spec(legal, "fold"):
            return {"type": "fold"}
        if name == "check":
            if self._spec(legal, "check"):
                return {"type": "check"}
            if self._spec(legal, "call"):
                return {"type": "call"}
            return {"type": "fold"}
        if name in ("call", "check_call"):
            if self._spec(legal, "call"):
                return {"type": "call"}
            if self._spec(legal, "check"):
                return {"type": "check"}
            return {"type": "fold"}
        if name in ("raise", "bet"):
            spec = self._spec(legal, "raise", "bet")
            if spec:
                total = step[1] if isinstance(step, tuple) else spec["min"]
                total = max(spec["min"], min(spec["max"], total))
                return {"type": spec["type"], "amount": total}
            # fall through to stay-in
            return self._choose_fallback(legal)
        if name == "minraise":
            spec = self._spec(legal, "raise", "bet")
            if spec:
                return {"type": spec["type"], "amount": spec["min"]}
            return self._choose_fallback(legal)
        if name == "allin":
            spec = self._spec(legal, "raise", "bet")
            if spec:
                return {"type": spec["type"], "amount": spec["max"]}
            if self._spec(legal, "call"):
                return {"type": "call"}
            return {"type": "check"} if self._spec(legal, "check") else {"type": "fold"}
        return self._choose_fallback(legal)

    @staticmethod
    def _choose_fallback(legal):
        for want in ("check", "call", "fold"):
            for a in legal:
                if a["type"] == want:
                    return {"type": want}
        return {"type": "fold"}

    def act(self, view):
        action = self._choose(view)
        # Record a hashable snapshot of everything that should be engine-invariant.
        legal_key = tuple(tuple(sorted(a.items())) for a in view.legal_actions)
        hist_key = tuple(tuple(sorted(e.items(), key=lambda kv: kv[0]))
                         for e in view.history)
        # "contested" = the actor still has at least one opponent who can make a
        # future betting decision. Lone-runout decisions (everyone else all-in
        # or folded) are net-neutral and the legacy engine prompts them while
        # the Poker-Engine adapter elides them, so they are excluded from the
        # decision-trace comparison (net is still compared exactly).
        # Seat ORDER is part of the PlayerView contract: stacks /
        # seat_indices / opponents are listed dealer-first, so downstream
        # code may read the blinds off indices 1/2 (0/1 heads-up) and hero's
        # dict index must carry hero's own position label.
        seat_order = tuple(view.stacks)
        labels = _positions(len(seat_order))
        self.trace.append({
            "contested": len(view.acting_opponents or []) > 0,
            "street": view.street, "me": view.me, "position": view.position,
            "to_call": view.to_call, "pot": view.pot,
            "legal": legal_key, "chosen": (action["type"], action.get("amount")),
            "hist": hist_key,
            "hole": tuple(sorted(view.hole_cards)), "board": tuple(view.board),
            "seat_order": seat_order,
            "seat_indices": tuple((view.seat_indices or {}).items()),
            "opponents": tuple(view.opponents or ()),
            "acting": tuple(view.acting_opponents or ()),
            "all_in": tuple(view.all_in_opponents or ()),
            "label_ok": (view.me in seat_order
                         and labels[seat_order.index(view.me)] == view.position),
        })
        return action


# --------------------------------------------------------------------------
# deck injection
# --------------------------------------------------------------------------

class _FixedDeckRng(random.Random):
    """A Random whose shuffle() imposes a fixed deck order (legacy engine)."""

    def __init__(self, deck):
        super().__init__(0)
        self._deck = list(deck)

    def shuffle(self, x, *args, **kwargs):
        x[:] = self._deck


def _ring_dealer_first(seats, dealer_index):
    n = len(seats)
    order = list(range(dealer_index, n)) + list(range(0, dealer_index))
    return [i for i in order if seats[i].chips > 0 and not seats[i].is_sitting_out]


def _target_cards(ring_pids, rng):
    """Assign 2 holes per pid (dealer-first) + a 5-card board from a shuffle."""
    deck = list(_FULL_DECK)
    rng.shuffle(deck)
    holes, k = {}, 0
    for pid in ring_pids:
        holes[pid] = (deck[k], deck[k + 1])
        k += 2
    board = deck[k:k + 5]
    used = set(sum([list(h) for h in holes.values()], [])) | set(board)
    filler = [c for c in _FULL_DECK if c not in used]
    return holes, board, filler


def _legacy_deck(ring_pids, holes, board, filler):
    """Legacy pops from the END, 2 per player in ring order, then board."""
    pop_seq = []
    for pid in ring_pids:
        pop_seq.extend([holes[pid][0], holes[pid][1]])
    pop_seq.extend(board)
    return list(filler) + list(reversed(pop_seq))


def _pe_preset_deck(ring_pids, holes, board, filler):
    """PE deals from the FRONT: round1 each seat, round2 each seat, then board.
    PE deal order = ring[1:] + [ring[0]]."""
    deal_order = ring_pids[1:] + ring_pids[:1]
    front = [holes[p][0] for p in deal_order] + [holes[p][1] for p in deal_order]
    front = front + list(board) + list(filler)
    return PEDeck(preset_order=[_legacy_to_pe(c) for c in front])


# --------------------------------------------------------------------------
# scenario runner
# --------------------------------------------------------------------------

def _mk_seats(spec):
    return [Seat(pid, chips) for pid, chips in spec]


def run_scenario(scn, inject_cards=True, card_rng=None):
    """Play one scenario on both engines; return (legacy, pe) result dicts."""
    seat_spec = scn["seats"]
    sb, bb = scn["sb"], scn["bb"]
    dealer = scn.get("dealer", 0)
    ante = scn.get("ante", 0)
    scripts = scn["scripts"]

    ring_pids = [_mk_seats(seat_spec)[i].player_id
                 for i in _ring_dealer_first(_mk_seats(seat_spec), dealer)]

    holes = board = filler = None
    if inject_cards:
        holes, board, filler = _target_cards(ring_pids, card_rng or random.Random(0))

    def _play(TableCls, use_pe):
        seats = _mk_seats(seat_spec)
        trace = []
        bots = {pid: InProcessBot(ScriptedBot(pid, scripts.get(pid, []), trace))
                for pid, _ in seat_spec}
        table = TableCls(rng=random.Random(1234))
        kwargs = {}
        if inject_cards:
            if use_pe:
                kwargs["preset_deck"] = _pe_preset_deck(ring_pids, holes, board, filler)
            else:
                table.rng = _FixedDeckRng(_legacy_deck(ring_pids, holes, board, filler))
        net = table.play_hand(seats, sb, bb, dealer, bots, ante=ante, **kwargs)
        return {"net": net, "trace": trace}

    legacy = _play(Table, use_pe=False)
    pe = _play(PokerEngineTable, use_pe=True)
    return legacy, pe


# --------------------------------------------------------------------------
# scenarios (R-checklist)
# --------------------------------------------------------------------------

SCENARIOS = [
    {   # R3/R4/R9: 3-handed limped pot to showdown; bet then raise postflop.
        "name": "R3/R4/R9 order + bet-vs-raise labels",
        "seats": [("A", 10000), ("B", 10000), ("C", 10000)], "sb": 50, "bb": 100,
        "dealer": 0,
        "scripts": {
            "A": ["call", "check", "check", "check"],   # BTN
            "B": ["call", "check", "check", "check"],   # SB
            "C": ["check", ("bet", 300), "check", "check"],  # BB opens flop
        },
    },
    {   # R7: heads-up over-bet, opponent folds -> uncalled chips returned.
        "name": "R7 uncalled bet returned (fold-win)",
        "seats": [("A", 10000), ("B", 10000)], "sb": 50, "bb": 100, "dealer": 0,
        "scripts": {"A": [("raise", 900)], "B": ["fold"]},
    },
    {   # R1/R5: short stack shoves preflop, big stack must be offered a decision
        # and calls for less; call amount recorded is min(stack, to_call).
        "name": "R1/R5 lone-live all-in closure + short call amount",
        "seats": [("A", 220), ("B", 10000)], "sb": 50, "bb": 100, "dealer": 0,
        "scripts": {"A": ["allin"], "B": ["call", "check", "check", "check"]},
    },
    {   # R2: preflop raise, caller, then a short all-in that cannot reopen —
        # the earlier caller must not be offered a re-raise.
        "name": "R2 short all-in does not reopen",
        "seats": [("A", 10000), ("B", 900), ("C", 10000)], "sb": 50, "bb": 100,
        "dealer": 0,
        "scripts": {
            "A": [("raise", 300), "call", "check", "check", "check"],  # BTN opens
            "B": ["allin"],                                            # SB short shove
            "C": ["call", "check", "check", "check"],                  # BB
        },
    },
    {   # R2b: a sub-min all-in OPENING bet must reopen the betting to a
        # player who already checked (legacy treats any opening bet as full).
        # C posts the BB with only 155, leaving 55 behind; on the flop C opens
        # all-in for 55 (< bb 100). When action returns to B (who checked), B
        # must be offered a re-raise. This is the divergence legacy_compat fixes.
        "name": "R2b sub-min all-in open reopens the betting",
        "seats": [("A", 10000), ("B", 10000), ("C", 155)], "sb": 50, "bb": 100,
        "dealer": 0,
        "scripts": {
            "A": ["call", "call"],            # BTN limps, then calls C's shove
            "B": ["call", "check", "call"],   # SB limps, checks flop, then calls
            "C": ["check", "allin"],          # BB checks option, shoves 55 on flop
        },
    },
    {   # R8: 3-way all-in with unequal stacks -> main + side pots.
        "name": "R8 multiway side pots",
        "seats": [("A", 1500), ("B", 3000), ("C", 6000)], "sb": 50, "bb": 100,
        "dealer": 0,
        "scripts": {"A": ["allin"], "B": ["allin"], "C": ["call"]},
    },
]


def _compare(name, legacy, pe, odd_chip_tolerant=False):
    problems = []
    # Compare only contested decisions (see ScriptedBot.act).
    lt = [d for d in legacy["trace"] if d["contested"]]
    pt = [d for d in pe["trace"] if d["contested"]]
    if lt != pt:
        for i, (a, b) in enumerate(zip(lt, pt)):
            if a != b:
                diff = [k for k in a if a[k] != b.get(k)]
                problems.append(f"contested decision {i} differs on {diff}:\n"
                                f"    legacy={ {k: a[k] for k in diff} }\n"
                                f"    pe    ={ {k: b.get(k) for k in diff} }")
                break
        else:
            problems.append(f"contested trace length differs: "
                            f"legacy={len(lt)} pe={len(pt)}")
    ln, pn = legacy["net"], pe["net"]
    if set(ln) != set(pn):
        problems.append(f"net keys differ: {set(ln)} vs {set(pn)}")
    elif ln != pn:
        if odd_chip_tolerant:
            total_ok = sum(ln.values()) == sum(pn.values()) == 0
            close = all(abs(ln[p] - pn[p]) <= 1 for p in ln)
            if not (total_ok and close):
                problems.append(f"net differs beyond odd-chip tolerance:\n"
                                f"    legacy={ln}\n    pe    ={pn}")
        else:
            problems.append(f"net differs:\n    legacy={ln}\n    pe    ={pn}")
    return problems


def main():
    failures = 0
    checks = 0

    for idx, scn in enumerate(SCENARIOS):
        checks += 1
        # Deterministic per-scenario card seed (NOT hash(), which is
        # process-randomized). Odd-chip tolerance applies because a random
        # showdown layout can chop, and a chopped pot legitimately differs by
        # one chip between the engines (documented divergence).
        legacy, pe = run_scenario(scn, inject_cards=True,
                                  card_rng=random.Random(1000 + idx))
        problems = _compare(scn["name"], legacy, pe, odd_chip_tolerant=True)
        if problems:
            failures += 1
            print(f"CHECK {checks} FAIL: {scn['name']}")
            for p in problems:
                print("   " + p.replace("\n", "\n   "))
        else:
            print(f"CHECK {checks} PASS: {scn['name']} "
                  f"(net={legacy['net']})")

    # R6: engineered split pot with an odd chip (tolerant comparison).
    checks += 1
    split = {
        "name": "R6 split pot odd chip (tolerant)",
        "seats": [("A", 10000), ("B", 10000), ("C", 10000)], "sb": 50, "bb": 100,
        "dealer": 0,
        "scripts": {  # everyone limps and checks down; board plays -> chop
            "A": ["call", "check", "check", "check"],
            "B": ["call", "check", "check", "check"],
            "C": ["check", "check", "check", "check"],
        },
    }
    # Board is a royal flush so all three chop (board plays) -> 3-way split of 300.
    ring_pids = [_mk_seats(split["seats"])[i].player_id
                 for i in _ring_dealer_first(_mk_seats(split["seats"]), 0)]
    holes = {ring_pids[0]: (("2", "c"), ("3", "d")),
             ring_pids[1]: (("4", "c"), ("5", "d")),
             ring_pids[2]: (("7", "c"), ("8", "d"))}
    board = [("A", "h"), ("K", "h"), ("Q", "h"), ("J", "h"), ("T", "h")]
    used = set(sum([list(h) for h in holes.values()], [])) | set(board)
    filler = [c for c in _FULL_DECK if c not in used]

    def _play_fixed(TableCls, use_pe):
        seats = _mk_seats(split["seats"])
        trace = []
        bots = {pid: InProcessBot(ScriptedBot(pid, split["scripts"].get(pid, []), trace))
                for pid, _ in split["seats"]}
        t = TableCls(rng=random.Random(1))
        if use_pe:
            net = t.play_hand(seats, 50, 100, 0, bots,
                              preset_deck=_pe_preset_deck(ring_pids, holes, board, filler))
        else:
            t.rng = _FixedDeckRng(_legacy_deck(ring_pids, holes, board, filler))
            net = t.play_hand(seats, 50, 100, 0, bots)
        return {"net": net, "trace": trace}

    legacy = _play_fixed(Table, False)
    pe = _play_fixed(PokerEngineTable, True)
    problems = _compare(split["name"], legacy, pe, odd_chip_tolerant=True)
    if problems:
        failures += 1
        print(f"CHECK {checks} FAIL: {split['name']}")
        for p in problems:
            print("   " + p.replace("\n", "\n   "))
    else:
        print(f"CHECK {checks} PASS: {split['name']} "
              f"(legacy={legacy['net']} pe={pe['net']})")

    # Fuzz: many injected-card hands, 2-6 players, random scripts.
    checks += 1
    fuzz_fail = 0
    N = 300
    rng = random.Random(20260706)
    for t in range(N):
        nplayers = rng.randint(2, 6)
        stacks = [("P%d" % i, rng.choice([300, 800, 1500, 5000, 10000]))
                  for i in range(nplayers)]
        steps_pool = ["check_call", "call", "check", ("raise", None), "minraise",
                      "allin", "fold"]

        def mk_steps():
            out = []
            for _ in range(rng.randint(1, 5)):
                s = rng.choice(steps_pool)
                if s == ("raise", None):
                    s = ("raise", rng.choice([200, 400, 800, 2000]))
                out.append(s)
            return out

        scn = {
            "name": f"fuzz-{t}",
            "seats": stacks, "sb": 50, "bb": 100,
            "dealer": rng.randint(0, nplayers - 1),
            "ante": rng.choice([0, 0, 0, 25]),
            "scripts": {pid: mk_steps() for pid, _ in stacks},
        }
        legacy, pe = run_scenario(scn, inject_cards=True, card_rng=random.Random(t))
        problems = _compare(scn["name"], legacy, pe, odd_chip_tolerant=True)
        if problems:
            fuzz_fail += 1
            if fuzz_fail <= 5:
                print(f"   FUZZ FAIL {scn['name']} (seats={stacks}, "
                      f"dealer={scn['dealer']}, ante={scn['ante']}):")
                for p in problems:
                    print("      " + p.replace("\n", "\n      "))
    if fuzz_fail:
        failures += 1
        print(f"CHECK {checks} FAIL: fuzz {fuzz_fail}/{N} hands diverged")
    else:
        print(f"CHECK {checks} PASS: fuzz {N}/{N} injected-card hands match")

    print()
    if failures:
        print(f"PARITY FAILED: {failures} check group(s) failed")
        return 1
    print(f"PARITY OK: all {checks} check groups passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
