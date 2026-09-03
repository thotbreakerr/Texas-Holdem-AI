"""Poker-Engine-backed drop-in for :class:`core.engine.Table`.

This adapter exposes the exact ``PlayerView`` / ``BotAdapter`` / ``play_hand``
contract the rest of Texas-Holdem-AI depends on, but runs the sibling
``Poker-Engine`` repo's tested rules core (dealing, betting, closure, side
pots) underneath. It is the default engine as of P6 (see :func:`make_table` and
the ``THAI_ENGINE_IMPL`` env var); the legacy engine remains selectable via
``engine_impl="legacy"`` for the legacy regression suite.

Design (see docs/plans/ENGINE_MIGRATION_PLAN.md):
  * Each legacy bot is wrapped as a Poker-Engine "agent" (duck-typed ``act``).
  * On every decision the wrapper builds the legacy ``PlayerView`` from the
    Poker-Engine ``Observation``, runs the bot, sanitizes exactly as the
    legacy engine does, and records a byte-exact legacy history entry
    (``{street, pid, type, amount, to_call_before, pot_before}``) — blinds and
    antes are deliberately NOT recorded, matching the legacy engine, because
    downstream CFR / opponent-stat code reconstructs blinds from the first
    preflop ``pot_before``.
  * ``stacks`` / ``seat_indices`` / ``opponents`` (and the acting / all-in
    splits) are listed DEALER-FIRST like the legacy engine's ring, not in
    Poker-Engine's absolute seat order: bots and tooling index that order
    positionally (the blinds at indices 1/2, or 0/1 heads-up; hero's own
    position label at hero's index). ``sanity_engine_parity.py`` pins it.
  * Showdown ranking is delegated back to the legacy ``eval_hand`` via
    Poker-Engine's ``rank_fn`` seam, so winners/ties are identical to legacy.

One intentional behavioral change vs. legacy: the odd chip in a split pot goes
to the first winner clockwise from the button (standard poker), whereas the
legacy engine gave it to the earliest contributor in dict-insertion order.
Totals are conserved; only the recipient of a single leftover chip can differ.
"""

import os
import random
import sys
from collections import deque
from typing import Any, Dict, List, Optional

from core.bot_api import PlayerView
from core.engine import RANKS, SUITS, Seat, eval_hand
from core.logger import DecisionLogger


# --- locate and import the sibling Poker-Engine repo ------------------------

def _bootstrap_poker_engine() -> str:
    """Put the Poker-Engine repo on sys.path and return its root."""
    candidates = []
    override = os.environ.get("THAI_POKER_ENGINE_PATH")
    if override:
        candidates.append(override)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.normpath(os.path.join(here, "..", "..", "Poker-Engine")))
    for path in candidates:
        if os.path.isdir(os.path.join(path, "engine")):
            if path not in sys.path:
                sys.path.insert(0, path)
            return path
    raise ImportError(
        "Poker-Engine repo not found. Set THAI_POKER_ENGINE_PATH or place it "
        "at ../Poker-Engine relative to this repository."
    )


_PE_PATH = _bootstrap_poker_engine()

from config import Config as _PEConfig  # noqa: E402
from engine.betting import Action as _PEAction  # noqa: E402
from engine.cards import Card as _PECard  # noqa: E402
from engine.events import EventWriter as _PEEventWriter  # noqa: E402
from engine.hand import play_hand as _pe_play_hand  # noqa: E402


# The adapter discards Poker-Engine's per-street equity field entirely, so we
# request 0 samples, which skips the Monte Carlo (guarded in engine/hand.py).
ADAPTER_EQUITY_SIMS = 0

_POSITION_TAGS = ["BTN", "SB", "BB", "UTG", "UTG+1", "MP", "LJ", "HJ", "CO"]


# --- card conversion (legacy (rank_str, suit_str) <-> PE Card(int, int)) ----
# Both engines share the "cdhs" suit order and the same rank chars, so the
# conversions are pure lookups.
_SUIT_TO_IDX = {ch: i for i, ch in enumerate(SUITS)}


def _pe_to_legacy(card) -> tuple:
    return (RANKS[card.rank - 2], SUITS[card.suit])


def _legacy_to_pe(card):
    rank_str, suit_str = card
    return _PECard(RANKS.index(rank_str) + 2, _SUIT_TO_IDX[suit_str])


def _rank_fn(hole_cards_pe, board_pe) -> int:
    """Showdown ranking delegated to the legacy evaluator (higher = better)."""
    hole = [_pe_to_legacy(c) for c in hole_cards_pe]
    board = [_pe_to_legacy(c) for c in board_pe]
    return eval_hand(hole, board)


def _positions(n: int) -> List[str]:
    """Legacy position labels in ring order (matches Table._positions)."""
    if n == 2:
        return ["BTN", "BB"]
    return _POSITION_TAGS[:n]


def _build_legacy_legal(to_call, current_bet, me_stack, bb, lfrs,
                        max_total, can_raise) -> List[Dict[str, Any]]:
    """Reconstruct the legacy legal-action dict list from Poker-Engine state.

    Mirrors core/engine.py:634-691 exactly, including the short-all-in row
    ``{"all_in": True, "reopens": False}`` that downstream bots read.
    """
    legal: List[Dict[str, Any]] = []
    if to_call == 0:
        legal.append({"type": "check"})
        if current_bet == 0:
            if me_stack > 0:
                min_bet = min(bb, me_stack)
                max_bet = me_stack
                if max_bet >= min_bet:
                    legal.append({"type": "bet", "min": min_bet, "max": max_bet})
        else:
            if can_raise and me_stack > 0:
                min_total = max(current_bet + lfrs, current_bet + bb)
                if max_total >= min_total:
                    legal.append({"type": "raise", "min": min_total, "max": max_total})
                elif max_total > current_bet:
                    legal.append({"type": "raise", "min": max_total, "max": max_total,
                                  "all_in": True, "reopens": False})
    else:
        legal.append({"type": "fold"})
        if min(me_stack, to_call) > 0:
            legal.append({"type": "call"})
        if can_raise and me_stack > to_call:
            min_total = max(current_bet + lfrs, current_bet + bb)
            if max_total >= min_total:
                legal.append({"type": "raise", "min": min_total, "max": max_total})
            elif max_total > current_bet:
                legal.append({"type": "raise", "min": max_total, "max": max_total,
                              "all_in": True, "reopens": False})
    return legal


def _sanitize(raw, legal):
    """Mirror the legacy engine's action sanitization (core/engine.py:744-771):
    illegal type -> call/check/fold; bet/raise amount clamped to [min, max]."""
    if isinstance(raw, dict):
        action_type = raw.get("type")
        action_amt = raw.get("amount")
    else:
        action_type = getattr(raw, "type", None)
        action_amt = getattr(raw, "amount", None)

    legal_types = {a["type"] for a in legal}
    if action_type not in legal_types:
        if "call" in legal_types:
            action_type, action_amt = "call", None
        elif "check" in legal_types:
            action_type, action_amt = "check", None
        else:
            action_type, action_amt = "fold", None

    if action_type in ("bet", "raise"):
        spec = next(a for a in legal if a["type"] == action_type)
        lo, hi = spec["min"], spec["max"]
        if action_amt is None:
            action_amt = lo
        amt = int(action_amt)
        if amt < lo:
            amt = lo
        if amt > hi:
            amt = hi
        action_amt = amt
    else:
        action_amt = None
    return action_type, action_amt


class _HandContext:
    """Per-hand shared state the wrapped agents read/write."""

    def __init__(self):
        self.legacy_history: List[Dict[str, Any]] = []
        self.pos_by_pid: Dict[str, str] = {}
        self.seat_to_pid: Dict[int, str] = {}
        self.hand_id: Optional[int] = None
        self.blind_increase_every: Optional[int] = None
        self.logger: Optional[DecisionLogger] = None


class _PEAgent:
    """Wraps a legacy bot so Poker-Engine can drive it."""

    def __init__(self, ctx: _HandContext, bot):
        self.ctx = ctx
        self.bot = bot
        self.rng = random.Random()  # Poker-Engine may seed this; unused here.

    # Poker-Engine lifecycle hooks (no-ops for the adapter).
    def on_action(self, event) -> None:
        pass

    def on_hand_end(self, result) -> None:
        pass

    def reset(self) -> None:
        pass

    def act(self, obs):
        ctx = self.ctx
        my_seat = obs.seat
        pid = ctx.seat_to_pid[my_seat]

        me_sc = obs.street_committed[my_seat]
        me_stack = obs.stacks[my_seat]
        to_call = obs.to_call
        current_bet = to_call + me_sc

        # Observation.to_call is the raw current_bet minus our commitment, but
        # Poker-Engine's legality runs it through _match_target (betting.py):
        # nobody can be forced to match a bet no live opponent can cover, so a
        # nominal short-all-in blind shrinks the price — possibly to 0, where
        # PE allows only check, never call. Mirror that cap here so the legacy
        # legal list never offers a call PE would reject.
        reach = [obs.street_committed[s] + obs.stacks[s]
                 for s in obs.seats if s != my_seat and not obs.folded[s]]
        if reach:
            to_call = max(0, min(current_bet, max(max(reach), me_sc)) - me_sc)
        else:
            to_call = 0
        bb = obs.blind_level["big_blind"]
        lfrs = obs.last_full_raise_size
        max_total = me_sc + me_stack
        can_raise = obs.legal_actions.can_raise

        legal = _build_legacy_legal(to_call, current_bet, me_stack, bb, lfrs,
                                    max_total, can_raise)

        if to_call == 0:
            pv_min_raise, pv_max_raise = bb, me_stack
        else:
            pv_min_raise = max(0, (current_bet + lfrs) - me_sc)
            pv_max_raise = me_stack

        # Seat ORDER is part of the legacy PlayerView contract: stacks,
        # seat_indices and opponents are listed dealer-first (the ring), so
        # downstream code reads the blinds off indices 1/2 (0/1 heads-up)
        # and hero's dict index carries hero's position label. ctx.seat_to_pid
        # is that ring; obs.stacks / obs.seats are in ABSOLUTE seat order
        # (engine/hand.py sorts players by seat number), which put the button
        # at index 0 only when the dealer was the lowest active seat
        # (sanity_engine_parity pins the order on both engines).
        ring = list(ctx.seat_to_pid)
        stacks_view = {ctx.seat_to_pid[s]: obs.stacks[s] for s in ring}
        seat_indices = {ctx.seat_to_pid[s]: s for s in ring}
        opponents = [ctx.seat_to_pid[s] for s in ring
                     if s != my_seat and not obs.folded[s]]
        acting_opponents = [op for op in opponents if stacks_view.get(op, 0) > 0]
        all_in_opponents = [op for op in opponents if stacks_view.get(op, 0) <= 0]

        view = PlayerView(
            me=pid,
            street=obs.street,
            position=ctx.pos_by_pid[pid],
            hole_cards=[_pe_to_legacy(c) for c in obs.hole_cards],
            board=[_pe_to_legacy(c) for c in obs.board],
            pot=obs.pot_total,
            to_call=to_call,
            min_raise=pv_min_raise,
            max_raise=pv_max_raise,
            legal_actions=legal,
            stacks=stacks_view,
            opponents=opponents,
            history=list(ctx.legacy_history),
            hand_id=ctx.hand_id,
            seat_indices=seat_indices,
            acting_opponents=acting_opponents,
            all_in_opponents=all_in_opponents,
            blind_increase_every=ctx.blind_increase_every,
        )

        raw = self.bot.act(view)
        atype, aamt = _sanitize(raw, legal)

        # Record the legacy-format history entry (blinds/antes are never
        # recorded — see module docstring). Calls record the chips actually
        # paid; bet/raise record the target total; fold/check record None.
        if atype == "call":
            hist_amt = min(me_stack, to_call)
        elif atype in ("bet", "raise"):
            hist_amt = aamt
        else:
            hist_amt = None
        ctx.legacy_history.append({
            "street": obs.street,
            "pid": pid,
            "type": atype,
            "amount": hist_amt,
            "to_call_before": to_call,
            "pot_before": obs.pot_total,
        })

        if ctx.logger is not None:
            ctx.logger.log_decision({
                "player": pid,
                "position": view.position,
                "street": obs.street,
                "hole": view.hole_cards,
                "board": view.board,
                "pot": view.pot,
                "to_call": to_call,
                "legal": view.legal_actions,
                "chosen_action": {"type": atype, "amount": aamt},
                "stacks": view.stacks,
                "opponents": view.opponents,
                "acting_opponents": view.acting_opponents,
                "all_in_opponents": view.all_in_opponents,
                "seat_indices": view.seat_indices,
                "folded": False,
                "hand_id": ctx.hand_id,
            })

        if atype == "fold":
            return _PEAction.fold()
        if atype == "check":
            return _PEAction.check()
        if atype == "call":
            return _PEAction.call()
        return _PEAction.raise_to(int(aamt))


class PokerEngineTable:
    """Drop-in for :class:`core.engine.Table` backed by Poker-Engine."""

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random(7331)
        self.hand_counter = 0

    def play_hand(self, seats, small_blind: int, big_blind: int,
                  dealer_index: int, bot_for, on_event=None,
                  log_decisions: bool = False,
                  logger: Optional[DecisionLogger] = None,
                  ante: int = 0,
                  blind_increase_every: Optional[int] = None,
                  preset_deck=None) -> Dict[str, int]:
        if ante < 0:
            raise ValueError("ante must be non-negative")

        owns_logger = logger is None
        if owns_logger:
            logger = DecisionLogger(enabled=log_decisions)

        hand_id = self.hand_counter
        logger.start_hand(hand_id)
        self.hand_counter += 1

        seats = [s if isinstance(s, Seat) else Seat(**s) for s in seats]
        by_pid = {s.player_id: s for s in seats}
        start_chips = {s.player_id: s.chips for s in seats}

        # Dealer-first ring of active seats (matches the legacy ring filter).
        n = len(seats)
        order = deque(range(n))
        order.rotate(-dealer_index)
        ring = [i for i in order
                if seats[i].chips > 0 and not seats[i].is_sitting_out]
        if len(ring) < 2:
            raise ValueError("Not enough active players to play a hand")

        positions = _positions(len(ring))
        pos_by_pid = {seats[idx].player_id: pos for pos, idx in zip(positions, ring)}

        # Poker-Engine seat number == absolute index in `seats` (active only),
        # so positions, blind order, and seat_indices all align by construction.
        seat_to_pid = {i: seats[i].player_id for i in ring}
        stacks_by_seat = {i: seats[i].chips for i in ring}
        names_by_seat = {i: seats[i].player_id for i in ring}
        button = ring[0]  # legacy's effective button is the first ring seat.

        ctx = _HandContext()
        ctx.pos_by_pid = pos_by_pid
        ctx.seat_to_pid = seat_to_pid
        ctx.hand_id = hand_id
        ctx.blind_increase_every = blind_increase_every
        ctx.logger = logger

        agents_by_seat = {i: _PEAgent(ctx, bot_for[seats[i].player_id]) for i in ring}

        blind_level = {
            "level": 0,
            "small_blind": small_blind,
            "big_blind": big_blind,
            "ante": 0,               # big-blind ante: unused by the adapter
            "player_ante": ante,     # legacy per-player (classic) ante
        }

        writer = _PEEventWriter(path=None)
        master_rng = random.Random(self.rng.getrandbits(64))
        violations = {i: 0 for i in ring}

        result = _pe_play_hand(
            hand_id, stacks_by_seat, names_by_seat, agents_by_seat,
            button, blind_level, _PEConfig(equity_sims=ADAPTER_EQUITY_SIMS),
            writer, master_rng, violations,
            preset_deck=preset_deck, rank_fn=_rank_fn, legacy_compat=True,
        )

        stacks_after = result["stacks_after"]
        for i in ring:
            seats[i].chips = stacks_after[i]

        net = {pid: by_pid[pid].chips - start_chips.get(pid, by_pid[pid].chips)
               for pid in start_chips}

        for pid, delta in net.items():
            logger.log_result(pid, delta)
        logger.flush()
        if owns_logger:
            logger.close()

        return net
# Engine selection lives in core.engine_factory.make_table, which imports this
# adapter lazily so the default legacy path never pulls in Poker-Engine.
