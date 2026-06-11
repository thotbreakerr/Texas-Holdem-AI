"""
Monte Carlo Counterfactual Regret Minimisation (MCCFR) Bot
----------------------------------------------------------
A simplified MCCFR agent for No-Limit Texas Hold'em that reduces
counterfactual regret over sampled trajectories. NOTE on guarantees:
with six players, card/bet/position/SPR abstraction, decision-rooted
traversals, and the engineering approximations below, this is a
best-effort regret minimiser — NOT a provably Nash-convergent solver
(CFR's equilibrium guarantee holds for two-player zero-sum games
without these approximations).

Key design choices:
  * Bet abstraction – six sizing buckets: 33/50/67/75/100% pot + all-in.
  * Card abstraction – preflop hand-strength tiers (10 buckets) and
    postflop hand-strength percentile bins (10 buckets).
  * Position abstraction – 4 positional buckets (early/middle/late/blinds).
  * SPR abstraction – 3 stack-to-pot ratio buckets (low/mid/high).
  * External-sampling MCCFR with regret-matching (vanilla, per Lanctot
    et al. 2009 — NOT CFR+/MCCFR+: no regret flooring, no linear
    averaging; see TRAINING_PLAN.md Phase-3 note 2026-06-10).
  * Strategy profile + cumulative regret tables persist across hands
    within a session and can be serialised to disk.
"""
from __future__ import annotations

import math
import os
import pickle
import random
import time as _time
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from core.bot_api import Action, PlayerView, acting_opponents_for
from core.engine import eval_hand, _FULL_DECK, EVAL_HAND_MAX
from core.equity import equity as _canonical_equity
from core.equity import equity_bucket as _canonical_equity_bucket
from core.action_history import ActionEvent, tokenize as _canonical_tokenize
from core.action_history import sizing_token as _sizing_token
from core.aivat import Snapshot as _Snapshot, value as _aivat_value
from core.opponent_stats import OpponentStatTracker as _OpponentStatTracker
from core.table_order import street_action_order as _shared_street_action_order

# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

RANKS = "23456789TJQKA"
RANK_TO_INT = {r: i for i, r in enumerate(RANKS)}

# Abstract action labels (indices into strategy / regret vectors)
ABSTRACT_ACTIONS: List[str] = [
    "fold",
    "check_call",     # check when no bet, call when facing a bet
    "bet_33",         # bet / raise 33% of pot
    "bet_50",         # bet / raise 50% of pot (half-pot)
    "bet_67",         # bet / raise 67% of pot
    "bet_75",         # bet / raise 75% of pot (three-quarter pot)
    "bet_100",        # bet / raise 100% of pot (pot-sized)
    "all_in",         # shove
]
NUM_ACTIONS = len(ABSTRACT_ACTIONS)

# Number of Monte Carlo rollouts for postflop hand-strength estimation
# Bumped 100→200 in Gate 2A: finer 50-buckets need tighter sims to land
# hands in the right bin.
_HS_SIMS = 200

# Number of preflop buckets (hand-strength tiers)
# Bumped 20→50 in Gate 2A. Info-set key space grows ~2.5× per dimension;
# combined with opp-stat bucket this is ~10-20× the Gate 1 size.
_PREFLOP_BUCKETS = 50
# Number of postflop buckets (hand-strength percentile ranges)
_POSTFLOP_BUCKETS = 50

# ── Tree CFR constants (Gate 2A) ──────────────────────────────────────────────
_MAX_CFR_DEPTH = 8        # covers ~2 betting rounds in 6-handed play
_DEFAULT_SEARCH_DEPTH = 3 # real-time subgame solve depth at inference

# CFR profile format version (Phase 3.1). Version 3 marks profiles trained
# with the Phase-3 semantics: canonical (realized-size) tree tokenization,
# unweighted ES regret updates, and opponent-node + traversal-root
# strategy_sum averaging. Older/unstamped profiles mix incompatible
# regret/strategy semantics and must NOT be resumed for training.
PROFILE_FORMAT_VERSION = 3


# ═══════════════════════════════════════════════════════════════════════════════
#  Position & SPR abstraction helpers
# ═══════════════════════════════════════════════════════════════════════════════

# Map engine position labels into 4 strategic buckets so CFR can learn
# different opening/calling ranges for each seat category.
_POSITION_BUCKETS = {
    # Late position: most info, widest range
    "BTN": "late", "CO": "late", "HJ": "late",
    # Middle position
    "MP": "middle", "LJ": "middle",
    # Early position: tightest range
    "UTG": "early", "UTG+1": "early", "UTG+2": "early",
    # Blinds: posted dead money, last to act preflop
    "SB": "blinds", "BB": "blinds",
}


def _engine_positions(n: int) -> List[str]:
    """Mirror core.engine.Table._positions without importing Table."""
    if n == 2:
        return ["BTN", "BB"]
    tags = ["BTN", "SB", "BB", "UTG", "UTG+1", "MP", "LJ", "HJ", "CO"]
    return tags[:n]


def _position_bucket(position: str) -> str:
    """Compress engine position label into one of 4 strategic buckets."""
    return _POSITION_BUCKETS.get(position, "middle")  # safe fallback


def _spr_bucket(hero_stack: int, pot: int, opp_stacks: list) -> str:
    """
    Classify the effective stack-to-pot ratio.

    Effective stack = min(hero, biggest active opponent) — that's the
    most chips that can actually be wagered between us.
    """
    if pot <= 0:
        # Preflop before blinds posted — treat as deep
        return "high"
    effective = hero_stack
    if opp_stacks:
        effective = min(hero_stack, max(opp_stacks))
    spr = effective / pot
    if spr < 5:    return "low"     # commitment/stack-off territory
    if spr < 15:   return "mid"     # standard play
    return "high"                   # deep, implied odds matter


def _seat_labels_for_view(view: PlayerView, pids: List[str],
                          hero_seat: int) -> Dict[int, str]:
    """Infer engine position labels from ordered PlayerView.stacks keys."""
    labels = _engine_positions(len(pids))
    if hero_seat < len(labels) and labels[hero_seat] == view.position:
        return {i: labels[i] for i in range(len(pids))}
    if view.position in labels:
        # Synthetic tests sometimes put hero at a different dict index than
        # the engine would. Rotate labels so hero's known label stays true.
        hero_label_idx = labels.index(view.position)
        offset = hero_seat - hero_label_idx
        return {
            i: labels[(i - offset) % len(labels)]
            for i in range(len(pids))
        }
    return {i: labels[i] if i < len(labels) else "MP" for i in range(len(pids))}


def _street_action_order(street: str, ring_order: List[int]) -> List[int]:
    """Engine-equivalent action order for a fresh betting street."""
    return _shared_street_action_order(street, ring_order)


def _order_after_seat(order: List[int], seat: int,
                      include_seat: bool = False) -> List[int]:
    """Return seats in order after seat, wrapping once around the table."""
    if seat not in order:
        return list(order)
    idx = order.index(seat)
    tail = order[idx + 1:] + order[:idx]
    if include_seat:
        return [seat] + tail
    return tail


def _seat_indices_for_view(view: PlayerView) -> Dict[str, int]:
    mapping = getattr(view, "seat_indices", None) or {}
    if mapping:
        return {pid: int(idx) for pid, idx in mapping.items()}
    return {pid: i for i, pid in enumerate(view.stacks.keys())}


def _stable_seat_index(view: PlayerView, pid: str,
                       pids: Optional[List[str]] = None) -> Optional[int]:
    mapping = _seat_indices_for_view(view)
    if pid in mapping:
        return mapping[pid]
    pids = pids if pids is not None else list(view.stacks.keys())
    if pid in pids:
        return pids.index(pid)
    return None


def _tracker_size_for_view(view: PlayerView) -> int:
    mapping = _seat_indices_for_view(view)
    if mapping:
        return max(mapping.values()) + 1
    return len(view.stacks)


def _infer_big_blind_from_view(view: PlayerView) -> int:
    """Best-effort BB inference from the frozen PlayerView shape."""
    to_call = int(getattr(view, "to_call", 0) or 0)
    min_raise = int(getattr(view, "min_raise", 0) or 0)
    first_pot = None
    for entry in getattr(view, "history", []) or []:
        if entry.get("street") == "preflop" and "pot_before" in entry:
            first_pot = int(entry.get("pot_before") or 0)
            break
    if first_pot and first_pot > 0:
        return max(1, int(round(first_pot * 2 / 3)))
    if getattr(view, "street", None) == "preflop":
        pot = int(getattr(view, "pot", 0) or 0)
        if pot > 0:
            return max(1, int(round(pot * 2 / 3)))
    if to_call > 0 and min_raise > to_call:
        return max(1, min_raise - to_call)
    if to_call == 0:
        return max(1, min_raise or 1)
    return max(1, min_raise or 1)


def _infer_last_raise_size_from_view(view: PlayerView, big_blind: int) -> int:
    """Infer the current street's last full raise size from public history."""
    pids = list(getattr(view, "stacks", {}) or {})
    pid_to_seat = {pid: i for i, pid in enumerate(pids)}
    contrib = [0 for _ in pids]
    bb = max(1, int(big_blind or 1))
    last_raise_size = bb
    current_street = "preflop"
    history = list(getattr(view, "history", []) or [])

    explicit_blinds = any(e.get("type") == "blind" for e in history)
    if explicit_blinds:
        for entry in history:
            if entry.get("type") != "blind":
                continue
            pid = entry.get("pid", "")
            if pid in pid_to_seat:
                contrib[pid_to_seat[pid]] += max(0, int(entry.get("amount") or 0))
    else:
        first_pot = None
        for entry in history:
            if entry.get("street") == "preflop" and "pot_before" in entry:
                first_pot = int(entry.get("pot_before") or 0)
                break
        if first_pot is None and getattr(view, "street", None) == "preflop":
            first_pot = int(getattr(view, "pot", 0) or 0)
        if first_pot and len(pids) >= 2:
            inferred_bb = max(1, int(round(first_pot * 2 / 3)))
            inferred_sb = max(0, first_pot - inferred_bb)
            if len(pids) == 2:
                sb_idx, bb_idx = 0, 1
            else:
                sb_idx, bb_idx = 1, 2 % len(pids)
            contrib[sb_idx] += inferred_sb
            contrib[bb_idx] += inferred_bb
            last_raise_size = inferred_bb

    for entry in history:
        atype = entry.get("type", "")
        if atype == "blind":
            continue
        street = entry.get("street", current_street)
        if street != current_street:
            contrib = [0 for _ in pids]
            last_raise_size = bb
            current_street = street

        pid = entry.get("pid", "")
        if pid not in pid_to_seat:
            continue
        idx = pid_to_seat[pid]
        if atype in ("check", "fold"):
            continue
        if atype == "call":
            amount = entry.get("amount")
            if amount is None:
                amount = entry.get("to_call_before", 0)
            contrib[idx] += max(0, int(amount or 0))
            continue
        if atype not in ("bet", "raise", "all_in"):
            continue

        prev_bet = max(contrib) if contrib else 0
        target = entry.get("amount")
        if target is None:
            target = contrib[idx] + int(entry.get("to_call_before", 0) or 0)
        target = max(0, int(target or 0))
        raise_size = target - prev_bet
        if raise_size > 0 and (prev_bet == 0 or raise_size >= last_raise_size):
            last_raise_size = raise_size
        contrib[idx] += max(0, target - contrib[idx])

    if current_street != getattr(view, "street", current_street):
        return bb

    legal_raise = next(
        (a for a in (getattr(view, "legal_actions", None) or [])
         if a.get("type") == "raise"),
        None,
    )
    if legal_raise is not None:
        current_bet = max(contrib) if contrib else 0
        min_total = int(legal_raise.get("min") or 0)
        if current_bet > 0 and min_total > current_bet:
            last_raise_size = max(last_raise_size, min_total - current_bet)
    return max(1, int(last_raise_size or bb))


def _reconstruct_contributions_from_view(view: PlayerView):
    """Return (street_committed, total_committed, reliable) for PlayerView.

    ``street_committed`` mirrors the engine's current-street ``contrib`` map.
    ``total_committed`` mirrors the hand-level ``total_contrib`` plus the
    current street's outstanding contributions.
    """
    pids = list(view.stacks.keys())
    n_seats = len(pids)
    pid_to_seat = {pid: i for i, pid in enumerate(pids)}
    total_committed = [0 for _ in pids]
    street_contrib = [0 for _ in pids]
    history = list(view.history or [])
    reliable = bool(pids)

    def add_to(pid, amount):
        nonlocal reliable
        if pid not in pid_to_seat:
            reliable = False
            return
        amt = max(0, int(amount or 0))
        idx = pid_to_seat[pid]
        total_committed[idx] += amt
        street_contrib[idx] += amt

    explicit_blinds = any(e.get("type") == "blind" for e in history)
    if explicit_blinds:
        for entry in history:
            if entry.get("type") == "blind":
                add_to(entry.get("pid", ""), entry.get("amount") or 0)
    else:
        first_pot = None
        for entry in history:
            if entry.get("street") == "preflop" and "pot_before" in entry:
                first_pot = int(entry.get("pot_before") or 0)
                break
        if first_pot is None and view.street == "preflop":
            first_pot = int(view.pot or 0)
        if first_pot and n_seats >= 2:
            bb = max(1, int(round(first_pot * 2 / 3)))
            sb = max(0, first_pot - bb)
            if n_seats == 2:
                sb_idx, bb_idx = 0, 1
            else:
                sb_idx, bb_idx = 1, 2 % n_seats
            total_committed[sb_idx] += sb
            street_contrib[sb_idx] += sb
            total_committed[bb_idx] += bb
            street_contrib[bb_idx] += bb

    current_street = "preflop"
    for entry in history:
        atype = entry.get("type", "")
        if atype == "blind":
            continue
        street = entry.get("street", current_street)
        if street != current_street:
            street_contrib = [0 for _ in pids]
            current_street = street

        pid = entry.get("pid", "")
        if pid not in pid_to_seat:
            reliable = False
            continue
        idx = pid_to_seat[pid]

        if atype in ("check", "fold"):
            continue
        if atype == "call":
            amt = entry.get("amount")
            if amt is None:
                amt = entry.get("to_call_before", 0)
            add_to(pid, amt)
        elif atype in ("bet", "raise", "all_in"):
            target = entry.get("amount")
            if target is None:
                target = street_contrib[idx] + int(entry.get("to_call_before", 0) or 0)
            delta = max(0, int(target or 0) - street_contrib[idx])
            add_to(pid, delta)
        else:
            reliable = False

    if current_street != view.street:
        street_contrib = [0 for _ in pids]

    view_pot = int(view.pot or 0)
    is_real = reliable and abs(sum(total_committed) - view_pot) <= 1
    return street_contrib, total_committed, is_real


# ═══════════════════════════════════════════════════════════════════════════════
#  Card abstraction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _preflop_bucket(hole: List[Tuple[str, str]]) -> int:
    """
    Bucket a preflop hand into one of ``_PREFLOP_BUCKETS`` tiers based on a
    simplified hand-strength heuristic (inspired by Sklansky–Malmuth groups).

    Returns an integer in [0, _PREFLOP_BUCKETS-1] where higher = stronger.
    """
    if len(hole) < 2:
        return 0

    r1 = RANK_TO_INT[hole[0][0]]
    r2 = RANK_TO_INT[hole[1][0]]
    high, low = max(r1, r2), min(r1, r2)
    suited = hole[0][1] == hole[1][1]
    pair = (r1 == r2)

    # Raw score: pairs get a big bonus, high cards contribute, suited/connected
    # hands get a small bump.
    score = high + low * 0.6
    if pair:
        score += 20 + high * 1.5
    if suited:
        score += 3
    gap = high - low
    if gap <= 2 and not pair:
        score += 2  # connector / one-gapper

    # Normalise ``score`` into [0, _PREFLOP_BUCKETS-1].  Empirical range of
    # ``score`` is ~[1.2  (2-3o), ~46  (AA)].
    max_score = 46.0
    bucket = int(score / max_score * (_PREFLOP_BUCKETS - 1))
    return max(0, min(_PREFLOP_BUCKETS - 1, bucket))


def _postflop_bucket(hole: List[Tuple[str, str]],
                     board: List[Tuple[str, str]],
                     n_opponents: int) -> int:
    """
    Estimate hand-strength percentile via Monte-Carlo rollout against
    ``n_opponents`` random hands, then bucket into one of
    ``_POSTFLOP_BUCKETS`` bins.

    In multiway pots hero must beat ALL opponents to "win".

    Returns an integer in [0, _POSTFLOP_BUCKETS-1] where higher = stronger.

    Delegates to core.equity.equity_bucket for the canonical implementation.
    """
    return _canonical_equity_bucket(
        hole, board, n_opponents,
        n_buckets=_POSTFLOP_BUCKETS, n_sims=_HS_SIMS,
    )


def _info_set_key(street: str, bucket: int, history_key: str,
                  n_opponents: int, position_bucket: str,
                  spr_bucket: str, opp_stat_bucket: str = "TA") -> str:
    """
    Build a compact information-set key from the street, active opponent
    count, position bucket, SPR bucket, opponent-stat bucket, card bucket,
    and abstracted action history.

    Gate 2A format: 7 fields separated by 6 colons.
    Pre-Gate-2A format had 6 fields / 5 colons — those keys are rejected
    at load time.
    """
    return (f"{street}:{n_opponents}:{position_bucket}"
            f":{spr_bucket}:{opp_stat_bucket}:{bucket}:{history_key}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Action mapping: abstract ↔ concrete
# ═══════════════════════════════════════════════════════════════════════════════

def _current_bet_from_view(view: PlayerView, to_call: int) -> int:
    """Highest street contribution at this decision (the engine's current_bet).

    Mirrors the ``_GameState`` path, where ``current_bet == max(committed_per
    _seat)``.  Reconstructed from the view's history; falls back to ``to_call``
    (i.e. assume the hero has nothing committed yet) when reconstruction is
    unreliable.
    """
    try:
        street_committed, _total, _real = _reconstruct_contributions_from_view(view)
        if not street_committed:
            return int(to_call or 0)
        pids = list(view.stacks.keys())
        my_committed = 0
        if view.me in pids:
            my_committed = street_committed[pids.index(view.me)]
        # current_bet is the table-wide max; it must also cover hero's own
        # committed plus what they still owe.
        return max(max(street_committed), my_committed + int(to_call or 0))
    except Exception:
        return int(to_call or 0)


def _raise_sizing_target(frac: float, pot: int, to_call: int,
                         current_bet: int, is_raise: bool) -> int:
    """Translate a pot-fraction sizing bucket into an engine *target total*.

    The engine interprets a bet/raise amount as the actor's TOTAL street
    contribution.  A pot-fraction is only a total for an opening bet
    (``current_bet == 0``); when facing a bet, a pot-relative *raise* means
    "call, then raise by a fraction of the post-call pot":

        target_total = current_bet + frac * (pot + to_call)

    With the defaults (to_call=0, current_bet=0) this reduces to ``frac*pot``,
    preserving the opening-bet behaviour for callers that do not supply the
    betting context.
    """
    if is_raise:
        return current_bet + int(round(frac * (pot + to_call)))
    return int(pot * frac)


def _legal_abstract_actions(legal: List[Dict[str, Any]],
                            pot: int, to_call: int = 0,
                            current_bet: int = 0) -> List[int]:
    """
    Map the engine's concrete legal actions to abstract action indices.

    Returns a list of indices into ``ABSTRACT_ACTIONS`` that are available.
    ``to_call``/``current_bet`` describe the betting context so raise sizing
    buckets are spaced as true pot-relative raises rather than collapsing onto
    the min-raise when facing a bet.
    """
    types = {a["type"] for a in legal}
    result: List[int] = []

    # fold / check_call are always available when their concrete counterparts are
    if "fold" in types:
        result.append(0)  # fold
    if "check" in types or "call" in types:
        result.append(1)  # check_call

    # bet / raise sizing buckets
    has_bet_raise = "bet" in types or "raise" in types
    if has_bet_raise:
        spec = next(a for a in legal if a["type"] in ("bet", "raise"))
        lo, hi = spec["min"], spec["max"]
        if spec.get("all_in") or lo == hi:
            result.append(7)  # all_in
            return sorted(set(result))

        # Generate the six sizing targets (indices match ABSTRACT_ACTIONS)
        is_raise = spec["type"] == "raise"
        sizes = {
            2: _raise_sizing_target(0.33, pot, to_call, current_bet, is_raise),
            3: _raise_sizing_target(0.50, pot, to_call, current_bet, is_raise),
            4: _raise_sizing_target(0.67, pot, to_call, current_bet, is_raise),
            5: _raise_sizing_target(0.75, pot, to_call, current_bet, is_raise),
            6: _raise_sizing_target(1.00, pot, to_call, current_bet, is_raise),
            7: hi,                # all_in
        }

        seen_amts = set()
        for idx, target in sizes.items():
            # Clamp target into [lo, hi], then skip duplicate concrete bets.
            clamped = max(lo, min(hi, target))
            if clamped in seen_amts:
                continue
            seen_amts.add(clamped)
            result.append(idx)

    return sorted(set(result)) if result else [1]  # fallback: check/call


def _abstract_to_concrete(abstract_idx: int,
                          legal: List[Dict[str, Any]],
                          pot: int, to_call: int = 0,
                          current_bet: int = 0) -> Action:
    """
    Convert an abstract action index back into a concrete ``Action`` the
    engine accepts.  ``to_call``/``current_bet`` give the betting context so a
    bet/raise bucket maps to the correct engine *target total* (see
    :func:`_raise_sizing_target`).
    """
    types = {a["type"] for a in legal}
    label = ABSTRACT_ACTIONS[abstract_idx]

    if label == "fold":
        if "fold" in types:
            return Action("fold")
        # Not allowed to fold → check/call
        return _fallback_passive(legal)

    if label == "check_call":
        if "check" in types:
            return Action("check")
        if "call" in types:
            return Action("call")
        return _fallback_passive(legal)

    # Sizing actions
    frac_map = {
        "bet_33": 0.33, "bet_50": 0.50, "bet_67": 0.67,
        "bet_75": 0.75, "bet_100": 1.00, "all_in": None,
    }
    frac = frac_map.get(label)

    bet_raise = [a for a in legal if a["type"] in ("bet", "raise")]
    if not bet_raise:
        return _fallback_passive(legal)

    spec = bet_raise[0]
    lo, hi = spec["min"], spec["max"]

    if frac is None:
        # all-in
        amt = hi
    else:
        amt = _raise_sizing_target(frac, pot, to_call, current_bet,
                                   spec["type"] == "raise")

    amt = max(lo, min(hi, amt))
    return Action(spec["type"], amt)


def _fallback_passive(legal: List[Dict[str, Any]]) -> Action:
    """Fallback: check > call > fold."""
    for t in ("check", "call", "fold"):
        if any(a["type"] == t for a in legal):
            return Action(t)
    # absolute last resort
    a = legal[0]
    return Action(a["type"], a.get("min"))


# ═══════════════════════════════════════════════════════════════════════════════
#  History abstraction
# ═══════════════════════════════════════════════════════════════════════════════

def _abstract_history(history: List[Dict[str, Any]], pot: int) -> str:
    """
    Compress the engine action history into a compact string of abstract
    action labels suitable for use as an information-set key suffix.

    Delegates to core.action_history.tokenize for the canonical
    implementation. Constructs ActionEvent objects from raw history
    dicts, using pot_before from the engine (falls back to current pot
    for old-format histories).

    Tokens:
      F = fold, K = check, C = call,
      S = small (~33%), Q = quarter-pot-ish (~50%),
      M = medium (~67%), L = large (~75%),
      P = pot-sized (~100%), A = all-in / over-pot
    """
    events = []
    for entry in history:
        atype = entry.get("type", "")
        amt = entry.get("amount") or 0
        ref_pot = entry.get("pot_before", pot)
        events.append(ActionEvent(
            seat=0,  # seat not used by tokenizer
            street=entry.get("street", "preflop"),
            action=atype if atype else "check",
            amount=int(amt),
            pot_before=int(ref_pot),
        ))
    return _canonical_tokenize(events)


# ═══════════════════════════════════════════════════════════════════════════════
#  Lightweight GameState for recursive tree CFR (Gate 2A)
# ═══════════════════════════════════════════════════════════════════════════════

_STREET_ORDER = ["preflop", "flop", "turn", "river"]
_STREET_CARDS = {"preflop": 0, "flop": 3, "turn": 1, "river": 1}


class _GameState:
    """Lightweight game state for recursive CFR traversal.

    Does NOT reuse the engine's Table — too heavyweight. Owns: pot, stacks,
    committed_per_seat, alive, street, board, hole_cards (full info during
    training, hero-only during inference), seat_order (who acts next),
    history (list of abstract action indices).
    """
    __slots__ = (
        "pot", "stacks", "committed_per_seat", "total_committed_per_seat",
        "alive", "street", "board", "hole_cards", "seat_order", "action_idx",
        "history_tokens", "deck_remaining", "n_seats", "hero_seat",
        "real_contributions", "ring_order", "position_labels",
        "opp_stat_bucket", "big_blind", "street_actions",
        "last_raise_size", "raise_blocked", "acted",
    )

    def __init__(self, *, pot, stacks, committed_per_seat, alive, street,
                 board, hole_cards, seat_order, action_idx=0,
                 history_tokens="", deck_remaining=None, hero_seat=0,
                 real_contributions=False, ring_order=None,
                 position_labels=None, opp_stat_bucket="TA", big_blind=10,
                 street_actions=0, total_committed_per_seat=None,
                 last_raise_size=None, raise_blocked=None, acted=None):
        self.pot = pot
        self.stacks = list(stacks)
        self.committed_per_seat = list(committed_per_seat)
        # Cumulative per-seat contributions across all streets (mirrors
        # engine.py's total_contrib). Used by AIVAT for side-pot settlement.
        # If not provided, seed from the (possibly cumulative) initial
        # committed_per_seat so the field is well-defined from the start.
        self.total_committed_per_seat = (
            list(total_committed_per_seat)
            if total_committed_per_seat is not None
            else list(committed_per_seat)
        )
        self.alive = list(alive)
        self.street = street
        self.board = list(board)
        self.hole_cards = dict(hole_cards)  # seat -> (card, card)
        self.seat_order = list(seat_order)  # seats in action order
        self.action_idx = action_idx        # index into seat_order
        self.history_tokens = history_tokens
        self.deck_remaining = list(deck_remaining) if deck_remaining else []
        self.n_seats = len(stacks)
        self.hero_seat = hero_seat
        self.real_contributions = real_contributions
        self.ring_order = list(ring_order) if ring_order else list(range(self.n_seats))
        self.position_labels = dict(position_labels) if position_labels else {}
        self.opp_stat_bucket = opp_stat_bucket
        self.big_blind = max(1, int(big_blind or 1))
        self.street_actions = int(street_actions)
        self.last_raise_size = max(
            1,
            int(last_raise_size if last_raise_size is not None else self.big_blind),
        )
        self.raise_blocked = set(raise_blocked or [])
        if acted is None:
            acted = self.seat_order[:max(0, min(self.action_idx, len(self.seat_order)))]
        self.acted = set(acted)

    def copy(self):
        return _GameState(
            pot=self.pot,
            stacks=list(self.stacks),
            committed_per_seat=list(self.committed_per_seat),
            total_committed_per_seat=list(self.total_committed_per_seat),
            alive=list(self.alive),
            street=self.street,
            board=list(self.board),
            hole_cards=dict(self.hole_cards),
            seat_order=list(self.seat_order),
            action_idx=self.action_idx,
            history_tokens=self.history_tokens,
            deck_remaining=list(self.deck_remaining),
            hero_seat=self.hero_seat,
            real_contributions=self.real_contributions,
            ring_order=list(self.ring_order),
            position_labels=dict(self.position_labels),
            opp_stat_bucket=self.opp_stat_bucket,
            big_blind=self.big_blind,
            street_actions=self.street_actions,
            last_raise_size=self.last_raise_size,
            raise_blocked=set(self.raise_blocked),
            acted=set(self.acted),
        )

    def is_terminal(self):
        """True if only 0-1 alive players remain, or street is past river."""
        alive_count = sum(1 for a in self.alive if a)
        if alive_count <= 1:
            return True
        if self.street not in _STREET_ORDER:
            return True
        # Past river = terminal
        if self.street == "river" and self.action_idx >= len(self.seat_order):
            return True
        # Soft cap to bound CFR tree depth per street.
        if self.street_actions >= 50:
            return True
        return False

    def is_chance_node(self):
        """True if we need to deal the next street's cards."""
        if self.is_terminal():
            return False
        # If we've cycled through all actors for this street, advance
        return self.action_idx >= len(self.seat_order)

    def seat_to_act(self):
        """Return the seat index of the next player to act."""
        if self.action_idx < len(self.seat_order):
            return self.seat_order[self.action_idx]
        return -1

    def legal_abstract_actions(self):
        """Return list of abstract action indices legal for seat_to_act."""
        seat = self.seat_to_act()
        if seat < 0 or not self.alive[seat]:
            return _legal_abstract_actions(self.legal_actions(), self.pot)
        max_contrib = max(self.committed_per_seat[s]
                          for s in range(self.n_seats) if self.alive[s])
        to_call = max(0, max_contrib - self.committed_per_seat[seat])
        return _legal_abstract_actions(self.legal_actions(), self.pot,
                                       to_call=to_call, current_bet=max_contrib)

    def _active_seats(self):
        return [
            i for i in self.ring_order
            if self.alive[i] and self.stacks[i] > 0
        ]

    def _ordered_after(self, seat, candidates):
        candidates = list(candidates)
        if not candidates:
            return []
        ring = list(self.ring_order)
        if seat not in ring:
            return candidates
        seat_pos = ring.index(seat)
        order = ring[seat_pos + 1:] + ring[:seat_pos]
        allowed = set(candidates)
        return [i for i in order if i in allowed]

    def legal_actions(self):
        """Build engine-shaped legal actions from this lightweight state."""
        seat = self.seat_to_act()
        if seat < 0 or not self.alive[seat]:
            return [{"type": "check"}]

        max_contrib = max(self.committed_per_seat[s]
                          for s in range(self.n_seats) if self.alive[s])
        to_call = max_contrib - self.committed_per_seat[seat]
        chips = self.stacks[seat]
        if chips <= 0:
            return [{"type": "check"}]
        legal = []
        can_raise = seat not in self.raise_blocked

        if to_call > 0:
            legal.append({"type": "fold"})
            if chips > 0:
                legal.append({"type": "call"})
            if can_raise and chips > to_call:
                max_total = chips + self.committed_per_seat[seat]
                min_total = max_contrib + self.last_raise_size
                min_total = max(min_total, max_contrib + self.big_blind)
                if max_total >= min_total:
                    legal.append({
                        "type": "raise",
                        "min": min_total,
                        "max": max_total,
                    })
                elif max_total > max_contrib:
                    legal.append({
                        "type": "raise",
                        "min": max_total,
                        "max": max_total,
                        "all_in": True,
                        "reopens": False,
                    })
        else:
            legal.append({"type": "check"})
            if max_contrib == 0:
                min_bet = min(self.big_blind, chips)
                if chips >= min_bet:
                    legal.append({"type": "bet", "min": min_bet, "max": chips})
            else:
                if can_raise:
                    max_total = chips + self.committed_per_seat[seat]
                    min_total = max_contrib + self.last_raise_size
                    min_total = max(min_total, max_contrib + self.big_blind)
                    if max_total >= min_total:
                        legal.append({
                            "type": "raise",
                            "min": min_total,
                            "max": max_total,
                        })
                    elif max_total > max_contrib:
                        legal.append({
                            "type": "raise",
                            "min": max_total,
                            "max": max_total,
                            "all_in": True,
                            "reopens": False,
                        })
        return legal

    def apply_action(self, seat, abstract_idx):
        """Return a new _GameState after seat takes abstract_idx action."""
        s = self.copy()
        label = ABSTRACT_ACTIONS[abstract_idx]

        max_contrib = max(s.committed_per_seat[i]
                          for i in range(s.n_seats) if s.alive[i])
        to_call = max_contrib - s.committed_per_seat[seat]
        prev_bet = max_contrib
        prev_last_raise_size = s.last_raise_size
        acted_before = set(s.acted)

        reopens_action = False
        raise_size = 0
        if label == "fold":
            s.alive[seat] = False
        elif label == "check_call":
            cost = min(to_call, s.stacks[seat])
            s.stacks[seat] -= cost
            s.committed_per_seat[seat] += cost
            s.total_committed_per_seat[seat] += cost
            s.pot += cost
        else:
            # Bet/raise sizing
            frac_map = {
                "bet_33": 0.33, "bet_50": 0.50, "bet_67": 0.67,
                "bet_75": 0.75, "bet_100": 1.00, "all_in": None,
            }
            frac = frac_map.get(label, 0.5)
            bet_raise = [a for a in s.legal_actions()
                         if a["type"] in ("bet", "raise")]
            if not bet_raise:
                return s
            spec = bet_raise[0]
            lo, hi = spec["min"], spec["max"]
            reopens_action = bool(spec.get("reopens", True))
            if frac is None:
                bet_total = hi
            else:
                bet_total = _raise_sizing_target(
                    frac, s.pot, to_call, max_contrib,
                    spec["type"] == "raise")
                bet_total = max(lo, min(hi, bet_total))
            need = bet_total - s.committed_per_seat[seat]
            need = max(0, min(need, s.stacks[seat]))
            s.stacks[seat] -= need
            s.committed_per_seat[seat] += need
            s.total_committed_per_seat[seat] += need
            s.pot += need
            new_bet = max(
                s.committed_per_seat[i]
                for i in range(s.n_seats) if s.alive[i]
            )
            raise_size = max(0, new_bet - prev_bet)

        # Token for history.
        # Sizing actions tokenize from the REALIZED total and the pot
        # before the action — the same bucketing live play uses
        # (core.action_history.sizing_token). A fixed per-bucket token
        # diverged from live tokenization for raises, min-raise clamps,
        # and all-in clamps, orphaning tree-trained info sets.
        if abstract_idx >= 2:
            # Engine `amount` semantics: actor's TOTAL street contribution.
            realized_total = s.committed_per_seat[seat]
            token = _sizing_token(realized_total, self.pot)
        elif abstract_idx == 0:
            token = "F"
        else:
            token = "K" if to_call == 0 else "C"
        s.history_tokens += token
        s.street_actions += 1

        full_raise = (
            abstract_idx >= 2
            and raise_size > 0
            and reopens_action
            and (prev_bet == 0 or raise_size >= prev_last_raise_size)
        )
        # Soft cap on tree depth per street; engine has its own 500-iteration
        # safety counter at engine.py:547 — CFR uses 50 to keep recursion bounded.
        if full_raise and s.street_actions < 50:
            s.last_raise_size = raise_size
            s.raise_blocked.clear()
            s.acted = {seat}
            new_bet = max(s.committed_per_seat)
            responders = [
                i for i in s._active_seats()
                if i != seat and s.committed_per_seat[i] < new_bet
            ]
            s.seat_order = s._ordered_after(seat, responders)
            s.action_idx = 0
        elif abstract_idx >= 2 and raise_size > 0 and s.street_actions < 50:
            s.raise_blocked.update(acted_before)
            s.acted = acted_before | {seat}
            new_bet = max(s.committed_per_seat)
            responders = [
                i for i in s._active_seats()
                if i != seat and s.committed_per_seat[i] < new_bet
            ]
            s.seat_order = s._ordered_after(seat, responders)
            s.action_idx = 0
        elif seat == self.seat_to_act():
            s.acted.add(seat)
            s.action_idx += 1

        return s

    def advance_street(self):
        """Deal next street's cards via MC sample. Returns new _GameState."""
        s = self.copy()
        si = _STREET_ORDER.index(s.street)
        if si + 1 >= len(_STREET_ORDER):
            return s  # past river — terminal

        next_street = _STREET_ORDER[si + 1]
        n_cards = _STREET_CARDS[next_street]

        if n_cards > 0 and len(s.deck_remaining) >= n_cards:
            dealt = random.sample(s.deck_remaining, n_cards)
            s.board.extend(dealt)
            s.deck_remaining = [c for c in s.deck_remaining if c not in dealt]

        s.street = next_street
        # Reset action order to the engine's next-street order.
        s.seat_order = [
            seat for seat in _street_action_order(next_street, s.ring_order)
            if s.alive[seat] and s.stacks[seat] > 0
        ]
        s.action_idx = 0
        s.street_actions = 0
        s.last_raise_size = s.big_blind
        s.raise_blocked.clear()
        s.acted.clear()
        # Mirror engine.py L430: reset per-seat contributions for the new
        # street so preflop blinds/bets do not leak into postflop to_call.
        # total_committed_per_seat is NOT reset — it tracks cumulative
        # contributions across all streets (mirrors engine's total_contrib),
        # needed by AIVAT for side-pot settlement at leaves.
        s.committed_per_seat = [0] * s.n_seats
        return s

    def to_call_for(self, seat):
        if not self.alive[seat]:
            return 0
        max_contrib = max(self.committed_per_seat[i]
                          for i in range(self.n_seats) if self.alive[i])
        return max(0, max_contrib - self.committed_per_seat[seat])


def _build_game_state_from_view(
    view: PlayerView,
    full_hole_cards=None,
    opp_stat_bucket: str = "TA",
):
    """Construct a _GameState from a PlayerView for inference-time search.

    full_hole_cards: if provided, dict {seat_idx: (card, card)} for all seats.
    During inference we only know hero's cards; opponents get None.
    """
    stacks_dict = view.stacks
    pids = list(stacks_dict.keys())
    n_seats = len(pids)
    hero_pid = view.me
    hero_seat = pids.index(hero_pid) if hero_pid in pids else 0

    stacks = [stacks_dict[pid] for pid in pids]
    alive = [stacks[i] >= 0 for i, _pid in enumerate(pids)]
    opp_set = set(view.opponents) if view.opponents else set()
    for i, pid in enumerate(pids):
        if pid != hero_pid and pid not in opp_set:
            alive[i] = False

    big_blind = _infer_big_blind_from_view(view)
    last_raise_size = _infer_last_raise_size_from_view(view, big_blind)

    # Prefer reconstructed current-street contributions when history is enough,
    # then force the public PlayerView.to_call contract to be true at the root.
    committed, total_committed, real = _reconstruct_contributions_from_view(view)
    if not committed or len(committed) != n_seats:
        committed = [0] * n_seats
        total_committed = [0] * n_seats
        real = False

    active_seats = [i for i, ok in enumerate(alive) if ok]
    desired_to_call = max(0, int(view.to_call or 0))
    current_bet = max((committed[i] for i in active_seats), default=0)
    legal_raise = next(
        (a for a in (view.legal_actions or []) if a.get("type") == "raise"),
        None,
    )
    if desired_to_call > 0:
        if current_bet < desired_to_call:
            current_bet = desired_to_call
        committed[hero_seat] = max(0, current_bet - desired_to_call)
        if max((committed[i] for i in active_seats if i != hero_seat), default=0) < current_bet:
            for i in active_seats:
                if i != hero_seat:
                    committed[i] = current_bet
                    break
        if legal_raise is not None:
            min_total = int(legal_raise.get("min") or 0)
            if min_total > current_bet:
                last_raise_size = max(1, min_total - current_bet)
    elif current_bet == 0:
        legal_types = {a.get("type") for a in view.legal_actions or []}
        if "raise" in legal_types and "bet" not in legal_types:
            min_total = int(legal_raise.get("min") or 0) if legal_raise else 0
            current_bet = max(
                1,
                (min_total - last_raise_size) if min_total else int(view.min_raise or 0) - big_blind,
            )
            for i in active_seats:
                committed[i] = current_bet
    elif legal_raise is not None:
        min_total = int(legal_raise.get("min") or 0)
        if min_total > current_bet:
            last_raise_size = max(1, min_total - current_bet)

    if not real:
        total_committed = list(committed)

    hole_cards = {}
    if full_hole_cards:
        hole_cards = dict(full_hole_cards)
    else:
        hole_cards[hero_seat] = tuple(view.hole_cards)

    used = set()
    for cards in hole_cards.values():
        if cards:
            for c in cards:
                used.add(tuple(c))
    for c in view.board:
        used.add(tuple(c))
    deck_remaining = [c for c in _FULL_DECK if c not in used]

    ring_order = list(range(n_seats))
    position_labels = _seat_labels_for_view(view, pids, hero_seat)
    full_order = [
        seat for seat in _street_action_order(view.street, ring_order)
        if alive[seat] and stacks[seat] > 0
    ]
    seat_order = _order_after_seat(full_order, hero_seat, include_seat=True)

    return _GameState(
        pot=view.pot,
        stacks=stacks,
        committed_per_seat=committed,
        total_committed_per_seat=total_committed,
        alive=alive,
        street=view.street,
        board=list(view.board),
        hole_cards=hole_cards,
        seat_order=seat_order,
        action_idx=0,
        history_tokens=_abstract_history(view.history or [], view.pot),
        deck_remaining=deck_remaining,
        hero_seat=hero_seat,
        real_contributions=False,
        ring_order=ring_order,
        position_labels=position_labels,
        opp_stat_bucket=opp_stat_bucket,
        big_blind=big_blind,
        last_raise_size=last_raise_size,
    ), hero_seat


# ═══════════════════════════════════════════════════════════════════════════════
#  CFR Node
# ═══════════════════════════════════════════════════════════════════════════════

class _CFRNode:
    """
    Stores cumulative regret and cumulative strategy for a single
    information set.
    """
    __slots__ = ("regret_sum", "strategy_sum")

    def __init__(self):
        self.regret_sum: List[float] = [0.0] * NUM_ACTIONS
        self.strategy_sum: List[float] = [0.0] * NUM_ACTIONS

    def get_strategy(self, legal_mask: List[int]) -> List[float]:
        """
        Regret-matching: derive current strategy from positive cumulative
        regrets, restricted to ``legal_mask`` action indices.
        """
        strategy = [0.0] * NUM_ACTIONS
        pos_sum = 0.0
        for a in legal_mask:
            val = max(0.0, self.regret_sum[a])
            strategy[a] = val
            pos_sum += val

        if pos_sum > 0:
            for a in legal_mask:
                strategy[a] /= pos_sum
        else:
            # Uniform over legal actions
            n = len(legal_mask)
            for a in legal_mask:
                strategy[a] = 1.0 / n

        return strategy

    def get_average_strategy(self, legal_mask: List[int]) -> List[float]:
        """
        Cumulative average strategy — the standard CFR deployment policy.
        (In two-player zero-sum unabstracted CFR this is the iterate that
        converges to equilibrium; in this six-player abstracted setup it
        is a best-effort approximation with no such guarantee.)
        """
        strategy = [0.0] * NUM_ACTIONS
        total = sum(self.strategy_sum[a] for a in legal_mask)
        if total > 0:
            for a in legal_mask:
                strategy[a] = self.strategy_sum[a] / total
        else:
            n = len(legal_mask)
            for a in legal_mask:
                strategy[a] = 1.0 / n
        return strategy

    def to_dict(self) -> Dict:
        return {
            "regret_sum": list(self.regret_sum),
            "strategy_sum": list(self.strategy_sum),
        }

    @staticmethod
    def from_dict(d: Dict) -> "_CFRNode":
        node = _CFRNode()
        node.regret_sum = list(d["regret_sum"])
        node.strategy_sum = list(d["strategy_sum"])
        return node


# ═══════════════════════════════════════════════════════════════════════════════
#  CFR Bot
# ═══════════════════════════════════════════════════════════════════════════════

class CFRBot:
    """
    Monte Carlo Counterfactual Regret Minimisation (MCCFR) bot.

    Parameters
    ----------
    iterations : int
        Number of MCCFR self-play iterations to run *per decision point* to
        refine regrets before choosing an action.
    profile_path : str | None
        Path for persisting regret / strategy tables. ``None`` = in-memory only.
    use_average : bool
        If ``True`` (default), play the cumulative average strategy (the
        standard CFR deployment policy). If ``False``, play the current
        regret-matched strategy.
    """

    def __init__(
        self,
        iterations: int = 100,
        profile_path: Optional[str] = None,
        use_average: bool = True,
        inference_mode: bool = False,
        search_depth: int = _DEFAULT_SEARCH_DEPTH,
        allow_stale_profile: bool = False,
    ):
        self.iterations = iterations
        self.profile_path = profile_path
        self.use_average = use_average
        # When True, skip _run_iterations during act() so the loaded regret
        # table is used as-is without being overwritten by online updates.
        self.inference_mode = inference_mode
        self._training = not inference_mode  # inverse for clarity in tree-CFR
        self._search_depth = search_depth
        # Explicit override for the load() profile-format gate: permits
        # resuming TRAINING from a profile whose format_version predates
        # PROFILE_FORMAT_VERSION. Off by default on purpose — stale
        # profiles carry pre-Phase-3 regret/strategy semantics.
        self.allow_stale_profile = allow_stale_profile

        # Node map: info_set_key → _CFRNode
        self._nodes: Dict[str, _CFRNode] = {}
        self._legacy_nodes: Dict[str, _CFRNode] = {}
        self._profile_loaded = False

        # Session statistics
        self._hands_played = 0
        self._total_iterations = 0
        self._recursion_calls = 0   # Gate 3A: anti-substitution counter

        # Per-instance RNG for decision sampling. Seeded from the global so
        # `random.seed(N)` at script start still cascades into reproducibility,
        # but multiple bots in one process have independent decision streams
        # (vs. all coupling through the module-level random state).
        self._rng = random.Random(random.getrandbits(64))

        # Gate 2A item 6: internal opponent stat tracker.
        # Constructed lazily on first act() with known n_seats.
        # Limitation: no hand-end observation, so showdown_freq and
        # fold_to_cbet read low-sample. VPIP and AF (the dimensions
        # that drive the bucket) work correctly.
        self._opp_stats: Optional[_OpponentStatTracker] = None
        self._last_history_len = 0
        self._last_history_snapshot: List[Dict[str, Any]] = []
        self._last_hand_id = None

        # Attempt to load persisted profile
        if profile_path:
            self.load(profile_path)

    # ──────────────────────────────────────────────────────────────────────────
    #  Public interface: act(state) → Action
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_hand_boundary(self, view: PlayerView) -> bool:
        """Return True when view.history belongs to a new hand."""
        hand_id = getattr(view, "hand_id", None)
        if hand_id is not None:
            return self._last_hand_id is not None and hand_id != self._last_hand_id

        current = list(view.history or [])
        if not current:
            return self._last_history_snapshot != []
        if len(current) < self._last_history_len:
            return True
        if current[:self._last_history_len] != self._last_history_snapshot:
            return True
        return False

    def act(self, state: PlayerView) -> Action:
        """
        Choose an action for the current game state.

        1. Update internal opponent stat tracker from new history entries.
        2. Compute card bucket, position, SPR, opp-stat bucket.
        3. If NOT in inference_mode, run MCCFR iterations to update regrets.
        4. If profile is loaded and inference mode, do real-time subgame search.
        5. Select an action from the (average) strategy profile, falling back
           to an equity-based heuristic for unseen information sets.
        """
        hole = state.hole_cards
        board = state.board
        pot = state.pot
        to_call = state.to_call
        legal = state.legal_actions
        street = state.street
        history = state.history or []
        hero_stack = int(state.stacks.get(state.me, 0))
        call_amount = max(0, int(to_call))
        # Betting context for pot-relative raise sizing (shared by the legal
        # mask and the concrete action so they stay consistent).
        current_bet = _current_bet_from_view(state, to_call)

        # How many opponents can still affect future action?
        acting_opponents = acting_opponents_for(state)
        n_opp = len(acting_opponents) if acting_opponents else 1
        n_opp = max(1, n_opp)

        # Bail fast if we have no cards
        if not hole or len(hole) < 2:
            return _fallback_passive(legal)

        # ── Item 6: Update internal opponent stat tracker ────────
        n_seats = _tracker_size_for_view(state)
        if self._opp_stats is None:
            self._opp_stats = _OpponentStatTracker(n_seats=n_seats, window=50)
            self._last_history_len = 0
            self._last_history_snapshot = []
        else:
            self._opp_stats.ensure_n_seats(n_seats)

        # Hand-end detection: history resets per engine hand, but the first view
        # we see in a hand can already contain earlier actions.
        if self._detect_hand_boundary(state):
            self._opp_stats.observe_hand_end([])
            self._last_history_len = 0
            self._last_history_snapshot = []

        # Walk new history entries since last act() call
        pids = list(state.stacks.keys())
        new_entries = history[self._last_history_len:]
        for entry in new_entries:
            pid = entry.get("pid", "")
            seat_idx = _stable_seat_index(state, pid, pids)
            if seat_idx is not None:
                self._opp_stats.observe_action(
                    seat_idx=seat_idx,
                    street=entry.get("street", "preflop"),
                    action=entry.get("type", "check"),
                    pot_before=entry.get("pot_before", 0),
                )
        self._last_history_len = len(history)
        self._last_history_snapshot = list(history)
        self._last_hand_id = getattr(state, "hand_id", None)

        # ── Card abstraction ────────────────────────────────────
        if street == "preflop":
            bucket = _preflop_bucket(hole)
        else:
            bucket = _postflop_bucket(hole, board, n_opponents=n_opp)

        # ── History abstraction ─────────────────────────────────
        hist_key = _abstract_history(history, pot)

        # ── Position & SPR abstraction ──────────────────────────
        pos_b = _position_bucket(state.position)
        opp_stacks = [
            int(state.stacks.get(o, 0))
            for o in acting_opponents
            if int(state.stacks.get(o, 0)) > 0
        ]
        spr_b = _spr_bucket(hero_stack, pot, opp_stacks)

        # ── Item 3: Opponent stat bucket ────────────────────────
        opp_stat_b = self._compute_opp_stat_bucket(state)

        # ── Information-set key (7 fields, 6 colons) ────────────
        info_key = _info_set_key(street, bucket, hist_key,
                                 n_opponents=n_opp,
                                 position_bucket=pos_b,
                                 spr_bucket=spr_b,
                                 opp_stat_bucket=opp_stat_b)

        # ── Legal abstract actions ──────────────────────────────
        legal_mask = _legal_abstract_actions(legal, pot, to_call=to_call,
                                             current_bet=current_bet)

        # ── MCCFR updates: only during training ─────────────────
        if not self.inference_mode:
            self._run_iterations(info_key, legal_mask, pot, hole, board,
                                 street, n_opponents=n_opp,
                                 call_amount=call_amount,
                                 hero_stack=hero_stack, view=state)

        # ── Choose action from strategy ─────────────────────────
        node = self._lookup_node(info_key)

        # Deployability (Phase 3.1): a node can hold positive regret with
        # zero strategy_sum on the legal actions (profiles trained before
        # root averaging landed, or infosets visited only in regret role).
        # Deploy the regret-matched current strategy then, instead of
        # discarding the training.
        has_avg = node is not None and any(
            node.strategy_sum[a] > 0.0 for a in legal_mask)
        has_regret = node is not None and any(
            node.regret_sum[a] > 0.0 for a in legal_mask)

        # Untrained node: fall back to subgame search (inference) or an
        # equity-based heuristic (training).
        if not has_avg and not has_regret:
            # Item 5: Even with empty profile, try subgame search
            if self.inference_mode:
                return self._search_fallback(state, legal_mask, legal, pot,
                                             hole, board, n_opp, to_call,
                                             current_bet=current_bet)
            equity = self._quick_equity(hole, board, n_opponents=n_opp)
            return self._heuristic_action(legal_mask, equity, pot, to_call,
                                          legal, current_bet=current_bet)

        if self.use_average and has_avg:
            avg_strategy = node.get_average_strategy(legal_mask)
        else:
            avg_strategy = node.get_strategy(legal_mask)

        # ── Item 5: Real-time subgame search at inference ───────
        if self._profile_loaded and self.inference_mode:
            t0 = _time.monotonic()
            refined = self._subgame_search(state, avg_strategy, legal_mask,
                                           depth=self._search_depth)
            elapsed = _time.monotonic() - t0
            if elapsed > 5.0:
                print(f"[CFRBot] [WARN] _subgame_search took {elapsed:.2f}s "
                      f"(budget: 2s at depth={self._search_depth})")
            abstract_idx = self._sample_action(refined, legal_mask)
        else:
            abstract_idx = self._sample_action(avg_strategy, legal_mask)

        self._hands_played += 1
        return _abstract_to_concrete(abstract_idx, legal, pot,
                                     to_call=to_call, current_bet=current_bet)

    # ──────────────────────────────────────────────────────────────────────────
    #  MCCFR iteration (simplified external-sampling)
    # ──────────────────────────────────────────────────────────────────────────

    def _sample_opponent_hands(self, hero_hole, board, n_opponents: int):
        """Sample one private-card assignment for live opponents."""
        used = {tuple(c) for c in hero_hole} | {tuple(c) for c in board}
        remaining = [c for c in _FULL_DECK if c not in used]
        need = max(0, int(n_opponents)) * 2
        if len(remaining) < need:
            return []
        sampled = random.sample(remaining, need)
        return [
            tuple(sampled[i * 2:(i + 1) * 2])
            for i in range(max(0, int(n_opponents)))
        ]

    def _reconstruct_committed_per_seat(self, view: PlayerView):
        """Walk PlayerView.history and return street/total contributions."""
        return _reconstruct_contributions_from_view(view)

    def _build_training_game_state(self, view: PlayerView, hero_hole,
                                   opp_hands, board):
        """Build a full-info _GameState for recursive training traversal."""
        pids = list(view.stacks.keys())
        if not pids or view.me not in pids:
            return None
        hero_seat = pids.index(view.me)
        stacks = [int(view.stacks[pid]) for pid in pids]
        street_committed, total_committed, real = self._reconstruct_committed_per_seat(view)
        if not real:
            return None
        big_blind = _infer_big_blind_from_view(view)
        last_raise_size = _infer_last_raise_size_from_view(view, big_blind)

        opp_set = set(view.opponents or [])
        alive = [
            (pid == view.me or pid in opp_set) and stacks[i] >= 0
            for i, pid in enumerate(pids)
        ]
        hole_cards = {hero_seat: tuple(hero_hole)}
        hand_iter = iter(opp_hands)
        for opid in view.opponents or []:
            if opid not in pids:
                continue
            try:
                hole_cards[pids.index(opid)] = tuple(next(hand_iter))
            except StopIteration:
                return None

        used = set()
        for cards in hole_cards.values():
            for c in cards:
                used.add(tuple(c))
        for c in board:
            used.add(tuple(c))
        deck_remaining = [c for c in _FULL_DECK if c not in used]

        ring_order = list(range(len(pids)))
        position_labels = _seat_labels_for_view(view, pids, hero_seat)
        full_order = [
            seat for seat in _street_action_order(view.street, ring_order)
            if alive[seat] and stacks[seat] > 0
        ]
        seat_order = _order_after_seat(full_order, hero_seat, include_seat=True)

        return _GameState(
            pot=int(view.pot),
            stacks=stacks,
            committed_per_seat=street_committed,
            total_committed_per_seat=total_committed,
            alive=alive,
            street=view.street,
            board=list(board),
            hole_cards=hole_cards,
            seat_order=seat_order,
            action_idx=0,
            history_tokens=_abstract_history(view.history or [], view.pot),
            deck_remaining=deck_remaining,
            hero_seat=hero_seat,
            real_contributions=True,
            ring_order=ring_order,
            position_labels=position_labels,
            opp_stat_bucket=self._compute_opp_stat_bucket(view),
            big_blind=big_blind,
            last_raise_size=last_raise_size,
        )

    def _info_key_for_state(self, state: _GameState, seat: int) -> str:
        """Build the Gate-2A 7-field key for a recursive state node."""
        n_opponents = sum(
            1 for i, live in enumerate(state.alive)
            if i != seat and live and state.stacks[i] > 0
        )
        n_opponents = max(1, n_opponents)
        hole = state.hole_cards.get(seat, [])
        if hole and len(hole) >= 2:
            if state.street == "preflop":
                bucket = _preflop_bucket(list(hole))
            else:
                bucket = _postflop_bucket(
                    list(hole), state.board, n_opponents=n_opponents
                )
        else:
            bucket = 0

        pos_label = state.position_labels.get(seat, "MP")
        pos_b = _position_bucket(pos_label)
        opp_stacks = [
            int(state.stacks[i]) for i, live in enumerate(state.alive)
            if i != seat and live and int(state.stacks[i]) > 0
        ]
        spr_b = _spr_bucket(int(state.stacks[seat]), state.pot, opp_stacks)
        return _info_set_key(
            state.street,
            bucket,
            state.history_tokens,
            n_opponents=n_opponents,
            position_bucket=pos_b,
            spr_bucket=spr_b,
            opp_stat_bucket=state.opp_stat_bucket,
        )

    def _run_iterations(
        self,
        info_key: str,
        legal_mask: List[int],
        pot: int,
        hole: List[Tuple[str, str]],
        board: List[Tuple[str, str]],
        street: str,
        n_opponents: int,
        call_amount: int,
        hero_stack: int,
        view: PlayerView,
    ):
        """
        Run ``self.iterations`` external-sampling MCCFR traversals rooted at
        the current decision. Each traversal samples opponent hole cards,
        builds a full-info _GameState from PlayerView.history, and recurses.
        """
        completed = 0
        for _ in range(self.iterations):
            sample_opponents = max(1, len(view.opponents or []))
            opp_hands = self._sample_opponent_hands(hole, board, sample_opponents)
            state = self._build_training_game_state(
                view, hero_hole=hole, opp_hands=opp_hands, board=board
            )
            if state is None:
                continue
            self._cfr_recurse(state, state.hero_seat, depth=0)
            completed += 1

        self._total_iterations += completed

    def _estimate_action_value(
        self,
        abstract_idx: int,
        pot: int,
        equity: float,
        n_opponents: int,
        call_amount: int,
        hero_stack: int,
    ) -> float:
        """
        DEPRECATED training value function.

        Estimate the chip EV of taking the given abstract action.

        Fold is the neutral reference point. Other actions are scored in
        chips, then normalized by pot size so regrets stay comparable across
        short-stack and deep-stack spots.

        Recursive MCCFR training now uses _cfr_recurse + _leaf_value. This
        helper remains only for sanity diagnostics and heuristic fallbacks.
        """
        label = ABSTRACT_ACTIONS[abstract_idx]
        pot = max(0, int(pot))
        hero_stack = max(0, int(hero_stack))
        call_amount = max(0, int(call_amount))

        if label == "fold":
            # Neutral reference point — we stop putting chips in.
            return 0.0

        if label == "check_call":
            if call_amount == 0:
                # Free check: no extra risk, so we realize our pot share.
                ev = equity * pot
            else:
                # Calling pays only the amount we can cover.
                cost = min(call_amount, hero_stack)
                ev = equity * (pot + cost) - (1.0 - equity) * cost
            return ev / max(pot, 1)

        # Bet / raise actions
        frac_map = {
            "bet_33": 0.33, "bet_50": 0.50, "bet_67": 0.67,
            "bet_75": 0.75, "bet_100": 1.00, "all_in": 2.0,
        }
        sizing_frac = frac_map.get(label, 0.5)
        bet_size = min(sizing_frac * pot, hero_stack)

        # Bigger bets generate more folds, but with diminishing returns.
        fold_equity = min(0.45, 0.20 * sizing_frac ** 0.7)

        # If everyone folds, we win the pot that already exists.
        fold_value = pot

        # If called, villain matches the bet. We can win the larger pot,
        # but we must subtract the chips we invested.
        called_pot = pot + 2 * bet_size
        called_value = equity * called_pot - bet_size

        # Large bets are volatile; keep a simple chip-risk penalty.
        risk_penalty = 0.05 * bet_size

        ev = fold_equity * fold_value + (1.0 - fold_equity) * called_value
        ev -= risk_penalty

        return ev / max(pot, 1)

    def _quick_equity(
        self,
        hole: List[Tuple[str, str]],
        board: List[Tuple[str, str]],
        n_opponents: int,
    ) -> float:
        """
        Fast Monte-Carlo equity estimate against ``n_opponents`` random
        opponents.

        Delegates to core.equity.equity for the canonical implementation.
        Uses 100 sims for tighter MC estimates (~±0.022 noise).
        """
        return _canonical_equity(hole, board, n_opponents, n_sims=100)

    def _heuristic_action(
        self,
        legal_mask: List[int],
        equity: float,
        pot: int,
        to_call: int,
        legal: List[Dict[str, Any]],
        current_bet: int = 0,
    ) -> Action:
        """
        Equity-based fallback for information sets not seen during training.

        Tiers:
          equity ≥ 0.65  → bet (raise if possible) for value
          equity ≥ 0.45  → call/check if pot odds justify it, else fold
          equity < 0.45  → check if free, else fold
        """
        if equity >= 0.65:
            # Strong hand: bet for value using the largest available sizing.
            bet_actions = [a for a in legal_mask if a >= 2]
            if bet_actions:
                return _abstract_to_concrete(max(bet_actions), legal, pot,
                                             to_call=to_call,
                                             current_bet=current_bet)
            # No bet available → check/call
            return _abstract_to_concrete(1, legal, pot)

        if equity >= 0.45:
            # Marginal hand: call only if pot odds warrant it.
            total = pot + to_call
            pot_odds = to_call / total if total > 0 else 0.0
            if pot_odds <= equity:
                return _abstract_to_concrete(1, legal, pot)   # check/call
            # Bad odds → fold if allowed, else forced call
            if 0 in legal_mask:
                return _abstract_to_concrete(0, legal, pot)   # fold
            return _abstract_to_concrete(1, legal, pot)

        # Weak hand: check for free, otherwise fold.
        if to_call == 0 and 1 in legal_mask:
            return _abstract_to_concrete(1, legal, pot)       # free check
        if 0 in legal_mask:
            return _abstract_to_concrete(0, legal, pot)       # fold
        return _abstract_to_concrete(1, legal, pot)           # forced call

    # ──────────────────────────────────────────────────────────────────────────
    #  Gate 2A: opponent stat bucket (item 3)
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_opp_stat_bucket(self, state: PlayerView) -> str:
        """Build the opp_stat_bucket string for the info-set key.

        Heads-up: single opponent's bucket (e.g. "TA").
        Multiway: sorted concatenation of all live opponents' buckets,
        separated by hyphens (e.g. "LA-LP-TA").
        """
        if self._opp_stats is None:
            return "TA"

        pids = list(state.stacks.keys())
        opp_pids = acting_opponents_for(state)

        buckets = []
        for opid in opp_pids:
            seat_idx = _stable_seat_index(state, opid, pids)
            if seat_idx is not None:
                buckets.append(self._opp_stats.bucket(seat_idx))
            else:
                buckets.append("TA")

        if not buckets:
            return "TA"

        buckets.sort()
        return "-".join(buckets)

    # ──────────────────────────────────────────────────────────────────────────
    #  Gate 2A: AIVAT leaf evaluator (item 2)
    # ──────────────────────────────────────────────────────────────────────────

    def _leaf_value(self, game_state: _GameState, hero_seat: int) -> float:
        """Evaluate a tree leaf.

        Training mode (real_contributions=True):
            Uses core.aivat.value() with full opponent hole cards visible. This
            is the equity-shaping value signal for Path A, with correct
            main/side-pot settlement during regret learning.

        Inference mode (real_contributions=False):
            AIVAT needs exact per-seat contributions and full opponent hole
            cards. Inference has neither, so it falls back to vanilla
            equity-vs-random opponents times pot instead of pretending to be
            side-pot-correct. See SESSION_LOG_2026-04-26.md for context.
        """
        if not game_state.alive[hero_seat]:
            return 0.0

        if not game_state.real_contributions:
            hole = game_state.hole_cards.get(hero_seat, [])
            n_opponents = sum(
                1 for i, live in enumerate(game_state.alive)
                if i != hero_seat and live
            )
            eq = _canonical_equity(
                list(hole),
                list(game_state.board),
                n_opponents=max(1, n_opponents),
                n_sims=200,
            )
            return eq * game_state.pot

        assert sum(game_state.total_committed_per_seat) >= game_state.pot - 1, (
            f"Real-contribution claim violated: total_committed_per_seat="
            f"{game_state.total_committed_per_seat} vs pot={game_state.pot}"
        )

        snapshot = _Snapshot(
            hole_cards=game_state.hole_cards,
            board=tuple(game_state.board),
            pot=game_state.pot,
            stacks=tuple(game_state.stacks),
            alive=tuple(game_state.alive),
            to_call=game_state.to_call_for(hero_seat),
            hero_committed=game_state.total_committed_per_seat[hero_seat],
            committed_per_seat=tuple(game_state.total_committed_per_seat),
        )
        return _aivat_value(snapshot, hero_seat, mode="chip_ev", n_sims=200)

    # ──────────────────────────────────────────────────────────────────────────
    #  Gate 2A: Recursive tree CFR (item 1)
    # ──────────────────────────────────────────────────────────────────────────

    def _cfr_recurse(self, state: _GameState, hero_seat: int,
                     depth: int) -> float:
        """External-sampling MCCFR recursive traversal (Lanctot et al. 2009).

        state: current _GameState
        hero_seat: which seat we're computing regret for this iteration
        depth: recursion depth, bounded by _MAX_CFR_DEPTH

        No explicit reach weighting appears in the updates, on purpose:
        opponent actions are SAMPLED from their current strategy, so any
        sampled path already occurs with frequency proportional to the
        opponents' reach probability. Therefore:
          * Regret increments at hero nodes are unweighted. (A previous
            version also multiplied by cf_reach, which counted opponent
            reach twice — expected weight pi_opp^2 instead of pi_opp.)
          * strategy_sum accumulates at OPPONENT nodes, unweighted: the
            opponent's own sampled actions supply their reach weighting
            in expectation. All training seats share one table
            (train_cfr_bot_multiway), so every info set receives
            average-strategy mass from the traversals of other seats.
          * strategy_sum ALSO accumulates at the hero TRAVERSAL-ROOT node
            (depth 0, Phase 3.1). Traversals are decision-rooted with the
            hero acting first, so the hero's reach at the root is exactly
            1 and the unweighted increment is the textbook average-
            strategy update. Without it, infosets that only ever occur as
            a traversal root — e.g. the hand's FIRST voluntary action,
            whose token history is empty — collect regret but zero
            strategy_sum, and act() would discard the training entirely.
        """
        self._recursion_calls += 1  # Gate 3A: anti-substitution counter
        if state.is_terminal() or depth >= _MAX_CFR_DEPTH:
            return self._leaf_value(state, hero_seat)

        if state.is_chance_node():
            next_state = state.advance_street()
            return self._cfr_recurse(next_state, hero_seat, depth + 1)

        seat = state.seat_to_act()
        if seat < 0:
            return self._leaf_value(state, hero_seat)

        info_key = self._info_key_for_state(state, seat)
        legal_mask = state.legal_abstract_actions()
        node = self._get_node(info_key)
        strategy = node.get_strategy(legal_mask)

        if seat != hero_seat:
            # Opponent: accumulate the average strategy here (textbook
            # external sampling), then sample one action on-policy.
            for a in legal_mask:
                node.strategy_sum[a] += strategy[a]
            action = self._sample_action(strategy, legal_mask)
            next_state = state.apply_action(seat, action)
            return self._cfr_recurse(next_state, hero_seat, depth + 1)

        # Hero at the traversal root: hero reach is exactly 1 here, so the
        # unweighted increment is the exact average-strategy update (see
        # docstring). Deeper hero nodes still accumulate nothing — their
        # mass arrives via other seats' traversals (shared table).
        if depth == 0:
            for a in legal_mask:
                node.strategy_sum[a] += strategy[a]

        # Hero: expand all legal actions
        action_utils = {}
        for a in legal_mask:
            before_commit = state.committed_per_seat[hero_seat]
            next_state = state.apply_action(hero_seat, a)
            added_cost = max(
                0, next_state.committed_per_seat[hero_seat] - before_commit
            )
            action_utils[a] = (
                self._cfr_recurse(next_state, hero_seat, depth + 1)
                - added_cost
            )

        node_util = sum(strategy[a] * action_utils.get(a, 0.0)
                        for a in legal_mask)

        # Unweighted regret update — sampling already supplies the
        # opponents' reach (see docstring).
        for a in legal_mask:
            node.regret_sum[a] += action_utils.get(a, 0.0) - node_util

        return node_util

    # ──────────────────────────────────────────────────────────────────────────
    #  Gate 2A: Real-time subgame search (item 5)
    # ──────────────────────────────────────────────────────────────────────────

    # Inference-search shaping constants (mirror Path B,
    # deep_cfr_bot.py:1140-1142). Subtree EVs come from ONE sampled
    # opponent line per node with unknown opponent cards, so they are
    # noisy; temper, clip, and blend instead of letting a raw chip-unit
    # exp() collapse onto the argmax of that noise.
    _SEARCH_TEMPERATURE = 20.0      # advantage scale, in big blinds
    _SEARCH_ADVANTAGE_CLIP = 2.0    # max |advantage| after temperature
    _SEARCH_BLEND = 0.25            # weight of search vs trained prior

    def _subgame_search(self, state: PlayerView, prior: List[float],
                        legal_mask: List[int], depth: int = 3) -> List[float]:
        """Depth-limited subgame search rooted at the current state.

        Builds a small game tree, evaluates leaves via AIVAT, and blends
        the resulting action EVs back into the trained prior (see
        _shape_search_strategy).
        """
        gs, hero_seat = _build_game_state_from_view(
            state,
            opp_stat_bucket=self._compute_opp_stat_bucket(state),
        )

        # Evaluate each candidate hero action. EVs are converted from
        # chips to big-blind units so shaping is chip-scale invariant.
        big_blind = max(1, int(gs.big_blind or 1))
        action_evs = {}
        for a in legal_mask:
            before_commit = gs.committed_per_seat[hero_seat]
            next_state = gs.apply_action(hero_seat, a)
            added_cost = max(
                0, next_state.committed_per_seat[hero_seat] - before_commit
            )
            ev = self._search_subtree(next_state, hero_seat, depth - 1) - added_cost
            action_evs[a] = ev / big_blind

        return self._shape_search_strategy(action_evs, prior, legal_mask)

    def _shape_search_strategy(self, action_evs: Dict[int, float],
                               prior: List[float],
                               legal_mask: List[int]) -> List[float]:
        """Blend noisy search EVs (in big-blind units) into the prior.

        Mirrors Path B's cautious shaping (deep_cfr_bot.py:1731-1762):
          1. advantage = (ev - prior-weighted baseline) / temperature
          2. clip advantage to +/- _SEARCH_ADVANTAGE_CLIP
          3. w = prior * exp(advantage), normalized
          4. final = 0.75 * prior + 0.25 * w   (_SEARCH_BLEND)
        A single noisy EV sample can therefore shift — but never
        override — the trained average strategy. The old shaping used
        exp() on raw chip EVs, where a 20-chip gap (e^20) annihilated
        the prior and argmaxed single-sample noise.
        """
        refined = [0.0] * NUM_ACTIONS
        if not legal_mask:
            return refined

        prior_total = sum(max(prior[a], 0.0) for a in legal_mask)
        if prior_total > 0:
            prior_norm = {a: max(prior[a], 0.0) / prior_total
                          for a in legal_mask}
        else:
            prior_norm = {a: 1.0 / len(legal_mask) for a in legal_mask}

        baseline = sum(prior_norm[a] * action_evs.get(a, 0.0)
                       for a in legal_mask)

        total = 0.0
        for a in legal_mask:
            adv = (action_evs.get(a, 0.0) - baseline) / self._SEARCH_TEMPERATURE
            adv = max(-self._SEARCH_ADVANTAGE_CLIP,
                      min(self._SEARCH_ADVANTAGE_CLIP, adv))
            w = max(prior_norm[a], 1e-6) * math.exp(adv)
            refined[a] = w
            total += w

        if total > 0:
            for a in legal_mask:
                refined[a] /= total
        else:
            for a in legal_mask:
                refined[a] = prior_norm[a]

        blend = self._SEARCH_BLEND
        for a in legal_mask:
            refined[a] = (1.0 - blend) * prior_norm[a] + blend * refined[a]

        return refined

    def _search_subtree(self, state: _GameState, hero_seat: int,
                        depth: int) -> float:
        """Recursive subtree evaluation for subgame search."""
        if state.is_terminal() or depth <= 0:
            return self._leaf_value(state, hero_seat)

        if state.is_chance_node():
            next_state = state.advance_street()
            return self._search_subtree(next_state, hero_seat, depth - 1)

        seat = state.seat_to_act()
        if seat < 0:
            return self._leaf_value(state, hero_seat)

        legal_mask = state.legal_abstract_actions()

        if seat != hero_seat:
            # Opponent: use profile avg strategy or uniform
            info_key = self._info_key_for_state(state, seat)
            node = self._lookup_node(info_key)
            if node and sum(node.strategy_sum) > 0:
                strategy = node.get_average_strategy(legal_mask)
            else:
                strategy = [0.0] * NUM_ACTIONS
                n = len(legal_mask)
                for a in legal_mask:
                    strategy[a] = 1.0 / n

            # Sample one opponent action
            action = self._sample_action(strategy, legal_mask)
            next_state = state.apply_action(seat, action)
            return self._search_subtree(next_state, hero_seat, depth - 1)

        # Hero: expand all actions, take max (search for best play)
        best_val = float('-inf')
        for a in legal_mask:
            before_commit = state.committed_per_seat[hero_seat]
            next_state = state.apply_action(hero_seat, a)
            added_cost = max(
                0, next_state.committed_per_seat[hero_seat] - before_commit
            )
            val = self._search_subtree(next_state, hero_seat, depth - 1) - added_cost
            best_val = max(best_val, val)
        return best_val

    def _search_fallback(self, state: PlayerView, legal_mask: List[int],
                         legal: List[Dict[str, Any]], pot: int,
                         hole, board, n_opp: int, to_call: int,
                         current_bet: int = 0) -> Action:
        """Fallback when no profile node exists: run search or heuristic."""
        try:
            uniform = [0.0] * NUM_ACTIONS
            n = len(legal_mask)
            for a in legal_mask:
                uniform[a] = 1.0 / n
            t0 = _time.monotonic()
            refined = self._subgame_search(state, uniform, legal_mask,
                                           depth=self._search_depth)
            elapsed = _time.monotonic() - t0
            if elapsed > 5.0:
                print(f"[CFRBot] [WARN] search fallback took {elapsed:.2f}s")
            return _abstract_to_concrete(
                self._sample_action(refined, legal_mask), legal, pot,
                to_call=to_call, current_bet=current_bet)
        except (KeyError, IndexError, ValueError) as e:
            print(
                f"[CFRBot] _search_fallback caught {type(e).__name__}: {e}; "
                "using heuristic action."
            )
            equity = self._quick_equity(hole, board, n_opponents=n_opp)
            return self._heuristic_action(legal_mask, equity, pot, to_call,
                                          legal, current_bet=current_bet)

    # ──────────────────────────────────────────────────────────────────────────
    #  Node management
    # ──────────────────────────────────────────────────────────────────────────

    def _get_node(self, key: str) -> _CFRNode:
        if key not in self._nodes:
            self._nodes[key] = _CFRNode()
        return self._nodes[key]

    @staticmethod
    def _legacy_key_candidates(info_key: str) -> List[str]:
        parts = info_key.split(":")
        if len(parts) < 7:
            return []
        street, n_opp, position, spr, _opp_stat, bucket = parts[:6]
        history = ":".join(parts[6:])
        return [
            f"{street}:{n_opp}:{position}:{spr}:{bucket}:{history}",
        ]

    def _lookup_node(self, key: str) -> Optional[_CFRNode]:
        node = self._nodes.get(key)
        if node is not None:
            return node
        for candidate in self._legacy_key_candidates(key):
            node = self._legacy_nodes.get(candidate)
            if node is not None:
                return node
        return None

    def _sample_action(self, strategy: List[float], legal_mask: List[int]) -> int:
        """Sample an action index from the strategy distribution."""
        r = self._rng.random()
        cumulative = 0.0
        for a in legal_mask:
            cumulative += strategy[a]
            if r <= cumulative:
                return a
        return legal_mask[-1]  # fallback to last legal action

    # ──────────────────────────────────────────────────────────────────────────
    #  Persistence: save / load
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None):
        """
        Persist the regret and strategy tables to disk as a pickle file.

        The write is atomic: data is written to ``path + ".tmp"`` first,
        flushed and fsynced, then renamed over ``path``.  A crash or
        KeyboardInterrupt during the dump therefore cannot corrupt the
        existing checkpoint.
        """
        path = path or self.profile_path
        if not path:
            return

        dirn = os.path.dirname(path)
        if dirn:
            os.makedirs(dirn, exist_ok=True)

        data = {
            "format_version": PROFILE_FORMAT_VERSION,
            "nodes": {k: v.to_dict() for k, v in self._nodes.items()},
            "hands_played": self._hands_played,
            "total_iterations": self._total_iterations,
        }
        tmp_path = path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    def load(self, path: Optional[str] = None):
        """
        Load regret and strategy tables from a pickle file.
        """
        path = path or self.profile_path
        if not path:
            return
        if not os.path.exists(path):
            if self.inference_mode:
                raise RuntimeError(
                    f"[CFRBot] Missing profile {path!r}. In inference_mode, "
                    "a trained CFR profile is required; train one or pass "
                    "cfr:<path> to an existing profile."
                )
            return

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            loaded_nodes = {
                k: _CFRNode.from_dict(v) for k, v in data["nodes"].items()
            }
            hands_played = data.get("hands_played", 0)
            total_iterations = data.get("total_iterations", 0)
            format_version = int(data.get("format_version", 0) or 0)
        except Exception as e:
            print(f"[CFRBot] Could not load profile from {path}: {e}")
            self._nodes = {}
            self._legacy_nodes = {}
            return

        # Profile format gate (Phase 3.1): refuse to RESUME TRAINING from a
        # profile that predates PROFILE_FORMAT_VERSION — its regrets and
        # strategy sums were accumulated under pre-Phase-3 semantics
        # (cf_reach double count, fixed-token tree histories, hero-node
        # averaging) and would silently poison a fresh run. Inference-mode
        # loads stay permitted (read-only play / eval against old profiles).
        if (not self.inference_mode
                and not self.allow_stale_profile
                and format_version < PROFILE_FORMAT_VERSION):
            raise RuntimeError(
                f"[CFRBot] Profile {path!r} has format_version="
                f"{format_version} (< {PROFILE_FORMAT_VERSION}). Training "
                f"must start from a fresh profile — move/delete the old "
                f"file, point --profile at a new path, or pass "
                f"allow_stale_profile=True to override explicitly."
            )

        # Gate 2A format: 7 fields, 6 colons:
        # street:n_opp:position:spr:opp_stat:bucket:history.
        # Older keys are retained for deterministic inference lookup fallbacks
        # but are not written back by save().
        self._nodes = {}
        self._legacy_nodes = {}
        for k, node in loaded_nodes.items():
            if k.count(":") >= 6:
                self._nodes[k] = node
            else:
                self._legacy_nodes[k] = node
        self._hands_played = hands_played
        self._total_iterations = total_iterations
        legacy = len(self._legacy_nodes)
        if legacy:
            print(
                f"[CFRBot] Loaded {legacy} old-format keys for "
                f"compatibility fallback. {len(self._nodes)} current keys remain."
            )

        if not self._nodes and not self._legacy_nodes and self.inference_mode:
            raise RuntimeError(
                f"[CFRBot] Profile {path!r} has 0 valid info-set keys after "
                f"loading. In inference_mode, a working profile is "
                f"required — refusing to silently fall back to heuristic. "
                f"Use models/cfr_regret_deep_v2.pkl or train a fresh profile."
            )

        if self._nodes or self._legacy_nodes:
            self._profile_loaded = True
        print(f"[CFRBot] Loaded profile from {path} "
              f"({len(self._nodes)} current info sets, "
              f"{len(self._legacy_nodes)} legacy info sets, "
              f"{self._total_iterations} iterations)")

    # ──────────────────────────────────────────────────────────────────────────
    #  Diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return diagnostic statistics about the CFR profile."""
        return {
            "info_sets": len(self._nodes),
            "legacy_info_sets": len(self._legacy_nodes),
            "hands_played": self._hands_played,
            "total_iterations": self._total_iterations,
            "recursion_calls": self._recursion_calls,
        }
