"""
core/action_history.py — Canonical action representation + tokenizer + tensor encoder
-------------------------------------------------------------------------------------
Two consumers (Path A wants tokens, Path B wants tensors), one source of truth.

Extracted from the inline tokenizer in bots/cfr_bot.py (session 2026-04-26).

Both Path A and Path B import from here. This module is read-only
after Gate 1 closes.

Phase-3 amendment (2026-06-10): sizing_token() extracted from tokenize()
so the tree traversal (_GameState.apply_action) and live-play tokenization
share one bucketing. No behavior change to tokenize() itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

try:
    import torch
except ImportError:
    torch = None  # graceful degradation for non-torch environments


@dataclass(frozen=True)
class ActionEvent:
    """Canonical representation of a single action in the hand history."""
    seat: int
    street: str        # "preflop" | "flop" | "turn" | "river"
    action: str        # "fold" | "check" | "call" | "bet" | "raise" | "all_in"
    amount: int        # chips put in by this action.
                       # For bet/raise: TOTAL bet size (target), matching the
                       #   engine's history entry["amount"]. NOT the delta
                       #   beyond an existing bet.
                       # For call: chips matched to the existing bet.
                       # For all_in: same total convention as bet/raise.
                       # For fold/check: 0.
    pot_before: int    # snapshot at action time -- last session's bug fix


# ─── Street and action vocabularies ──────────────────────────────────────────

_STREETS = ("preflop", "flop", "turn", "river")
_STREET_TO_IDX = {s: i for i, s in enumerate(_STREETS)}

# NOTE on "all_in" (2026-06-11, I3/B1 parity fix): the engine has no all-in
# action type (a shove is a bet/raise that happens to commit the whole stack),
# and the Deep CFR tree now mirrors that — neither producer emits "all_in"
# events anymore.  The label stays in the vocabulary so tensor shapes and
# previously trained checkpoints are unchanged; its one-hot channel simply
# goes unused.
_ACTIONS = ("fold", "check", "call", "bet", "raise", "all_in")
_ACTION_TO_IDX = {a: i for i, a in enumerate(_ACTIONS)}


def extract_history(player_view) -> list:
    """Pull the canonical action history off the engine's PlayerView.

    `amount` matches engine history entries directly (total, not delta).

    Parameters
    ----------
    player_view : PlayerView
        The engine's read-only game state.

    Returns
    -------
    list[ActionEvent]

    Engine contract dependency
    --------------------------
    This function relies on player_view.stacks.keys() returning pids in seat
    order, matching the order in which the engine constructs the stacks dict
    from the seating ring. If the engine ever reorders stacks (for example,
    active-only views or stack-sorted views), this mapping breaks silently.
    The defensive ValueError below catches the mismatch case where a history
    entry references a pid not in stacks. It does NOT catch reordering; fixing
    that would require an explicit seat_index field on PlayerView in a future
    Gate-2 amendment.
    """
    history = getattr(player_view, 'history', None) or []
    events = []

    # Build a mapping of player_id -> seat_idx from stacks
    stacks = getattr(player_view, 'stacks', {})
    pid_to_seat = {}
    if stacks:
        for i, pid in enumerate(stacks.keys()):
            pid_to_seat[pid] = i

    for entry in history:
        if not isinstance(entry, dict):
            continue

        pid = entry.get("pid", "")
        if pid not in pid_to_seat:
            raise ValueError(
                f"extract_history: pid {pid!r} not in stacks={list(stacks)}; "
                "engine seat-order contract violated"
            )
        seat = pid_to_seat[pid]
        street = entry.get("street", "preflop")
        action_type = entry.get("type", "check")

        # Normalize action type
        if action_type in ("bet", "raise"):
            action = action_type
        elif action_type == "fold":
            action = "fold"
        elif action_type == "check":
            action = "check"
        elif action_type == "call":
            action = "call"
        else:
            action = "check"  # safe fallback

        amount = entry.get("amount") or 0
        pot_before = entry.get("pot_before", 0)

        events.append(ActionEvent(
            seat=seat,
            street=street,
            action=action,
            amount=int(amount),
            pot_before=int(pot_before),
        ))

    return events


def sizing_token(amount: int, pot_before: int) -> str:
    """Map a bet/raise TOTAL street contribution to its sizing token.

    Single source of truth for sizing buckets. Used by tokenize() below
    (live play) AND by Path A's tree traversal (_GameState.apply_action),
    so info-set keys learned in the tree stay reachable at inference.

    amount:     the actor's resulting total street contribution
                (engine `amount` semantics — total, not delta).
    pot_before: pot snapshot before this action's chips went in.
    """
    if pot_before > 0:
        ratio = amount / pot_before
    else:
        ratio = 1.0

    # Bin ratio into one of 6 sizing tokens
    if ratio >= 1.2:
        return "A"   # all-in / over-pot
    if ratio >= 0.85:
        return "P"   # pot-sized (~100%)
    if ratio >= 0.70:
        return "L"   # large (~75%)
    if ratio >= 0.55:
        return "M"   # medium (~67%)
    if ratio >= 0.40:
        return "Q"   # quarter-ish (~50%)
    return "S"       # small (~33%)


def tokenize(events) -> str:
    """Path A's tokenizer -- replaces the inline scheme in cfr_bot.py.

    Tokens: F (fold), K (check), C (call), S (~33%), Q (~50%), M (~67%),
            L (~75%), P (~100%), A (all-in).

    Pot-fraction ratio: ratio = event.amount / event.pot_before.
    This matches cfr_bot.py's existing convention (`amt / pot_before`)
    and the engine's amount semantics (total, not delta). Path of least
    surprise -- do NOT switch to (amount - existing_bet) / pot_before.

    Parameters
    ----------
    events : list[ActionEvent]

    Returns
    -------
    str: concatenated token string
    """
    tokens = []
    for event in events:
        action = event.action

        if action == "fold":
            tokens.append("F")
        elif action == "check":
            tokens.append("K")
        elif action == "call":
            tokens.append("C")
        elif action in ("bet", "raise", "all_in"):
            tokens.append(sizing_token(event.amount, event.pot_before))
        else:
            tokens.append("?")

    return "".join(tokens)


# ─── Tensor encoder for Path B ───────────────────────────────────────────────

# Feature dimensions for tensor encoding
_N_SEATS_MAX = 10      # max seats at a table
_N_STREETS = len(_STREETS)
_N_ACTIONS = len(_ACTIONS)
# Per-event: seat_onehot + street_onehot + action_onehot + amount_norm + pot_norm + mask
FEATURE_DIM = _N_SEATS_MAX + _N_STREETS + _N_ACTIONS + 2 + 1  # = 23

# Reference big blind for the chip-feature scale (Fix 5 / I6).  Every chip
# quantity is rescaled to this blind level before normalization
# (chips * REF_BIG_BLIND / big_blind), making the encoding blind-level
# invariant: the same spot at 5/10 and at 50/100 produces identical tensors.
# At big_blind == REF_BIG_BLIND the rescale factor is exactly 1.0, so all
# historical 5/10 encodings (and the checkpoints trained on them) are
# byte-identical to before.
REF_BIG_BLIND = 10


def to_tensor(events, max_len=64, big_blind: int = REF_BIG_BLIND):
    """Path B's input -- fixed-shape padded tensor for GRU consumption.

    Each event encoded as [seat_onehot, street_onehot, action_onehot,
    amount_norm, pot_norm, mask]. Shape: (max_len, feature_dim).
    Padding token has its mask channel set to 0.

    Parameters
    ----------
    events : list[ActionEvent]
    max_len : int
    big_blind : int
        The game's big blind in chips.  Event amounts/pots are rescaled to
        the REF_BIG_BLIND chip scale before the log-normalization so the
        features are blind-level invariant (see REF_BIG_BLIND above).
        Defaults to REF_BIG_BLIND, which reproduces the historical encoding.

    Returns
    -------
    torch.Tensor of shape (max_len, FEATURE_DIM)
    """
    if torch is None:
        raise ImportError("torch is required for to_tensor()")

    import math

    bb = max(1, int(big_blind or REF_BIG_BLIND))
    tensor = torch.zeros(max_len, FEATURE_DIM, dtype=torch.float32)

    for i, event in enumerate(events[:max_len]):
        offset = 0

        # Seat one-hot (up to _N_SEATS_MAX)
        seat_idx = min(event.seat, _N_SEATS_MAX - 1)
        tensor[i, offset + seat_idx] = 1.0
        offset += _N_SEATS_MAX

        # Street one-hot
        street_idx = _STREET_TO_IDX.get(event.street, 0)
        tensor[i, offset + street_idx] = 1.0
        offset += _N_STREETS

        # Action one-hot
        action_idx = _ACTION_TO_IDX.get(event.action, 1)  # default to check
        tensor[i, offset + action_idx] = 1.0
        offset += _N_ACTIONS

        # Normalized amount (log scale, capped), at the reference blind scale.
        # Use log(1 + amount) / log(1 + 10000) to normalize to ~[0, 1]
        amount_eq = event.amount * REF_BIG_BLIND / bb
        amount_norm = math.log1p(amount_eq) / math.log1p(10000)
        tensor[i, offset] = min(amount_norm, 1.0)
        offset += 1

        # Normalized pot, at the reference blind scale
        pot_eq = event.pot_before * REF_BIG_BLIND / bb
        pot_norm = math.log1p(pot_eq) / math.log1p(10000)
        tensor[i, offset] = min(pot_norm, 1.0)
        offset += 1

        # Mask (1 = real event, 0 = padding)
        tensor[i, offset] = 1.0

    return tensor
