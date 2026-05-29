"""
bots/deep_cfr_bot.py — Path B Deep CFR Plus bot
-------------------------------------------------
Neural network approximation of CFR with:
  - Shared state encoder (card embeddings + action-history GRU + opponent pooling + scalars)
  - Three heads: regret, value, sizing
  - Real-time depth-limited subgame search at inference
  - ReservoirBuffer for Gate 3 training data collection

This module is read-only for Antigravity-A. Only Antigravity-B edits it.
"""
from __future__ import annotations

import math
import os
import random
import time as _time
import warnings
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.bot_api import Action, PlayerView, BotAdapter, acting_opponents_for
from core.action_history import (
    ActionEvent, extract_history, to_tensor as history_to_tensor,
    FEATURE_DIM as HIST_FEATURE_DIM,
)
from core.opponent_stats import OpponentStatTracker, OpponentStats
from core.table_order import street_action_order

# ═══════════════════════════════════════════════════════════════════════════════
#  Constants (shared with Path A's cfr_bot.py — same abstract action space)
# ═══════════════════════════════════════════════════════════════════════════════

ABSTRACT_ACTIONS: List[str] = [
    "fold", "check_call", "bet_33", "bet_50",
    "bet_67", "bet_75", "bet_100", "all_in",
]
NUM_ACTIONS = len(ABSTRACT_ACTIONS)

_POSITION_MAP = {
    "BTN": 2, "CO": 2, "HJ": 2,
    "MP": 1, "LJ": 1,
    "UTG": 0, "UTG+1": 0, "UTG+2": 0,
    "SB": 3, "BB": 3,
}
_STREET_MAP = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
_MAX_OPPONENTS = 5
_OPP_FEAT_DIM = 6  # stack, committed, can_act, all_in, reserved, reserved
_SCALAR_DIM = 3 + 4 + 3 + 4  # pot/call/stack + position(4) + spr(3) + street(4) = 14
_HISTORY_MAX_LEN = 64


# ═══════════════════════════════════════════════════════════════════════════════
#  DeepCFRConfig
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DeepCFRConfig:
    """Architecture knobs. Two scales: small and large."""
    card_embed_dim: int
    gru_hidden: int
    opp_embed_dim: int
    state_dim: int
    head_hidden: int
    head_layers: int

    @classmethod
    def small(cls) -> "DeepCFRConfig":
        """Smaller variant — aspirational target ~5M parameters, but accepted
        at whatever the architecture naturally lands on."""
        return cls(card_embed_dim=64, gru_hidden=128,
                   opp_embed_dim=32, state_dim=256,
                   head_hidden=256, head_layers=2)

    @classmethod
    def large(cls) -> "DeepCFRConfig":
        """Larger variant — aspirational target ~15M parameters, but accepted
        at whatever the architecture naturally lands on."""
        return cls(card_embed_dim=96, gru_hidden=256,
                   opp_embed_dim=64, state_dim=384,
                   head_hidden=512, head_layers=3)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper: card → integer mapping
# ═══════════════════════════════════════════════════════════════════════════════

_RANK_ORDER = "23456789TJQKA"
_SUIT_ORDER = "cdhs"
_FULL_DECK = [(r, s) for r in _RANK_ORDER for s in _SUIT_ORDER]
_STREET_ORDER = ("preflop", "flop", "turn", "river")
_NEXT_STREET = {"preflop": "flop", "flop": "turn", "turn": "river"}
_DEAL_ON_ADVANCE = {"preflop": 3, "flop": 1, "turn": 1}

def _card_to_idx(card) -> int:
    """Map (rank, suit) tuple to integer 0..51."""
    r, s = card
    return _RANK_ORDER.index(r) * 4 + _SUIT_ORDER.index(s)


def _seat_position_label(seat: int, n_seats: int) -> str:
    """Best-effort seat label for state-derived network inputs."""
    if n_seats <= 2:
        return "BTN" if seat == 0 else "BB"
    labels = ["BTN", "SB", "BB", "UTG", "UTG+1", "MP", "LJ", "HJ", "CO"]
    return labels[seat] if seat < len(labels) else "MP"


# ═══════════════════════════════════════════════════════════════════════════════
#  StateEncoder
# ═══════════════════════════════════════════════════════════════════════════════

class StateEncoder(nn.Module):
    """Shared front-end producing a (batch, state_dim) vector."""

    def __init__(self, config: DeepCFRConfig):
        super().__init__()
        self.config = config
        sd = config.state_dim
        quarter = sd // 4

        # Card embeddings
        self.card_embed = nn.Embedding(52, config.card_embed_dim)
        self.card_position_embed = nn.Embedding(7, config.card_embed_dim)
        self.card_proj = nn.Sequential(
            nn.Linear(config.card_embed_dim * 7, quarter),
            nn.ReLU(),
        )

        # Action history GRU
        self.history_gru = nn.GRU(
            input_size=HIST_FEATURE_DIM, hidden_size=config.gru_hidden,
            batch_first=True,
        )
        self.history_proj = nn.Sequential(
            nn.Linear(config.gru_hidden, quarter), nn.ReLU(),
        )

        # Opponent pooling
        self.opp_proj = nn.Sequential(
            nn.Linear(_OPP_FEAT_DIM, config.opp_embed_dim), nn.ReLU(),
            nn.Linear(config.opp_embed_dim, quarter), nn.ReLU(),
        )

        # Scalar projection
        self.scalar_proj = nn.Sequential(
            nn.Linear(_SCALAR_DIM, quarter), nn.ReLU(),
        )

        # Fusion MLP
        self.fuse = nn.Sequential(
            nn.Linear(sd, sd), nn.ReLU(),
            nn.Linear(sd, sd), nn.ReLU(),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        dev = next(self.parameters()).device

        # --- Cards: (batch, 7) integer indices, with -1 for absent cards ---
        card_ids = batch["card_ids"].to(dev)          # (B, 7)
        card_mask = (card_ids >= 0).float()            # (B, 7)
        safe_ids = card_ids.clamp(min=0)               # (B, 7)
        pos_idx = torch.arange(7, device=dev).unsqueeze(0).expand_as(safe_ids)
        card_vec = self.card_embed(safe_ids) + self.card_position_embed(pos_idx)
        card_vec = card_vec * card_mask.unsqueeze(-1)   # zero out absent
        B = card_vec.shape[0]
        card_flat = card_vec.view(B, -1)                # (B, 7*embed)
        card_out = self.card_proj(card_flat)             # (B, quarter)

        # --- History GRU ---
        hist = batch["history"].to(dev)                 # (B, 64, FEAT)
        hist_mask = batch["history_mask"].to(dev)       # (B, 64)
        gru_out, _ = self.history_gru(hist)             # (B, 64, gru_hidden)
        # Get last real event per sequence
        lengths = hist_mask.sum(dim=1).long().clamp(min=1) - 1  # (B,)
        last_hidden = gru_out[torch.arange(B, device=dev), lengths]
        hist_out = self.history_proj(last_hidden)       # (B, quarter)

        # --- Opponent pooling ---
        opp = batch["opp_features"].to(dev)             # (B, MAX_OPP, 6)
        opp_mask = batch["opp_mask"].to(dev)            # (B, MAX_OPP)
        opp_emb = self.opp_proj(opp)                    # (B, MAX_OPP, quarter)
        opp_emb = opp_emb * opp_mask.unsqueeze(-1)
        denom = opp_mask.sum(dim=1, keepdim=True).clamp(min=1)
        opp_out = opp_emb.sum(dim=1) / denom            # (B, quarter)

        # --- Scalars ---
        scalars = batch["scalars"].to(dev)               # (B, _SCALAR_DIM)
        scalar_out = self.scalar_proj(scalars)            # (B, quarter)

        # --- Fuse ---
        cat = torch.cat([card_out, hist_out, opp_out, scalar_out], dim=-1)
        return self.fuse(cat)


# ═══════════════════════════════════════════════════════════════════════════════
#  Heads
# ═══════════════════════════════════════════════════════════════════════════════

def _build_mlp(in_dim, hidden, layers, out_dim):
    mods = []
    for i in range(layers):
        mods.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
        mods.append(nn.ReLU())
    mods.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*mods)


class RegretHead(nn.Module):
    def __init__(self, c: DeepCFRConfig):
        super().__init__()
        self.mlp = _build_mlp(c.state_dim, c.head_hidden, c.head_layers, NUM_ACTIONS)

    def forward(self, state):
        return self.mlp(state)


class ValueHead(nn.Module):
    def __init__(self, c: DeepCFRConfig):
        super().__init__()
        self.mlp = _build_mlp(c.state_dim, c.head_hidden, c.head_layers, 1)

    def forward(self, state):
        return self.mlp(state)


class SizingHead(nn.Module):
    """Predicts bucket-fraction sizing, not true continuous bet sizing.

    Gate 3B training targets are the best abstract bet bucket's fraction
    ({0.33, 0.50, 0.67, 0.75, 1.00}). The output range remains [0, 2.0] as
    a scaffold for future continuous sizing work, but current targets only
    teach those discrete bucket fractions.
    """

    def __init__(self, c: DeepCFRConfig):
        super().__init__()
        self.mlp = _build_mlp(c.state_dim, c.head_hidden, c.head_layers, 1)

    def forward(self, state):
        return torch.sigmoid(self.mlp(state)) * 2.0


# ═══════════════════════════════════════════════════════════════════════════════
#  DeepCFRNetwork
# ═══════════════════════════════════════════════════════════════════════════════

class DeepCFRNetwork(nn.Module):
    def __init__(self, config: DeepCFRConfig):
        super().__init__()
        self.config = config
        self.encoder = StateEncoder(config)
        self.regret_head = RegretHead(config)
        self.value_head = ValueHead(config)
        self.sizing_head = SizingHead(config)

        # Kaiming init for all linear layers; small noise on output biases
        # so that initial outputs are varied (not constant).
        self.apply(self._init_weights)

        total = sum(p.numel() for p in self.parameters())
        print(f"[DeepCFRNetwork] config={config}, params={total:,}")

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.normal_(module.bias, mean=0.0, std=0.1)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.1)

    def forward(self, batch: dict) -> dict:
        state = self.encoder(batch)
        return {
            "regret": self.regret_head(state),
            "value": self.value_head(state).squeeze(-1),
            "sizing": self.sizing_head(state).squeeze(-1),
            "state": state,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Input construction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_card_ids(hole_cards, board) -> torch.Tensor:
    """Return (7,) int tensor: 2 hole + up to 5 board, -1 for absent."""
    ids = [-1] * 7
    for i, c in enumerate(hole_cards[:2]):
        ids[i] = _card_to_idx(c)
    for i, c in enumerate(board[:5]):
        ids[2 + i] = _card_to_idx(c)
    return torch.tensor(ids, dtype=torch.long)


def _build_scalars(pot, to_call, hero_stack, position, street, n_opp,
                   opp_stacks) -> torch.Tensor:
    """Return (_SCALAR_DIM,) float tensor."""
    norm = max(pot, 1.0)
    pot_n = math.log1p(pot) / math.log1p(10000)
    call_n = math.log1p(to_call) / math.log1p(10000)
    stack_n = math.log1p(hero_stack) / math.log1p(10000)

    pos_oh = [0.0] * 4
    pos_oh[_POSITION_MAP.get(position, 1)] = 1.0

    eff = hero_stack
    if opp_stacks:
        eff = min(hero_stack, max(opp_stacks))
    spr = eff / max(pot, 1)
    spr_oh = [0.0] * 3
    if spr < 5:
        spr_oh[0] = 1.0
    elif spr < 15:
        spr_oh[1] = 1.0
    else:
        spr_oh[2] = 1.0

    street_oh = [0.0] * 4
    street_oh[_STREET_MAP.get(street, 0)] = 1.0

    return torch.tensor(
        [pot_n, call_n, stack_n] + pos_oh + spr_oh + street_oh,
        dtype=torch.float32,
    )


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


def _safe_extract_history(view: PlayerView) -> List[ActionEvent]:
    try:
        return extract_history(view)
    except ValueError:
        valid_pids = set((view.stacks or {}).keys())
        clean_history = [
            entry for entry in (view.history or [])
            if not isinstance(entry, dict) or entry.get("pid", "") in valid_pids
        ]
        if len(clean_history) == len(view.history or []):
            return []
        try:
            return extract_history(replace(view, history=clean_history))
        except ValueError:
            return []


def _fill_public_opp_features(
    opp_features: torch.Tensor,
    opp_mask: torch.Tensor,
    index: int,
    stack: int,
    committed: int,
    pot: int,
    can_act: bool,
    all_in: bool,
):
    opp_mask[0, index] = 1.0
    opp_features[0, index, 0] = min(1.0, max(0, int(stack)) / 1000.0)
    opp_features[0, index, 1] = min(1.0, max(0, int(committed)) / max(int(pot), 1))
    opp_features[0, index, 2] = 1.0 if can_act else 0.0
    opp_features[0, index, 3] = 1.0 if all_in else 0.0


def build_network_input(view: PlayerView, opp_tracker=None) -> dict:
    """Convert a PlayerView into the dict of tensors the network expects."""
    card_ids = _build_card_ids(view.hole_cards, view.board).unsqueeze(0)

    events = _safe_extract_history(view)
    hist_tensor = history_to_tensor(events, max_len=_HISTORY_MAX_LEN)
    hist_mask = hist_tensor[:, -1]  # last channel is mask
    hist_tensor = hist_tensor.unsqueeze(0)
    hist_mask = hist_mask.unsqueeze(0)

    pids = list(view.stacks.keys())
    hero_stack = int(view.stacks.get(view.me, 0))
    opp_pids = view.opponents or []
    acting_pids = acting_opponents_for(view)
    n_opp = max(1, len(acting_pids))
    opp_stacks = [
        int(view.stacks.get(o, 0))
        for o in acting_pids
        if int(view.stacks.get(o, 0)) > 0
    ]

    opp_features = torch.zeros(1, _MAX_OPPONENTS, _OPP_FEAT_DIM)
    opp_mask = torch.zeros(1, _MAX_OPPONENTS)
    acting = set(getattr(view, "acting_opponents", None) or [
        opid for opid in opp_pids if int(view.stacks.get(opid, 0)) > 0
    ])
    all_in = set(getattr(view, "all_in_opponents", None) or [
        opid for opid in opp_pids if int(view.stacks.get(opid, 0)) <= 0
    ])
    # Reconstruct per-opponent street commitments so the committed/pot feature
    # matches training semantics.  Training feeds committed_per_seat; inference
    # previously hardcoded committed=0, a train/inference distribution shift on
    # a live network input.  Fall back to 0 if reconstruction is unavailable.
    opp_committed: Dict[str, int] = {}
    try:
        from bots.cfr_bot import _reconstruct_contributions_from_view
        street_committed, _total, _real = _reconstruct_contributions_from_view(view)
        vpids = list(view.stacks.keys())
        opp_committed = {
            pid: int(street_committed[i])
            for i, pid in enumerate(vpids)
            if i < len(street_committed)
        }
    except Exception:
        opp_committed = {}
    for i, opid in enumerate(opp_pids[:_MAX_OPPONENTS]):
        stack = int(view.stacks.get(opid, 0))
        _fill_public_opp_features(
            opp_features,
            opp_mask,
            i,
            stack=stack,
            committed=opp_committed.get(opid, 0),
            pot=view.pot,
            can_act=opid in acting,
            all_in=opid in all_in or stack <= 0,
        )

    scalars = _build_scalars(
        view.pot, view.to_call, hero_stack, view.position,
        view.street, n_opp, opp_stacks,
    ).unsqueeze(0)

    return {
        "card_ids": card_ids,
        "history": hist_tensor,
        "history_mask": hist_mask,
        "opp_features": opp_features,
        "opp_mask": opp_mask,
        "scalars": scalars,
    }


def _build_random_synthetic_input(config: DeepCFRConfig, batch_size: int = 1) -> dict:
    """Build random synthetic inputs for tensor shape and overfit tests."""
    card_ids = torch.randint(0, 52, (batch_size, 7))
    history = torch.randn(batch_size, _HISTORY_MAX_LEN, HIST_FEATURE_DIM)
    hist_mask = torch.zeros(batch_size, _HISTORY_MAX_LEN)
    hist_mask[:, :10] = 1.0
    opp_features = torch.rand(batch_size, _MAX_OPPONENTS, _OPP_FEAT_DIM)
    opp_mask = torch.zeros(batch_size, _MAX_OPPONENTS)
    opp_mask[:, :3] = 1.0
    scalars = torch.rand(batch_size, _SCALAR_DIM)
    return {
        "card_ids": card_ids,
        "history": history,
        "history_mask": hist_mask,
        "opp_features": opp_features,
        "opp_mask": opp_mask,
        "scalars": scalars,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ReservoirBuffer (Gate 3 training data collection)
# ═══════════════════════════════════════════════════════════════════════════════

class ReservoirBuffer:
    """Fixed-capacity buffer with uniform random replacement."""

    def __init__(self, capacity: int = 1_000_000):
        self.capacity = capacity
        self.buffer: list = []
        self._count = 0

    def add(self, item):
        self._count += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            idx = random.randint(0, self._count - 1)
            if idx < self.capacity:
                self.buffer[idx] = item

    def sample(self, n: int) -> list:
        return random.sample(self.buffer, min(n, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════════════════
#  Action mapping helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _preflop_raise_target(label: str, big_blind: int | None, min_total: int) -> int:
    """Map abstract raise buckets to BB-based preflop total raise sizes."""
    bb = max(1, int(round(big_blind or max(min_total / 2, 1))))
    mult = {
        "bet_33": 2.5,
        "bet_50": 3.0,
        "bet_67": 4.0,
        "bet_75": 5.0,
        "bet_100": 6.0,
    }.get(label, 3.0)
    return int(round(mult * bb))


def _legal_abstract_actions(
    legal: list,
    pot: int,
    *,
    street: str | None = None,
    big_blind: int | None = None,
) -> List[int]:
    """Map engine legal actions to abstract action indices."""
    types = {a["type"] for a in legal}
    result = []
    if "fold" in types:
        result.append(0)
    if "check" in types or "call" in types:
        result.append(1)
    if "bet" in types or "raise" in types:
        spec = next(a for a in legal if a["type"] in ("bet", "raise"))
        lo, hi = spec["min"], spec["max"]
        if spec.get("all_in") or lo == hi:
            result.append(7)
            return sorted(set(result))
        if street == "preflop" and spec["type"] == "raise":
            sizes = {
                idx: _preflop_raise_target(ABSTRACT_ACTIONS[idx], big_blind, lo)
                for idx in (2, 3, 4, 5, 6)
            }
            sizes[7] = hi
        else:
            sizes = {2: int(pot * 0.33), 3: int(pot * 0.50), 4: int(pot * 0.67),
                     5: int(pot * 0.75), 6: int(pot * 1.00), 7: hi}
        seen = set()
        for idx, target in sizes.items():
            clamped = max(lo, min(hi, target))
            if clamped not in seen:
                seen.add(clamped)
                result.append(idx)
    return sorted(set(result)) if result else [1]


def _abstract_to_concrete(
    abstract_idx: int,
    legal: list,
    pot: int,
    sizing_frac: float = None,
    *,
    street: str | None = None,
    big_blind: int | None = None,
) -> Action:
    """Convert abstract action index to concrete Action."""
    types = {a["type"] for a in legal}
    label = ABSTRACT_ACTIONS[abstract_idx]

    if label == "fold":
        return Action("fold") if "fold" in types else _passive(legal)
    if label == "check_call":
        if "check" in types:
            return Action("check")
        if "call" in types:
            return Action("call")
        return _passive(legal)

    frac_map = {"bet_33": 0.33, "bet_50": 0.50, "bet_67": 0.67,
                "bet_75": 0.75, "bet_100": 1.00, "all_in": None}
    frac = frac_map.get(label)

    bet_raise = [a for a in legal if a["type"] in ("bet", "raise")]
    if not bet_raise:
        return _passive(legal)
    spec = bet_raise[0]
    lo, hi = spec["min"], spec["max"]

    if label == "all_in":
        amt = hi
    elif street == "preflop" and spec["type"] == "raise":
        amt = _preflop_raise_target(label, big_blind, lo)
    elif sizing_frac is not None and frac is not None:
        # Refine WITHIN the selected bucket.  The network's sizing head emits a
        # single scalar per state, so using it to fully override the bucket
        # fraction collapsed every bet bucket to the same size (bet_33 and
        # bet_100 became identical), decoupling execution from the
        # regret-matched policy.  Clamp the prediction to a band around the
        # bucket's own fraction so distinct buckets stay distinct while the
        # head can still nudge the size within its category.
        lo_frac, hi_frac = frac * 0.75, frac * 1.25
        refined = min(max(sizing_frac, lo_frac), hi_frac)
        amt = int(refined * max(pot, 1))
    else:
        amt = int((frac or 0.5) * pot)
    amt = max(lo, min(hi, amt))
    return Action(spec["type"], amt)


def _passive(legal):
    for t in ("check", "call", "fold"):
        if any(a["type"] == t for a in legal):
            return Action(t)
    a = legal[0]
    return Action(a["type"], a.get("min"))


def _infer_big_blind(view: PlayerView) -> int:
    """Recover the big blind from PlayerView history when possible.

    Standard blind posting means the first preflop decision usually has
    pot_before = SB + BB = 1.5 * BB.
    """
    history = view.history or []
    to_call = int(view.to_call or 0)
    min_raise = int(view.min_raise or 0)
    for entry in history:
        if entry.get("street") == "preflop":
            pot_before = int(entry.get("pot_before", 0) or 0)
            if pot_before > 0:
                return max(1, round(pot_before / 1.5))
            break
    if view.street == "preflop":
        pot = int(view.pot or 0)
        if pot > 0:
            return max(1, round(pot / 1.5))
    if to_call > 0 and min_raise > to_call:
        return max(1, min_raise - to_call)
    return max(1, min_raise or 10)


# ═══════════════════════════════════════════════════════════════════════════════
#  Lightweight search/CFR game state
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class _DeepCFRGameState:
    pot: int
    stacks: List[int]
    committed_per_seat: List[int]
    alive: List[bool]
    street: str
    board: List[tuple]
    hole_cards: Dict[int, tuple]
    seat_order: List[int]
    action_idx: int
    history_events: List[ActionEvent]
    deck_remaining: List[tuple]
    big_blind: int = 10
    ring_order: List[int] = field(default_factory=list)
    street_actions: int = 0
    # Cumulative per-seat contributions across all streets (mirrors
    # engine.py's total_contrib). committed_per_seat is per-street and
    # gets reset at advance_street; this one accumulates.
    total_committed_per_seat: List[int] = field(default_factory=list)
    last_raise_size: int = 0
    raise_blocked: set = field(default_factory=set)
    acted: set = field(default_factory=set)

    def __post_init__(self):
        if not self.total_committed_per_seat:
            self.total_committed_per_seat = list(self.committed_per_seat)
        if self.last_raise_size <= 0:
            self.last_raise_size = max(1, int(self.big_blind or 1))
        self.raise_blocked = set(self.raise_blocked or [])
        if not self.acted and self.action_idx > 0:
            self.acted = set(self.seat_order[:self.action_idx])
        else:
            self.acted = set(self.acted or [])

    def is_terminal(self) -> bool:
        alive_count = sum(1 for ok in self.alive if ok)
        if alive_count <= 1:
            return True
        if self.street == "river" and self.action_idx >= len(self.seat_order):
            return True
        # Soft cap to bound tree depth per street.
        if self.street_actions >= 50:
            return True
        return False

    def is_chance_node(self) -> bool:
        return (not self.is_terminal()) and self.action_idx >= len(self.seat_order)

    def seat_to_act(self) -> int:
        if self.action_idx >= len(self.seat_order):
            return -1
        return self.seat_order[self.action_idx]

    def to_call_for(self, seat: int) -> int:
        if seat < 0 or seat >= len(self.committed_per_seat):
            return 0
        live_committed = [
            self.committed_per_seat[i]
            for i, ok in enumerate(self.alive)
            if ok
        ]
        current_bet = max(live_committed) if live_committed else 0
        return max(0, current_bet - self.committed_per_seat[seat])

    def _active_seats(self) -> List[int]:
        ring = self.ring_order if self.ring_order else list(range(len(self.stacks)))
        return [
            i for i in ring
            if self.alive[i] and self.stacks[i] > 0
        ]

    def _ordered_after(self, seat: int, candidates) -> List[int]:
        candidates = list(candidates)
        if not candidates:
            return []
        ring = self.ring_order if self.ring_order else list(range(len(self.stacks)))
        if seat not in ring:
            return candidates
        seat_pos = ring.index(seat)
        order = ring[seat_pos + 1:] + ring[:seat_pos]
        allowed = set(candidates)
        return [i for i in order if i in allowed]

    def legal_actions(self) -> List[dict]:
        seat = self.seat_to_act()
        if seat < 0 or not self.alive[seat] or self.stacks[seat] <= 0:
            return [{"type": "check"}]

        current_bet = max(
            (self.committed_per_seat[i] for i, ok in enumerate(self.alive) if ok),
            default=0,
        )
        to_call = max(0, current_bet - self.committed_per_seat[seat])
        stack = self.stacks[seat]
        can_raise = seat not in self.raise_blocked

        if to_call == 0:
            legal = [{"type": "check"}]
            if current_bet == 0:
                lo = min(self.big_blind, stack)
                hi = stack
                if hi >= lo:
                    legal.append({"type": "bet", "min": lo, "max": hi})
            else:
                if can_raise:
                    min_total = current_bet + self.last_raise_size
                    min_total = max(min_total, current_bet + self.big_blind)
                    max_total = stack + self.committed_per_seat[seat]
                    if max_total >= min_total:
                        legal.append({"type": "raise", "min": min_total, "max": max_total})
                    elif max_total > current_bet:
                        legal.append({
                            "type": "raise",
                            "min": max_total,
                            "max": max_total,
                            "all_in": True,
                            "reopens": False,
                        })
            return legal

        legal = [{"type": "fold"}]
        call_amt = min(stack, to_call)
        if call_amt > 0:
            legal.append({"type": "call"})
        if can_raise and stack > to_call:
            min_total = current_bet + self.last_raise_size
            min_total = max(min_total, current_bet + self.big_blind)
            max_total = stack + self.committed_per_seat[seat]
            if max_total >= min_total:
                legal.append({"type": "raise", "min": min_total, "max": max_total})
            elif max_total > current_bet:
                legal.append({
                    "type": "raise",
                    "min": max_total,
                    "max": max_total,
                    "all_in": True,
                    "reopens": False,
                })
        return legal

    def legal_abstract_actions(self) -> List[int]:
        return _legal_abstract_actions(
            self.legal_actions(),
            self.pot,
            street=self.street,
            big_blind=self.big_blind,
        )

    def apply_action(self, seat: int, abstract_idx: int) -> "_DeepCFRGameState":
        new = _DeepCFRGameState(
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
            history_events=list(self.history_events),
            deck_remaining=list(self.deck_remaining),
            big_blind=self.big_blind,
            ring_order=list(self.ring_order),
            street_actions=self.street_actions,
            last_raise_size=self.last_raise_size,
            raise_blocked=set(self.raise_blocked),
            acted=set(self.acted),
        )

        if seat < 0 or seat >= len(new.stacks) or not new.alive[seat]:
            new.action_idx += 1
            return new

        pre_pot = new.pot
        label = ABSTRACT_ACTIONS[abstract_idx]
        to_call = new.to_call_for(seat)
        action_type = "check"
        event_amount = 0
        reopens_action = False
        current_bet_before = max(
            (new.committed_per_seat[i] for i, ok in enumerate(new.alive) if ok),
            default=0,
        )
        prev_last_raise_size = new.last_raise_size
        acted_before = set(new.acted)
        raise_size = 0

        if label == "fold":
            if to_call > 0:
                new.alive[seat] = False
                action_type = "fold"
            else:
                action_type = "check"
        elif label == "check_call":
            if to_call > 0:
                cost = min(new.stacks[seat], to_call)
                new.stacks[seat] -= cost
                new.committed_per_seat[seat] += cost
                new.total_committed_per_seat[seat] += cost
                new.pot += cost
                action_type = "call"
                event_amount = cost
            else:
                action_type = "check"
        else:
            legal = new.legal_actions()
            bet_raise = [a for a in legal if a["type"] in ("bet", "raise")]
            if bet_raise:
                spec = bet_raise[0]
                lo, hi = spec["min"], spec["max"]
                reopens_action = bool(spec.get("reopens", True))
                frac_map = {
                    "bet_33": 0.33, "bet_50": 0.50, "bet_67": 0.67,
                    "bet_75": 0.75, "bet_100": 1.00,
                }
                if label == "all_in":
                    target_total = hi
                    action_type = "all_in"
                elif new.street == "preflop" and spec["type"] == "raise":
                    target_total = _preflop_raise_target(label, new.big_blind, lo)
                    target_total = max(lo, min(hi, target_total))
                    action_type = spec["type"]
                else:
                    target_total = int(frac_map.get(label, 0.5) * max(new.pot, 1))
                    target_total = max(lo, min(hi, target_total))
                    action_type = spec["type"]

                need = max(0, target_total - new.committed_per_seat[seat])
                need = min(need, new.stacks[seat])
                new.stacks[seat] -= need
                new.committed_per_seat[seat] += need
                new.total_committed_per_seat[seat] += need
                new.pot += need
                event_amount = new.committed_per_seat[seat]
                current_bet_after = max(
                    (new.committed_per_seat[i] for i, ok in enumerate(new.alive) if ok),
                    default=0,
                )
                raise_size = max(0, current_bet_after - current_bet_before)
            elif to_call > 0:
                cost = min(new.stacks[seat], to_call)
                new.stacks[seat] -= cost
                new.committed_per_seat[seat] += cost
                new.total_committed_per_seat[seat] += cost
                new.pot += cost
                action_type = "call"
                event_amount = cost

        new.history_events.append(ActionEvent(
            seat=seat,
            street=new.street,
            action=action_type,
            amount=int(event_amount),
            pot_before=int(pre_pot),
        ))

        if new.stacks[seat] <= 0:
            new.stacks[seat] = 0

        full_raise = (
            label not in ("fold", "check_call")
            and raise_size > 0
            and reopens_action
            and (
                current_bet_before == 0
                or raise_size >= prev_last_raise_size
            )
        )
        new.street_actions += 1
        # Soft cap on tree depth per street; engine has its own 500-iteration
        # safety counter at engine.py:547 — CFR uses 50 to keep recursion bounded.
        if full_raise and new.street_actions < 50:
            new.last_raise_size = raise_size
            new.raise_blocked.clear()
            new.acted = {seat}
            new_bet = max(new.committed_per_seat)
            responders = [
                i for i in new._active_seats()
                if i != seat and new.committed_per_seat[i] < new_bet
            ]
            new.seat_order = new._ordered_after(seat, responders)
            new.action_idx = 0
        elif (
            label not in ("fold", "check_call")
            and raise_size > 0
            and new.street_actions < 50
        ):
            new.raise_blocked.update(acted_before)
            new.acted = acted_before | {seat}
            new_bet = max(new.committed_per_seat)
            responders = [
                i for i in new._active_seats()
                if i != seat and new.committed_per_seat[i] < new_bet
            ]
            new.seat_order = new._ordered_after(seat, responders)
            new.action_idx = 0
        else:
            assert new.action_idx < len(new.seat_order) and new.seat_order[new.action_idx] == seat, (
                f"apply_action expected seat {seat} at action_idx {new.action_idx} "
                f"in seat_order {new.seat_order!r}"
            )
            new.acted.add(seat)
            new.action_idx += 1
        return new

    def advance_street(self) -> "_DeepCFRGameState":
        if self.street not in _NEXT_STREET:
            return _DeepCFRGameState(
                pot=self.pot,
                stacks=list(self.stacks),
                committed_per_seat=list(self.committed_per_seat),
                total_committed_per_seat=list(self.total_committed_per_seat),
                alive=list(self.alive),
                street=self.street,
                board=list(self.board),
                hole_cards=dict(self.hole_cards),
                seat_order=list(self.seat_order),
                action_idx=len(self.seat_order),
                history_events=list(self.history_events),
                deck_remaining=list(self.deck_remaining),
                big_blind=self.big_blind,
                ring_order=list(self.ring_order),
                street_actions=self.street_actions,
                last_raise_size=self.last_raise_size,
                raise_blocked=set(self.raise_blocked),
                acted=set(self.acted),
            )

        next_street = _NEXT_STREET[self.street]
        n_cards = min(_DEAL_ON_ADVANCE[self.street], len(self.deck_remaining))
        deck = list(self.deck_remaining)
        dealt = random.sample(deck, n_cards) if n_cards > 0 else []
        dealt_set = set(dealt)
        deck = [c for c in deck if c not in dealt_set]
        ring = self.ring_order if self.ring_order else list(range(len(self.stacks)))
        alive_order = [
            i for i in street_action_order(next_street, ring)
            if self.alive[i] and self.stacks[i] > 0
        ]

        return _DeepCFRGameState(
            pot=self.pot,
            stacks=list(self.stacks),
            # Mirror engine.py L430: reset per-seat contributions for the new
            # street so preflop blinds/bets do not leak into postflop to_call.
            # total_committed_per_seat is NOT reset — it tracks cumulative
            # contributions across all streets for AIVAT side-pot settlement.
            committed_per_seat=[0] * len(self.stacks),
            total_committed_per_seat=list(self.total_committed_per_seat),
            alive=list(self.alive),
            street=next_street,
            board=list(self.board) + dealt,
            hole_cards=dict(self.hole_cards),
            seat_order=alive_order,
            action_idx=0,
            history_events=list(self.history_events),
            deck_remaining=deck,
            big_blind=self.big_blind,
            ring_order=list(self.ring_order),
            street_actions=0,
            last_raise_size=self.big_blind,
            raise_blocked=set(),
            acted=set(),
        )

    def to_network_input(self, hero_seat: int) -> dict:
        hole = list(self.hole_cards.get(hero_seat, ()))
        card_ids = _build_card_ids(hole, self.board).unsqueeze(0)

        hist_tensor = history_to_tensor(self.history_events, max_len=_HISTORY_MAX_LEN)
        hist_mask = hist_tensor[:, -1]

        opp_features = torch.zeros(1, _MAX_OPPONENTS, _OPP_FEAT_DIM)
        opp_mask = torch.zeros(1, _MAX_OPPONENTS)
        opp_seats = [
            i for i, ok in enumerate(self.alive)
            if ok and i != hero_seat
        ]
        for i, seat in enumerate(opp_seats[:_MAX_OPPONENTS]):
            stack = self.stacks[seat]
            _fill_public_opp_features(
                opp_features,
                opp_mask,
                i,
                stack=stack,
                committed=self.committed_per_seat[seat],
                pot=self.pot,
                can_act=stack > 0,
                all_in=stack <= 0,
            )

        opp_stacks = [self.stacks[i] for i in opp_seats if self.stacks[i] > 0]
        scalars = _build_scalars(
            self.pot,
            self.to_call_for(hero_seat),
            self.stacks[hero_seat] if 0 <= hero_seat < len(self.stacks) else 0,
            _seat_position_label(hero_seat, len(self.stacks)),
            self.street,
            max(1, len(opp_seats)),
            opp_stacks,
        ).unsqueeze(0)

        return {
            "card_ids": card_ids,
            "history": hist_tensor.unsqueeze(0),
            "history_mask": hist_mask.unsqueeze(0),
            "opp_features": opp_features,
            "opp_mask": opp_mask,
            "scalars": scalars,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  DeepCFRBot
# ═══════════════════════════════════════════════════════════════════════════════

class DeepCFRBot:
    """Path B Deep CFR Plus bot. Implements core.bot_api.BotAdapter."""

    _DEFAULT_SEARCH_DEPTH = 4
    _MAX_CFR_DEPTH = 8
    _ALL_IN_WARMUP_ITERATIONS = 100_000
    _ALL_IN_DEPLOY_ITERATION = 150_000
    _ALL_IN_FULL_RELEASE_ITERATION = 350_000
    _ALL_IN_DEEP_STACK_SPR = 4.0
    _ALL_IN_STAGED_CAP_START = 0.05
    _ALL_IN_STAGED_CAP_END = 0.15
    _SEARCH_TEMPERATURE = 20.0
    _SEARCH_ADVANTAGE_CLIP = 2.0
    _SEARCH_BLEND = 0.25

    def __init__(self, config: DeepCFRConfig | None = None,
                 weights_path: str | None = None,
                 search_depth: int | None = None,
                 inference_mode: bool = True,
                 aivat_sims: int = 500):
        loaded_state = None
        loaded_config = None
        loaded_iteration = 0
        loaded_all_in_warmup = self._ALL_IN_WARMUP_ITERATIONS
        loaded_all_in_deploy = self._ALL_IN_DEPLOY_ITERATION
        loaded_all_in_full_release = self._ALL_IN_FULL_RELEASE_ITERATION
        if weights_path:
            if not os.path.exists(weights_path):
                raise RuntimeError(
                    f"[DeepCFRBot] Missing weights {weights_path!r}. "
                    "In inference_mode, a trained Deep CFR checkpoint is "
                    "required; train one or pass deep_cfr:<path> to an "
                    "existing checkpoint."
                )
            loaded = torch.load(weights_path, map_location="cpu", weights_only=False)
            if isinstance(loaded, dict) and "network_state_dict" in loaded:
                loaded_state = loaded["network_state_dict"]
                loaded_config = loaded.get("config")
                loaded_iteration = int(loaded.get("iteration", 0) or 0)
                loaded_all_in_warmup = int(
                    loaded.get("all_in_warmup_iterations", loaded_all_in_warmup)
                    or loaded_all_in_warmup
                )
                loaded_all_in_deploy = int(
                    loaded.get("all_in_deploy_iteration", loaded_all_in_deploy)
                    or loaded_all_in_deploy
                )
                loaded_all_in_full_release = int(
                    loaded.get(
                        "all_in_full_release_iteration",
                        loaded_all_in_full_release,
                    )
                    or loaded_all_in_full_release
                )
            else:
                loaded_state = loaded
                loaded_config = self._infer_config_from_state_dict(loaded_state)

        self.config = loaded_config or config or DeepCFRConfig.large()
        self.aivat_sims = int(aivat_sims)
        self.network = DeepCFRNetwork(self.config)

        # Device placement
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        self.network = self.network.to(self._device)

        self.search_depth = search_depth or self._DEFAULT_SEARCH_DEPTH
        self.inference_mode = inference_mode
        self.training_iteration = loaded_iteration
        self.all_in_warmup_iterations = loaded_all_in_warmup
        self.all_in_deploy_iteration = loaded_all_in_deploy
        self.all_in_full_release_iteration = loaded_all_in_full_release
        self._weights_loaded = False
        self._all_in_guardrails_disabled = False

        if loaded_state is not None:
            self.network.load_state_dict(loaded_state)
            self._weights_loaded = True
        self.network.eval()

        if self._weights_loaded:
            print(f"[DeepCFRBot] Loaded weights from {weights_path} "
                  f"(iteration {self.training_iteration})")

        self._opp_stats: Optional[OpponentStatTracker] = None
        self._last_history_len = 0
        self._last_history_snapshot = []
        self._last_hand_id = None
        self._search_leaf_calls = 0
        self._subgame_search_calls = 0
        self._recursion_calls = 0

        # Per-instance RNG for decision sampling. Seeded from the global so
        # `random.seed(N)` at script start still cascades into reproducibility,
        # but multiple bots in one process have independent decision streams
        # (vs. all coupling through the module-level random state).
        self._rng = random.Random(random.getrandbits(64))

        # Gate 3 buffers
        self.regret_buffer = ReservoirBuffer()
        self.value_buffer = ReservoirBuffer()
        self.sizing_buffer = ReservoirBuffer()

    def _warmup_all_in_mask_active(self) -> bool:
        """Whether loaded checkpoint inference should hide untrained all-in."""
        return self._inference_all_in_mask_active()

    def _inference_all_in_mask_active(self) -> bool:
        """Whether tournament inference should hide all-in for early checkpoints."""
        return (
            not self._all_in_guardrails_disabled and
            self._weights_loaded and
            self.training_iteration < self.all_in_deploy_iteration
        )

    @staticmethod
    def _mask_all_in(legal_mask: List[int]) -> List[int]:
        all_in_idx = ABSTRACT_ACTIONS.index("all_in")
        masked = [a for a in legal_mask if a != all_in_idx]
        return masked or legal_mask

    def _training_policy_legal_mask(
        self,
        legal_mask: List[int],
        all_in_policy_probability: float,
    ) -> List[int]:
        """Stochastically expose all-in to self-play while keeping targets raw."""
        all_in_idx = ABSTRACT_ACTIONS.index("all_in")
        if all_in_idx not in legal_mask:
            return legal_mask
        prob = max(0.0, min(1.0, float(all_in_policy_probability)))
        if prob >= 1.0:
            return legal_mask
        if prob <= 0.0 or self._rng.random() >= prob:
            return self._mask_all_in(legal_mask)
        return legal_mask

    def _all_in_release_progress(self) -> float:
        start = max(0, int(self.all_in_deploy_iteration))
        end = max(start + 1, int(self.all_in_full_release_iteration))
        return max(0.0, min(1.0, (self.training_iteration - start) / (end - start)))

    def _deep_stack_all_in_cap(self, state: PlayerView) -> float | None:
        """Return the staged inference all-in cap for deep-stack spots."""
        if self._all_in_guardrails_disabled:
            return None
        if not (self._weights_loaded and self.training_iteration < self.all_in_full_release_iteration):
            return None
        hero_stack = int((state.stacks or {}).get(state.me, 0) or 0)
        spr = hero_stack / max(int(state.pot or 0), 1)
        if spr < self._ALL_IN_DEEP_STACK_SPR:
            return None
        progress = self._all_in_release_progress()
        return (
            self._ALL_IN_STAGED_CAP_START +
            (self._ALL_IN_STAGED_CAP_END - self._ALL_IN_STAGED_CAP_START) * progress
        )

    @staticmethod
    def _renormalize_strategy(strategy: List[float], legal_mask: List[int]) -> List[float]:
        total = sum(max(strategy[a], 0.0) for a in legal_mask)
        if total > 0:
            for a in legal_mask:
                strategy[a] = max(strategy[a], 0.0) / total
        else:
            n = max(1, len(legal_mask))
            for a in legal_mask:
                strategy[a] = 1.0 / n
        return strategy

    def _cap_all_in_probability(
        self,
        strategy: List[float],
        legal_mask: List[int],
        state: PlayerView,
    ) -> List[float]:
        """Cap all-in probability during staged deployment and redistribute excess."""
        all_in_idx = ABSTRACT_ACTIONS.index("all_in")
        if all_in_idx not in legal_mask:
            return strategy
        cap = self._deep_stack_all_in_cap(state)
        if cap is None or strategy[all_in_idx] <= cap:
            return strategy

        capped = list(strategy)
        excess = max(0.0, capped[all_in_idx] - cap)
        capped[all_in_idx] = cap
        others = [a for a in legal_mask if a != all_in_idx]
        other_total = sum(max(capped[a], 0.0) for a in others)
        if others:
            if other_total > 0:
                for a in others:
                    capped[a] += excess * (max(capped[a], 0.0) / other_total)
            else:
                share = excess / len(others)
                for a in others:
                    capped[a] += share
        return self._renormalize_strategy(capped, legal_mask)

    def detox_all_in_regret_output(self, bias: float = -2.0) -> tuple[str, int]:
        """Reset only the all-in output row of the regret head."""
        all_in_idx = ABSTRACT_ACTIONS.index("all_in")
        final_linear = None
        final_name = ""
        for name, module in self.network.regret_head.mlp.named_modules():
            if isinstance(module, nn.Linear):
                final_linear = module
                final_name = name
        if final_linear is None:
            raise RuntimeError("regret head has no Linear output layer")
        with torch.no_grad():
            final_linear.weight[all_in_idx].zero_()
            if final_linear.bias is not None:
                final_linear.bias[all_in_idx] = float(bias)
        return f"regret_head.mlp.{final_name}", all_in_idx

    @staticmethod
    def _infer_config_from_state_dict(state_dict) -> DeepCFRConfig | None:
        """Infer architecture config from a raw network state_dict if possible."""
        if not isinstance(state_dict, dict):
            return None
        try:
            card_embed_dim = int(state_dict["encoder.card_embed.weight"].shape[1])
            gru_hidden = int(state_dict["encoder.history_gru.weight_hh_l0"].shape[1])
            opp_embed_dim = int(state_dict["encoder.opp_proj.0.weight"].shape[0])
            state_dim = int(state_dict["encoder.fuse.0.weight"].shape[0])
            head_hidden = int(state_dict["regret_head.mlp.0.weight"].shape[0])
            head_layers = 3 if "regret_head.mlp.6.weight" in state_dict else 2
            return DeepCFRConfig(
                card_embed_dim=card_embed_dim,
                gru_hidden=gru_hidden,
                opp_embed_dim=opp_embed_dim,
                state_dim=state_dim,
                head_hidden=head_hidden,
                head_layers=head_layers,
            )
        except (KeyError, IndexError, AttributeError, TypeError):
            return None

    def _detect_hand_boundary(self, view: PlayerView) -> bool:
        """Return True when PlayerView.history represents a new hand."""
        hand_id = getattr(view, "hand_id", None)
        if hand_id is not None:
            return self._last_hand_id is not None and hand_id != self._last_hand_id

        current = view.history or []
        if not current:
            return self._last_history_snapshot != []
        if len(current) < self._last_history_len:
            return True
        if current[:self._last_history_len] != self._last_history_snapshot:
            return True
        return False

    def act(self, state: PlayerView) -> Action:
        """Choose an action for the current game state."""
        hole = state.hole_cards
        board = state.board
        pot = state.pot
        legal = state.legal_actions
        history = state.history or []

        if not hole or len(hole) < 2:
            return _passive(legal)

        # Lazy-construct opponent tracker
        n_seats = _tracker_size_for_view(state)
        if self._opp_stats is None:
            self._opp_stats = OpponentStatTracker(n_seats=n_seats, window=50)
            self._last_history_len = 0
            self._last_history_snapshot = []
        else:
            self._opp_stats.ensure_n_seats(n_seats)

        if self._detect_hand_boundary(state):
            self._opp_stats.observe_hand_end(seats_to_showdown=[])
            self._last_history_len = 0
            self._last_history_snapshot = []

        pids = list(state.stacks.keys())
        for entry in history[self._last_history_len:]:
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

        # Build network input and forward pass
        batch = build_network_input(state, self._opp_stats)

        with torch.no_grad():
            out = self.network(batch)

        regret_logits = out["regret"][0].cpu()
        sizing_val = out["sizing"].item() if out["sizing"].dim() == 0 else out["sizing"][0].item()

        big_blind = _infer_big_blind(state)

        # Legal mask
        legal_mask = _legal_abstract_actions(
            legal,
            pot,
            street=state.street,
            big_blind=big_blind,
        )
        if self._inference_all_in_mask_active():
            legal_mask = self._mask_all_in(legal_mask)

        # Regret matching
        strategy = self._regret_match(regret_logits, legal_mask)

        # Real-time search (only with loaded weights)
        if self.inference_mode and self._weights_loaded:
            t0 = _time.monotonic()
            strategy = self._subgame_search(state, strategy, legal_mask,
                                            depth=self.search_depth)
            elapsed = _time.monotonic() - t0
            if elapsed > 5.0:
                warnings.warn(
                    f"[DeepCFRBot] _subgame_search took {elapsed:.2f}s "
                    f"(budget: 2s at depth={self.search_depth})")
        strategy = self._cap_all_in_probability(strategy, legal_mask, state)

        # Sample action
        abstract_idx = self._sample_action(strategy, legal_mask)

        # Sizing head for bet/raise actions
        if ABSTRACT_ACTIONS[abstract_idx] in ("bet_33", "bet_50", "bet_67",
                                               "bet_75", "bet_100"):
            return _abstract_to_concrete(abstract_idx, legal, pot,
                                         sizing_frac=sizing_val,
                                         street=state.street,
                                         big_blind=big_blind)
        return _abstract_to_concrete(
            abstract_idx, legal, pot, street=state.street, big_blind=big_blind)

    def _regret_match(self, regret_logits: torch.Tensor,
                      legal_mask: List[int]) -> List[float]:
        strategy = [0.0] * NUM_ACTIONS
        pos_sum = 0.0
        for a in legal_mask:
            raw = regret_logits[a]
            val = raw.item() if hasattr(raw, "item") else raw
            v = max(0.0, float(val))
            strategy[a] = v
            pos_sum += v
        if pos_sum > 0:
            for a in legal_mask:
                strategy[a] /= pos_sum
        else:
            n = len(legal_mask)
            for a in legal_mask:
                strategy[a] = 1.0 / n
        return strategy

    def _sample_action(self, strategy: List[float],
                       legal_mask: List[int]) -> int:
        probs = [strategy[a] for a in legal_mask]
        total = sum(probs)
        if total <= 0:
            return self._rng.choice(legal_mask)
        r = self._rng.random() * total
        cum = 0.0
        for a, p in zip(legal_mask, probs):
            cum += p
            if r <= cum:
                return a
        return legal_mask[-1]

    def _sample_action_idx(self, strategy: List[float],
                           legal_mask: List[int]) -> int:
        return self._sample_action(strategy, legal_mask)

    # ── Real-time search ──────────────────────────────────────────────────────

    def _build_search_game_state(self, view: PlayerView) -> Tuple[_DeepCFRGameState, int]:
        """Build a lightweight game state from the engine's PlayerView."""
        from bots.cfr_bot import (
            _infer_last_raise_size_from_view,
            _reconstruct_contributions_from_view,
        )

        pids = list(view.stacks.keys())
        hero_seat = pids.index(view.me) if view.me in pids else 0
        stacks = [int(view.stacks.get(pid, 0)) for pid in pids]
        opp_set = set(view.opponents or [])
        alive = [
            stacks[i] >= 0 and (pids[i] == view.me or pids[i] in opp_set)
            for i in range(len(pids))
        ]
        alive[hero_seat] = stacks[hero_seat] >= 0

        active = [i for i, ok in enumerate(alive) if ok]
        big_blind = _infer_big_blind(view)
        last_raise_size = _infer_last_raise_size_from_view(view, big_blind)
        committed, total_committed, real = _reconstruct_contributions_from_view(view)
        if not committed or len(committed) != len(pids):
            committed = [0] * len(pids)
            total_committed = [0] * len(pids)
            real = False

        desired_to_call = max(0, int(view.to_call or 0))
        legal = view.legal_actions or []
        legal_types = {a.get("type") for a in legal}
        legal_raise = next((a for a in legal if a.get("type") == "raise"), None)
        current_bet = max((committed[i] for i in active), default=0)

        if desired_to_call == 0 and "bet" in legal_types:
            committed = [0] * len(pids)
            current_bet = 0
        elif desired_to_call > 0:
            if current_bet < desired_to_call:
                current_bet = desired_to_call
            committed[hero_seat] = max(0, current_bet - desired_to_call)
            if max((committed[i] for i in active if i != hero_seat), default=0) < current_bet:
                for i in active:
                    if i != hero_seat:
                        committed[i] = current_bet
                        break
            if legal_raise is not None:
                min_total = int(legal_raise.get("min") or 0)
                if min_total > current_bet:
                    last_raise_size = max(1, min_total - current_bet)
        elif current_bet == 0 and legal_raise is not None and "bet" not in legal_types:
            min_total = int(legal_raise.get("min") or 0)
            current_bet = max(1, min_total - last_raise_size)
            for i in active:
                committed[i] = current_bet
        elif legal_raise is not None and current_bet > 0:
            min_total = int(legal_raise.get("min") or 0)
            if min_total > current_bet:
                last_raise_size = max(1, min_total - current_bet)

        if not real:
            total_committed = list(committed)

        used_cards = list(view.hole_cards or []) + list(view.board or [])
        deck = [c for c in _FULL_DECK if c not in used_cards]
        hole_cards: Dict[int, tuple] = {hero_seat: tuple(view.hole_cards or [])}
        for i in active:
            if i == hero_seat:
                continue
            if len(deck) >= 2:
                dealt = random.sample(deck, 2)
                hole_cards[i] = tuple(dealt)
                dealt_set = set(dealt)
                deck = [c for c in deck if c not in dealt_set]

        events = _safe_extract_history(view)

        after_hero = [
            i for i in active
            if i != hero_seat and stacks[i] > 0
        ]
        seat_order = [hero_seat] + after_hero

        state = _DeepCFRGameState(
            pot=int(view.pot),
            stacks=stacks,
            committed_per_seat=committed,
            alive=alive,
            street=view.street,
            board=list(view.board or []),
            hole_cards=hole_cards,
            seat_order=seat_order,
            action_idx=0,
            history_events=events,
            deck_remaining=deck,
            big_blind=big_blind,
            ring_order=list(range(len(pids))),
            total_committed_per_seat=total_committed,
            last_raise_size=last_raise_size,
        )
        return state, hero_seat

    def _subgame_search(self, state: PlayerView, prior: List[float],
                        legal_mask: List[int], depth: int) -> List[float]:
        """Real depth-limited subgame search using _search_subtree."""
        self._subgame_search_calls += 1
        if depth <= 0 or not legal_mask:
            return prior

        if self._inference_all_in_mask_active():
            legal_mask = self._mask_all_in(legal_mask)

        self._search_leaf_calls = 0
        game_state, hero_seat = self._build_search_game_state(state)
        values = {}
        for a_idx in legal_mask:
            before_commit = game_state.committed_per_seat[hero_seat]
            next_state = game_state.apply_action(hero_seat, a_idx)
            added_cost = max(0, next_state.committed_per_seat[hero_seat] - before_commit)
            added_cost_units = added_cost / max(game_state.big_blind, 1)
            values[a_idx] = (
                self._search_subtree(next_state, hero_seat, depth - 1)
                - added_cost_units
            )

        # Refine cautiously: the value head is learned and may be badly
        # calibrated early in training. Use clipped, temperature-scaled
        # advantages and blend them back into the regret-matched prior so
        # search cannot deterministically override policy on noisy values.
        prior_total = sum(max(prior[a], 0.0) for a in legal_mask)
        if prior_total > 0:
            prior_norm = {a: max(prior[a], 0.0) / prior_total for a in legal_mask}
        else:
            prior_norm = {a: 1.0 / len(legal_mask) for a in legal_mask}

        baseline = sum(prior_norm[a] * values.get(a, 0.0) for a in legal_mask)
        refined = [0.0] * NUM_ACTIONS
        total = 0.0
        for a in legal_mask:
            adv = (values.get(a, 0.0) - baseline) / self._SEARCH_TEMPERATURE
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

    def _search_subtree(self, state: _DeepCFRGameState, hero_seat: int,
                        depth: int) -> float:
        """Recursive depth-limited subgame evaluation for hero_seat."""
        if state.is_terminal() or depth <= 0:
            batch = state.to_network_input(hero_seat)
            with torch.no_grad():
                out = self.network(batch)
            self._search_leaf_calls += 1
            return out["value"].item()

        if state.is_chance_node():
            return self._search_subtree(state.advance_street(), hero_seat, depth - 1)

        seat = state.seat_to_act()
        legal_mask = state.legal_abstract_actions()
        if self._inference_all_in_mask_active():
            legal_mask = self._mask_all_in(legal_mask)
        if not legal_mask:
            return self._search_subtree(state.advance_street(), hero_seat, depth - 1)

        if seat != hero_seat:
            batch = state.to_network_input(seat)
            with torch.no_grad():
                out = self.network(batch)
            strategy = self._regret_match(out["regret"][0].cpu(), legal_mask)
            total = 0.0
            for a in legal_mask:
                if strategy[a] <= 1e-9:
                    continue
                next_state = state.apply_action(seat, a)
                total += strategy[a] * self._search_subtree(next_state, hero_seat, depth - 1)
            return total

        best = -float("inf")
        for a in legal_mask:
            before_commit = state.committed_per_seat[hero_seat]
            next_state = state.apply_action(hero_seat, a)
            added_cost = max(0, next_state.committed_per_seat[hero_seat] - before_commit)
            added_cost_units = added_cost / max(state.big_blind, 1)
            val = (
                self._search_subtree(next_state, hero_seat, depth - 1)
                - added_cost_units
            )
            best = max(best, val)
        return best if math.isfinite(best) else 0.0

    # ── Tree CFR with target collection (Gate 3B) ──────────────────────────

    def _cfr_recurse(
        self,
        state: _DeepCFRGameState,
        hero_seat: int,
        depth: int,
        *,
        iteration: int = 0,
        regret_buf: Optional[ReservoirBuffer] = None,
        value_buf: Optional[ReservoirBuffer] = None,
        sizing_buf: Optional[ReservoirBuffer] = None,
        exploration_epsilon: float = 0.0,
        allow_all_in: bool = True,
        all_in_policy_probability: float | None = None,
    ) -> float:
        """Recursive external-sampling CFR traversal.

        When regret_buf/value_buf/sizing_buf are provided, this is a Deep CFR
        training traversal: at each hero decision node, target regrets are
        appended; at leaves, target values are appended; at bet/raise hero
        nodes, target sizings are appended.  ``iteration`` is the outer loop
        index used for CFR+ linear weighting.

        When buffers are None, this is the Gate 2B inference-time / sanity
        traversal — return value only, no side effects.
        """
        self._recursion_calls += 1
        collecting = regret_buf is not None

        # ── Terminal / depth-limit leaf ──
        if state.is_terminal() or depth <= 0:
            if collecting:
                leaf_val = self._aivat_leaf_value(state, hero_seat)
                input_dict = state.to_network_input(hero_seat)
                value_buf.add((input_dict, float(leaf_val)))
                return leaf_val
            else:
                batch = state.to_network_input(hero_seat)
                with torch.no_grad():
                    out = self.network(batch)
                return out["value"].item()

        # ── Chance node ──
        if state.is_chance_node():
            return self._cfr_recurse(
                state.advance_street(), hero_seat, depth - 1,
                iteration=iteration,
                regret_buf=regret_buf, value_buf=value_buf,
                sizing_buf=sizing_buf,
                exploration_epsilon=exploration_epsilon,
                allow_all_in=allow_all_in,
                all_in_policy_probability=all_in_policy_probability,
            )

        seat = state.seat_to_act()
        target_legal_mask = state.legal_abstract_actions()
        if collecting:
            if all_in_policy_probability is None:
                policy_prob = 1.0 if allow_all_in else 0.0
            else:
                policy_prob = all_in_policy_probability
            policy_legal_mask = self._training_policy_legal_mask(
                target_legal_mask,
                policy_prob,
            )
        else:
            policy_legal_mask = target_legal_mask
        if not policy_legal_mask:
            return self._cfr_recurse(
                state.advance_street(), hero_seat, depth - 1,
                iteration=iteration,
                regret_buf=regret_buf, value_buf=value_buf,
                sizing_buf=sizing_buf,
                exploration_epsilon=exploration_epsilon,
                allow_all_in=allow_all_in,
                all_in_policy_probability=all_in_policy_probability,
            )

        # Strategy from network
        batch = state.to_network_input(seat)
        with torch.no_grad():
            out = self.network(batch)
        strategy = self._regret_match(out["regret"][0].cpu(), policy_legal_mask)

        # ── Opponent node: external sampling ──
        if seat != hero_seat:
            if collecting and self._rng.random() < exploration_epsilon:
                action = self._rng.choice(policy_legal_mask)
            else:
                action = self._sample_action_idx(strategy, policy_legal_mask)
            next_state = state.apply_action(seat, action)
            return self._cfr_recurse(
                next_state, hero_seat, depth - 1,
                iteration=iteration,
                regret_buf=regret_buf, value_buf=value_buf,
                sizing_buf=sizing_buf,
                exploration_epsilon=exploration_epsilon,
                allow_all_in=allow_all_in,
                all_in_policy_probability=all_in_policy_probability,
            )

        # ── Hero node: full expansion ──
        action_values = {}
        expansion_legal_mask = target_legal_mask if collecting else policy_legal_mask
        for a in expansion_legal_mask:
            before_commit = state.committed_per_seat[hero_seat]
            next_state = state.apply_action(hero_seat, a)
            added_cost = max(0, next_state.committed_per_seat[hero_seat] - before_commit)
            added_cost_units = added_cost / max(state.big_blind, 1)
            action_values[a] = self._cfr_recurse(
                next_state, hero_seat, depth - 1,
                iteration=iteration,
                regret_buf=regret_buf, value_buf=value_buf,
                sizing_buf=sizing_buf,
                exploration_epsilon=exploration_epsilon,
                allow_all_in=allow_all_in,
                all_in_policy_probability=all_in_policy_probability,
            ) - added_cost_units

        ev = sum(strategy[a] * action_values[a] for a in policy_legal_mask)

        # ── Collect targets ──
        if collecting:
            input_dict = state.to_network_input(hero_seat)

            # Regret target: action_value[a] - EV, with CFR+ linear weight
            regret_vec = torch.zeros(NUM_ACTIONS)
            legal_mask_vec = torch.zeros(NUM_ACTIONS)
            for a in expansion_legal_mask:
                regret_vec[a] = action_values[a] - ev
                legal_mask_vec[a] = 1.0
            weight = float(max(1, iteration))  # CFR+ linear weighting
            regret_buf.add((input_dict, regret_vec, legal_mask_vec, weight))

            # Value target at hero decision node
            value_buf.add((input_dict, float(ev)))

            # Gate 3B sizing-target collection.
            #
            # We collect the BUCKET FRACTION of the best abstract bet/raise
            # action (one of {0.33, 0.50, 0.67, 0.75, 1.00}). The sizing
            # head's nominal output range is [0, 2.0] but training only
            # teaches these specific fractions, so the head learns
            # "bucket-fraction sizing" -- refining within a discrete category
            # -- not truly continuous bet sizing.
            #
            # True continuous sizing (grid-search or line-search over
            # arbitrary fractions) is Gate 4 territory. Gate 3B's sizing head
            # is an architectural scaffold for that future work.
            best_bet_action = None
            best_bet_value = -float("inf")
            _BET_ACTIONS = {"bet_33", "bet_50", "bet_67", "bet_75", "bet_100"}
            for a in expansion_legal_mask:
                label = ABSTRACT_ACTIONS[a]
                if label in _BET_ACTIONS and action_values[a] > best_bet_value:
                    best_bet_value = action_values[a]
                    best_bet_action = a
            if best_bet_action is not None and sizing_buf is not None:
                _FRAC_MAP = {
                    "bet_33": 0.33, "bet_50": 0.50, "bet_67": 0.67,
                    "bet_75": 0.75, "bet_100": 1.00,
                }
                frac = _FRAC_MAP.get(ABSTRACT_ACTIONS[best_bet_action], 0.5)
                sizing_buf.add((input_dict, float(frac)))

        return ev

    def _aivat_leaf_value(self, state: _DeepCFRGameState, hero_seat: int) -> float:
        """Compute AIVAT chip-EV at a leaf node using core.aivat.value().

        Uses full information (all players' hole cards visible) — this is
        training-time only; at inference the value network replaces this.

        Returns value normalized by big blind so network targets stay in a
        sensible range (raw chip values would cause gradient explosion).
        """
        from core.aivat import value as aivat_value, Snapshot

        try:
            alive_tuple = tuple(state.alive)
            stacks_tuple = tuple(state.stacks)
            # Use cumulative contributions (mirrors engine's total_contrib) —
            # AIVAT needs whole-hand totals for correct side-pot settlement,
            # not the per-street snapshot that resets at advance_street.
            committed_tuple = tuple(state.total_committed_per_seat)
            if hero_seat < 0 or hero_seat >= len(alive_tuple):
                raise IndexError(f"hero_seat {hero_seat} outside alive={len(alive_tuple)}")
            if hero_seat >= len(stacks_tuple):
                raise IndexError(f"hero_seat {hero_seat} outside stacks={len(stacks_tuple)}")
            if hero_seat >= len(committed_tuple):
                raise ValueError(
                    f"hero_seat {hero_seat} missing from total_committed_per_seat="
                    f"{committed_tuple}"
                )
            if alive_tuple[hero_seat] and hero_seat not in state.hole_cards:
                raise KeyError(f"missing hole cards for live hero seat {hero_seat}")

            # Build hole_cards dict for Snapshot
            hole_dict = {}
            for seat, cards in state.hole_cards.items():
                hole_dict[seat] = tuple(cards)

            snap = Snapshot(
                hole_cards=hole_dict,
                board=tuple(state.board),
                pot=state.pot,
                stacks=stacks_tuple,
                alive=alive_tuple,
                to_call=state.to_call_for(hero_seat),
                hero_committed=committed_tuple[hero_seat],
                committed_per_seat=committed_tuple,
            )
            raw = aivat_value(
                snap, hero_seat=hero_seat, mode="chip_ev",
                n_sims=self.aivat_sims,
            )
            # Normalize by big blind for stable gradients
            return raw / max(state.big_blind, 1)
        except (KeyError, ValueError, IndexError) as e:
            print(f"[DeepCFRBot] _aivat_leaf_value caught {type(e).__name__}: {e}; "
                  f"returning 0 for this leaf.")
            return 0.0
