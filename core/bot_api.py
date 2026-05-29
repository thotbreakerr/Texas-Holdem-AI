# bot_api.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class Action:
    type: str                # "fold" | "check" | "call" | "bet" | "raise"
    amount: Optional[int] = None

@dataclass
class PlayerView:
    me: str
    street: str              # "preflop" | "flop" | "turn" | "river"
    position: str
    hole_cards: list
    board: list
    pot: int
    to_call: int
    min_raise: int
    max_raise: int
    legal_actions: List[Dict[str, Any]]
    stacks: Dict[str, int]
    opponents: List[str]
    history: List[Dict[str, Any]]
    hand_id: Optional[int] = None
    seat_indices: Optional[Dict[str, int]] = None
    acting_opponents: Optional[List[str]] = None
    all_in_opponents: Optional[List[str]] = None

class BotAdapter:
    def act(self, view: PlayerView) -> Action:
        raise NotImplementedError


def acting_opponents_for(view: PlayerView) -> List[str]:
    """Opponents who are still able to make future betting decisions."""
    explicit = getattr(view, "acting_opponents", None)
    if explicit is not None:
        return list(explicit)
    stacks = getattr(view, "stacks", {}) or {}
    return [
        pid for pid in (getattr(view, "opponents", None) or [])
        if int(stacks.get(pid, 0)) > 0
    ]
