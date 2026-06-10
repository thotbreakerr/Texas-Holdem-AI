"""
core/ml_features.py — Shared 26-feature builder for the supervised ML path
---------------------------------------------------------------------------
Single source of truth for the MLBot feature vector. Used by BOTH:

  * bots/ml_bot.py        — live inference (features from a PlayerView)
  * training/train_ml_bot.py — supervised training (features from a logged
                               decision row written by core/logger.py)

so that a logged decision and the equivalent live game state produce the
SAME 26 numbers. Phase 2 fixes consolidated here:

  1. Preflop hand strength is normalised by PREFLOP_EVAL_MAX (the rank-sum
     heuristic's own max, AA = 2*14 + 40 = 68), not by EVAL_HAND_MAX (the
     5-card table max, ~134M) which collapsed every preflop hand to ~0 on
     the training side.
  2. Position comes from the decision log (engine now writes it); both
     sides share POSITION_ORDER.
  3. Opponent memory (VPIP / aggression / tightness) is defined once in
     OpponentMemory. Checks do NOT count toward VPIP (no money put in).

Memory scope (intentional, documented): CUMULATIVE TOURNAMENT memory.
A live MLBot accumulates opponent stats across every hand it acts in
(one MLBot instance = one tournament). Training replays the same stream:
one OpponentMemory per (log file, hero), fed each hand's action prefix
exactly as the hero would have seen it in view.history at act() time.
One log file is assumed to be one session/tournament (use the engine's
session `logger=` parameter when generating training data; legacy
one-hand-per-file logs degrade gracefully to current-hand memory).

Feature layout (FEATURE_DIM = 26):
  [0]     street          (0=preflop 1=flop 2=turn 3=river)
  [1]     pot             / starting_chips
  [2]     to_call         / starting_chips
  [3]     hero_stack      / starting_chips
  [4]     eff_stack       / starting_chips (min of hero + live acting opps)
  [5]     n_players       (acting opponents + hero)
  [6:10]  hole cards      (rank, suit) x2, zero-padded
  [10:20] board cards     (rank, suit) x5, zero-padded
  [20]    hand_strength   eval_hand normalised to [0, 1]
  [21]    pot_odds        to_call / (pot + to_call)
  [22]    position_value  POSITION_ORDER lookup
  [23:26] memory          [avg_aggression, avg_tightness, avg_vpip]
"""
from core.engine import eval_hand, EVAL_HAND_MAX, RANK_TO_INT

FEATURE_DIM = 26

# Version of the FEATURE SEMANTICS (not the layout). Bump whenever the
# meaning of any feature changes so stale checkpoints are refused:
#   v1 (implicit, pre-Phase-2): raw state-dict checkpoints; preflop strength
#      collapsed to ~0 in training, position defaulted to MP, checks counted
#      as VPIP. Incompatible — retrain.
#   v2: shared builder (this module) used by both training and inference.
# Checkpoints are saved as {"state_dict": ..., "feature_schema_version": N}
# (see training/train_ml_bot.py); MLBot refuses any other version.
FEATURE_SCHEMA_VERSION = 2

RANKS = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
         "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
SUITS = {"c": 0, "d": 1, "h": 2, "s": 3}

STREET_MAP = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}

# Position encoding (0.0 = early/tight, 1.0 = late/loose)
POSITION_ORDER = {
    "UTG": 0.0, "UTG+1": 0.1, "MP": 0.3, "LJ": 0.4,
    "HJ": 0.6, "CO": 0.8, "BTN": 1.0, "SB": 0.5, "BB": 0.3,
}
DEFAULT_POSITION_VALUE = 0.5

# eval_hand uses a small rank-sum heuristic with fewer than 5 cards
# (preflop), whose maximum is pocket aces = 2*max_rank + 40.  Normalising
# those scores by EVAL_HAND_MAX (the 5-card table max) collapses every
# preflop hand to ~0, so we normalise the heuristic on its own scale.
PREFLOP_EVAL_MAX = 2 * max(RANK_TO_INT.values()) + 40

NEUTRAL_MEMORY = [0.5, 0.5, 0.5]


def encode_card(card):
    """Convert ('A','h') / ['A','h'] -> [14, 2]."""
    rank, suit = card
    return [RANKS[rank], SUITS[suit]]


def estimate_hand_strength(hole, board):
    """Hand strength via eval_hand, normalised to [0.0, 1.0].

    With 5+ known cards the 5-card table max applies; preflop / incomplete
    boards use the rank-sum heuristic's own max (see PREFLOP_EVAL_MAX).
    """
    if not hole or len(hole) < 2:
        return 0.0
    # JSON-decoded log rows store cards as lists; eval_hand's 5-card lookup
    # table requires hashable tuples, so normalise both sources here.
    hole = [tuple(c) for c in hole]
    board = [tuple(c) for c in (board or [])]
    score = eval_hand(hole, board)
    if len(hole) + len(board) >= 5:
        return score / EVAL_HAND_MAX
    return min(1.0, score / PREFLOP_EVAL_MAX)


def build_features(*, street, pot, to_call, stacks, me, hole, board, position,
                   opponents=None, acting_opponents=None,
                   memory_features=None, starting_chips=500):
    """Canonical 26-feature vector. Returns a list of FEATURE_DIM floats.

    Both inference (PlayerView fields) and training (logged-row fields)
    must source every argument from the same game state for parity.

    acting_opponents: opponents still able to make betting decisions.
    When None it is derived from `opponents` by filtering stacks > 0
    (same as core.bot_api.acting_opponents_for without explicit list).
    """
    scale = max(1, starting_chips)

    street_idx = STREET_MAP.get(street, 0)
    pot_n = float(pot) / scale
    to_call_n = float(to_call) / scale

    hero_stack_raw = float(stacks.get(me, 0))
    hero_stack = hero_stack_raw / scale

    if acting_opponents is None:
        acting_opponents = [
            pid for pid in (opponents or [])
            if float(stacks.get(pid, 0)) > 0
        ]
    opp_stacks = [
        float(stacks.get(pid, hero_stack_raw))
        for pid in acting_opponents
        if float(stacks.get(pid, 0)) > 0
    ]
    eff_stack = min([hero_stack_raw] + opp_stacks) / scale
    n_players = len(acting_opponents) + 1

    # Hole cards encoding (pad to 4 numbers)
    hole = hole or []
    hole_enc = []
    for i in range(2):
        if i < len(hole):
            hole_enc.extend(encode_card(hole[i]))
        else:
            hole_enc.extend([0, 0])

    # Board encoding (pad to 10 numbers for 5 cards)
    board = board or []
    board_enc = []
    for i in range(5):
        if i < len(board):
            board_enc.extend(encode_card(board[i]))
        else:
            board_enc.extend([0, 0])

    hand_strength = estimate_hand_strength(hole, board)

    if pot_n + to_call_n > 0:
        pot_odds = to_call_n / (pot_n + to_call_n)
    else:
        pot_odds = 0.0

    position_value = POSITION_ORDER.get(position, DEFAULT_POSITION_VALUE)

    if memory_features is None:
        memory_features = list(NEUTRAL_MEMORY)

    features = (
        [street_idx, pot_n, to_call_n, hero_stack, eff_stack, n_players]
        + hole_enc
        + board_enc
        + [hand_strength, pot_odds, position_value]
        + list(memory_features)
    )
    assert len(features) == FEATURE_DIM, f"feature dim {len(features)} != {FEATURE_DIM}"
    return features


class OpponentMemory:
    """Cumulative per-player action stats — the 3 ML memory features.

    Shared definitions (identical for training and live inference):
      aggression = (bet + raise) / actions
      tightness  = fold / actions
      vpip       = (call + bet + raise) / actions
                   (checks put no money in the pot — NOT counted as VPIP)

    Scope is cumulative tournament memory: stats persist across hands for
    the lifetime of the instance (live: one MLBot per tournament; training:
    one instance per (log file, hero), see train_ml_bot.py).
    """

    AGGRESSIVE_ACTIONS = ("bet", "raise")
    VPIP_ACTIONS = ("call", "bet", "raise")

    def __init__(self):
        self.stats = {}   # player_id -> {actions, aggressive, fold, vpip}
        self._seen = set()  # dedup keys for observe_history replays

    def reset(self):
        """Clear all stats and dedup state.

        Call at tournament/session boundaries (e.g. MLBot.reset_memory()):
        a new Table restarts hand ids at 0, so without a reset (or an explicit
        session_id) the (hand_id, index) dedup keys would collide with the
        previous tournament and silently drop the new tournament's actions.
        """
        self.stats = {}
        self._seen = set()

    def observe(self, player, action_type):
        """Record one voluntary action by `player` (any player at the table)."""
        if player is None or action_type is None:
            return
        s = self.stats.setdefault(
            player, {"actions": 0, "aggressive": 0, "fold": 0, "vpip": 0}
        )
        s["actions"] += 1
        if action_type in self.AGGRESSIVE_ACTIONS:
            s["aggressive"] += 1
        if action_type == "fold":
            s["fold"] += 1
        if action_type in self.VPIP_ACTIONS:
            s["vpip"] += 1

    def observe_history(self, history, hand_id=None, session_id=None):
        """Ingest an engine `view.history` list (entries: {"pid","type",...}).

        Live bots see overlapping history prefixes on every act() call within
        a hand, so entries are deduplicated by (session_id, hand_id, index).
        history is append-only within a hand and hand_id is unique per Table,
        making the key stable. Falls back to object identity when hand_id is
        missing.

        session_id disambiguates Tables: a NEW Table restarts hand ids at 0,
        which would collide with keys from a previous tournament and silently
        drop the new actions. Pass a distinct session_id per tournament, or
        call reset() at tournament boundaries (memory is tournament-scoped
        anyway — see MLBot.reset_memory()).
        """
        for idx, entry in enumerate(history or []):
            if not isinstance(entry, dict):
                continue
            if hand_id is not None:
                key = (session_id, hand_id, idx)
            else:
                key = (session_id, None, id(entry), idx)
            if key in self._seen:
                continue
            self._seen.add(key)
            self.observe(entry.get("pid"), entry.get("type"))

    def features_for(self, opponents):
        """[avg_aggression, avg_tightness, avg_vpip] over `opponents`.

        Averages per-opponent ratios across the opponents that have at least
        one observed action; neutral 0.5s when no data is available.
        """
        total_aggression = 0.0
        total_tightness = 0.0
        total_vpip = 0.0
        counted = 0

        for pid in (opponents or []):
            s = self.stats.get(pid)
            if not s or s["actions"] == 0:
                continue
            n = s["actions"]
            total_aggression += s["aggressive"] / n
            total_tightness += s["fold"] / n
            total_vpip += s["vpip"] / n
            counted += 1

        if counted == 0:
            return list(NEUTRAL_MEMORY)

        return [
            total_aggression / counted,
            total_tightness / counted,
            total_vpip / counted,
        ]
