import pickle

import torch
from core.bot_api import Action, acting_opponents_for
from bots.poker_mlp import PokerMLP

# Feature logic is shared with training/train_ml_bot.py — see core/ml_features.py.
# Names re-exported for backwards compatibility.
from core.ml_features import (
    FEATURE_DIM,
    FEATURE_SCHEMA_VERSION,
    RANKS,
    SUITS,
    STREET_MAP,
    PREFLOP_EVAL_MAX as _PREFLOP_EVAL_MAX,
    encode_card,
    estimate_hand_strength,
    build_features,
    OpponentMemory,
)

def _as_view(state):
    """Convert a dict state to an attribute-accessible object if needed."""
    if isinstance(state, dict):
        class _DictView:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)
        return _DictView(state)
    return state


# ML BOT -------------------------------------------------------------

class MLBot:
    def __init__(self, model_path="models/ml_model.pt", device="cpu",
                 use_fallback=True, temperature=0.8, training_mode=False,
                 starting_chips=500):
        self.device = device
        self.use_fallback = use_fallback
        self.temperature = temperature
        self.training_mode = training_mode
        self.starting_chips = max(1, starting_chips)  # normalise stack/pot features
        self.model = PokerMLP(input_dim=FEATURE_DIM, hidden=128, num_classes=6)
        self.model_trained = False

        # TOURNAMENT MEMORY: cumulative opponent stats across all hands this
        # instance acts in (shared semantics with training — core/ml_features.py)
        self.memory = OpponentMemory()

        try:
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = self._validate_checkpoint(checkpoint, model_path)
            if state_dict is not None:
                try:
                    self.model.load_state_dict(state_dict)
                    self.model_trained = True
                except (RuntimeError, ValueError, TypeError, KeyError,
                        AttributeError) as e:
                    print(f"Warning: REFUSING checkpoint {model_path}: "
                          f"state_dict does not load into PokerMLP ({e}). "
                          f"Retrain with training/train_ml_bot.py. "
                          f"Using fallback.")
                    # load_state_dict may partially copy params before
                    # raising — rebuild so fallback weights are untouched
                    # by the corrupt file.
                    self.model = PokerMLP(input_dim=FEATURE_DIM, hidden=128,
                                          num_classes=6)
        except (FileNotFoundError, OSError, RuntimeError,
                EOFError, pickle.UnpicklingError) as e:
            # EOFError: empty/truncated file; UnpicklingError: random bytes
            # or a non-checkpoint pickle — unreadable files fall back loudly
            # instead of crashing the caller.
            print(f"Warning: Could not load model from {model_path}: {e}")
            print("Using untrained model (random weights).")
        self.model.eval()

    def _validate_checkpoint(self, checkpoint, model_path):
        """Return the checkpoint's state dict if compatible, else None.

        Valid checkpoints are {"state_dict": ..., "feature_schema_version": N}
        with N == FEATURE_SCHEMA_VERSION (written by training/train_ml_bot.py).
        Raw state dicts predate feature-schema versioning — they were trained
        with the OLD feature semantics (collapsed preflop strength, MP-default
        position, check-as-VPIP) and are REFUSED: their weights are
        incompatible with what the current builder feeds the network.
        Also refused (never crash): version-marked checkpoints whose
        state_dict is missing, not a dict, or does not look like PokerMLP
        weights (no 2-D 'net.0.weight' tensor).
        """
        if isinstance(checkpoint, dict) and "feature_schema_version" in checkpoint:
            version = checkpoint["feature_schema_version"]
            if version != FEATURE_SCHEMA_VERSION:
                print(f"Warning: REFUSING checkpoint {model_path}: feature schema "
                      f"v{version} != current v{FEATURE_SCHEMA_VERSION}. "
                      f"Retrain with training/train_ml_bot.py. Using fallback.")
                return None
            state_dict = checkpoint.get("state_dict")
            if not isinstance(state_dict, dict) or not state_dict:
                print(f"Warning: REFUSING malformed checkpoint {model_path}: "
                      f"schema marker present but 'state_dict' is missing or "
                      f"not a dict (corrupt or incorrectly saved file). "
                      f"Retrain with training/train_ml_bot.py. Using fallback.")
                return None
        else:
            # Legacy raw state-dict checkpoint — no schema marker.
            print(f"Warning: REFUSING legacy checkpoint {model_path}: no "
                  f"feature_schema_version marker, so it was trained with the "
                  f"old (pre-parity) feature semantics. Retrain with "
                  f"training/train_ml_bot.py. Using fallback.")
            return None

        first_layer_weight = state_dict.get('net.0.weight', None)
        if (not isinstance(first_layer_weight, torch.Tensor)
                or first_layer_weight.dim() != 2):
            print(f"Warning: REFUSING malformed checkpoint {model_path}: "
                  f"'net.0.weight' is missing or not a 2-D tensor — the "
                  f"state_dict does not match PokerMLP. Using fallback.")
            return None
        input_dim = first_layer_weight.shape[1]
        if input_dim != FEATURE_DIM:
            print(f"Warning: Model has input dimension {input_dim}, expected "
                  f"{FEATURE_DIM}. Using fallback.")
            return None
        return state_dict

    def _make_features(self, state):
        """
        Produce the canonical 26-feature vector via the shared builder
        (core/ml_features.py) so training and inference stay in lockstep.
        """
        state = _as_view(state)

        features = build_features(
            street=state.street,
            pot=state.pot,
            to_call=state.to_call,
            stacks=state.stacks,
            me=state.me,
            hole=state.hole_cards,
            board=state.board,
            position=getattr(state, "position", None),
            opponents=getattr(state, "opponents", None),
            acting_opponents=getattr(state, "acting_opponents", None),
            memory_features=self.memory.features_for(acting_opponents_for(state)),
            starting_chips=self.starting_chips,
        )

        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        return x.to(self.device)

    def _estimate_hand_strength(self, hole, board):
        """Hand strength estimate using eval_hand, normalised to [0.0, 1.0]."""
        return estimate_hand_strength(hole, board)

    # ----------------------------------------------------------
    # ACT ------------------------------------------------------
    # ----------------------------------------------------------
    def act(self, state):
        state = _as_view(state)
        legal = state.legal_actions

        # Update cumulative opponent memory from this hand's history so far
        self._update_memory(state)

        # If model not trained and fallback enabled, use fallback
        if not self.model_trained and self.use_fallback:
            return self._fallback_strategy(state)

        # Build 26-dim features
        x = self._make_features(state)

        # Predict class
        logits = self.model(x)

        # Temperature-scaled softmax for exploration (non-training mode)
        if not self.training_mode and self.temperature != 1.0:
            scaled_logits = logits / self.temperature
        else:
            scaled_logits = logits

        probs = torch.softmax(scaled_logits, dim=1)[0]

        if self.training_mode:
            # Training: always argmax
            pred = int(probs.argmax().item())
        else:
            # Inference: sample from temperature-scaled distribution
            pred = int(torch.multinomial(probs, 1).item())

        confidence = float(probs[pred].item())

        # If low confidence and fallback enabled, use fallback
        if self.use_fallback and confidence < 0.5:
            return self._fallback_strategy(state)

        # -------- handle buckets --------
        if pred == 0:
            return self._choose("fold", legal)
        if pred == 1:
            return self._choose("check", legal)
        if pred == 2:
            return self._choose("call", legal)

        # RAISE BUCKETS (3,4,5)
        return self._raise_bucket(pred - 3, legal)

    # ----------------------------------------------------------
    def _choose(self, typ, legal):
        for a in legal:
            if a["type"] == typ:
                return Action(typ)
        for fallback in ("call", "check", "fold"):
            for a in legal:
                if a["type"] == fallback:
                    return Action(fallback)
        a = legal[0]
        return Action(a["type"], a.get("min"))

    # ----------------------------------------------------------
    def _raise_bucket(self, bucket, legal):
        raises = [a for a in legal if a["type"] == "raise"]
        bets = [a for a in legal if a["type"] == "bet"]

        if raises:
            a = raises[0]
        elif bets:
            a = bets[0]
        else:
            return self._choose("call", legal)

        lo, hi = a["min"], a["max"]
        amt = lo + (hi - lo) * (bucket / 2)   # 3 buckets

        return Action(a["type"], round(amt, 2))

    def _fallback_strategy(self, state):
        """Fallback to simple hand strength logic when model is untrained."""
        state = _as_view(state)
        hole = state.hole_cards
        board = state.board
        pot = state.pot
        to_call = state.to_call
        legal = state.legal_actions

        if not hole:
            return self._choose("fold", legal)

        strength = self._estimate_hand_strength(hole, board)

        # Facing a bet
        if to_call > 0:
            pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.5
            if strength < pot_odds:
                return self._choose("fold", legal)
            if strength < 0.55:
                return self._choose("call", legal)
            return self._raise_fallback(pot, legal)

        # No bet yet
        if strength > 0.60:
            return self._bet_fallback(pot, legal)
        return self._choose("check", legal)

    def _raise_fallback(self, pot, legal):
        """Raise helper for fallback strategy."""
        for a in legal:
            if a["type"] == "raise":
                amt = max(a["min"], min(a["max"], int(pot * 0.75)))
                return Action("raise", amt)
        return self._choose("call", legal)

    def _bet_fallback(self, pot, legal):
        """Bet helper for fallback strategy."""
        for a in legal:
            if a["type"] == "bet":
                amt = max(a["min"], min(a["max"], int(pot * 0.5)))
                return Action("bet", amt)
        return self._choose("check", legal)

    def reset_memory(self):
        """Start a fresh tournament: clear cumulative opponent memory.

        MUST be called between tournaments when one MLBot instance is reused
        across Tables — a new Table restarts hand ids at 0, so stale dedup
        keys would otherwise silently swallow the new tournament's actions
        (and opponent stats from the old tournament would leak in).
        """
        self.memory.reset()

    def _update_memory(self, state):
        """
        Feed the current hand's history into cumulative opponent memory.
        Stats accumulate across hands (deduplicated per hand_id + index);
        VPIP/aggression/tightness definitions live in core/ml_features.py.
        """
        self.memory.observe_history(
            getattr(state, "history", None),
            hand_id=getattr(state, "hand_id", None),
        )
