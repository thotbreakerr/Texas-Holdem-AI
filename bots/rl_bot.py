"""
Reinforcement Learning Bot using Proximal Policy Optimization (PPO).
Learns optimal strategies through trial and error.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from core.bot_api import Action, PlayerView, acting_opponents_for
from core.engine import eval_hand, EVAL_HAND_MAX, RANK_TO_INT

# eval_hand uses a small rank-sum heuristic with fewer than 5 cards (preflop),
# whose maximum is pocket aces = 2*max_rank + 40.  Normalising those scores by
# EVAL_HAND_MAX (the 5-card table max) collapses every preflop hand to ~0, so we
# normalise the heuristic on its own scale instead.
_PREFLOP_EVAL_MAX = 2 * max(RANK_TO_INT.values()) + 40

# Card encoding (same as MLBot)
RANKS = {"2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8,
         "9":9, "T":10, "J":11, "Q":12, "K":13, "A":14}
SUITS = {"c":0, "d":1, "h":2, "s":3}

def encode_card(card):
    rank, suit = card
    return [RANKS[rank], SUITS[suit]]

STREET_MAP = {"preflop":0, "flop":1, "turn":2, "river":3}

# Logit fill value for illegal actions.  exp(-1e9) underflows to exactly 0.0
# in float32, so masked actions get probability 0 and can never be sampled.
_MASK_FILL = -1e9

# Action-head layout: 0=fold, 1=check, 2=call, 3/4/5=raise small/medium/large.
_TYPE_TO_IDX = {"fold": 0, "check": 1, "call": 2}

# NOTE: no dropout in either network.  Dropout makes every forward pass
# stochastic, so the log-probs recorded at rollout time and the log-probs
# recomputed during the PPO update disagree even with unchanged weights —
# the importance ratio is then ≠ 1 before any optimisation step, which
# corrupts the clipped surrogate objective.  Removing the layers also shifts
# the nn.Sequential state-dict keys, so checkpoints saved by the old
# (dropout) architecture no longer load; callers already handle that by
# starting fresh, and those checkpoints were trained on corrupted PPO data.

class PolicyNetwork(nn.Module):
    """Policy network that outputs action logits (deterministic forward)."""
    def __init__(self, input_dim=26, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 6)  # 6 actions: fold, check, call, raise_small, raise_medium, raise_large
        )

    def forward(self, x):
        return self.net(x)


class ValueNetwork(nn.Module):
    """Value network that estimates V(s) for PPO baseline (deterministic forward)."""
    def __init__(self, input_dim=26, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),  # single scalar value output
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RLBot:
    """
    Reinforcement Learning Bot using PPO (Proximal Policy Optimization).
    Learns by playing games and updating policy based on rewards.
    """
    
    def __init__(self, model_path="models/rl_model.pt", device="cpu",
                 learning_rate=1e-4, training_mode=False, exploration_rate=0.1,
                 use_fallback=True, starting_chips=500):
        self.device = device
        self.training_mode = training_mode
        # NOTE: epsilon-greedy exploration was removed — uniform-random actions
        # recorded with on-policy log-probs corrupt the PPO importance ratio.
        # Exploration comes from sampling the policy + the entropy bonus.
        # The arg is kept so existing callers don't break.
        self.exploration_rate = exploration_rate
        self.use_fallback = use_fallback
        self.starting_chips = max(1, starting_chips)  # used to normalise stack/pot features
        self.model_loaded = False  # Track if model loaded successfully
        self.policy_net = PolicyNetwork(input_dim=26, hidden=512).to(device)
        self.value_net  = ValueNetwork(input_dim=26,  hidden=512).to(device)
        self.optimizer       = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(),  lr=learning_rate)

        # Episode tracking for PPO
        self.current_episode = []
        self.episode_rewards = []
        self._hand_step_start = 0   # index into current_episode where current hand's steps begin

        # Batch buffer: collect multiple episodes before each update
        self.episode_buffer = []
        self.batch_size     = 32

        # Load existing model if available
        try:
            checkpoint = torch.load(model_path, map_location=device)
            # New format: {'policy': ..., 'value': ...}
            if isinstance(checkpoint, dict) and 'policy' in checkpoint:
                self.policy_net.load_state_dict(checkpoint['policy'])
                self.value_net.load_state_dict(checkpoint['value'])
            else:
                # Old format: bare policy state dict — load policy only
                self.policy_net.load_state_dict(checkpoint)
            self.model_loaded = True
            if training_mode:
                print(f"Loaded RL model from {model_path} (training mode)")
            else:
                print(f"Loaded RL model from {model_path}")
        except (FileNotFoundError, OSError, RuntimeError) as e:
            self.model_loaded = False
            if training_mode:
                print(f"No existing RL model found. Starting fresh training.")
            else:
                print(f"Warning: Could not load RL model from {model_path}: {e}")
                if use_fallback:
                    print("Using fallback strategy.")

        if not training_mode:
            self.policy_net.eval()
            self.value_net.eval()
        else:
            self.policy_net.train()
            self.value_net.train()

        # Opponent memory
        self.opponent_stats = {}
        # Running mean of episode returns (kept for diagnostics)
        self.baseline_returns = deque(maxlen=1000)
    
    def _make_features(self, state):
        """Extract 26-dimensional feature vector (same as MLBot)."""
        # Handle both PlayerView and dict
        if isinstance(state, dict):
            class DictView:
                def __init__(self, d):
                    for k, v in d.items():
                        setattr(self, k, v)
            state = DictView(state)
        
        street = STREET_MAP.get(state.street, 0)
        # Normalise monetary values by starting stack so they are on [0, ~2]
        # rather than raw chip counts (0 – 1000+).
        scale = self.starting_chips
        pot = float(state.pot) / scale
        to_call = float(state.to_call) / scale
        hero_stack_raw = float(state.stacks.get(state.me, 0))
        hero_stack = hero_stack_raw / scale
        acting_opponents = acting_opponents_for(state)
        opp_stacks = [
            float(state.stacks.get(pid, hero_stack_raw))
            for pid in acting_opponents
            if float(state.stacks.get(pid, 0)) > 0
        ]
        eff_stack = min([hero_stack_raw] + opp_stacks) / scale
        n_players = len(acting_opponents) + 1
        
        # Hole cards encoding
        hole = state.hole_cards or []
        hole_enc = []
        for i in range(2):
            if i < len(hole):
                hole_enc.extend(encode_card(hole[i]))
            else:
                hole_enc.extend([0, 0])
        
        # Board encoding
        board = state.board or []
        board_enc = []
        for i in range(5):
            if i < len(board):
                board_enc.extend(encode_card(board[i]))
            else:
                board_enc.extend([0, 0])
        
        # Hand strength
        hand_strength = self._estimate_hand_strength(hole, board)
        
        # Pot odds
        if pot + to_call > 0:
            pot_odds = to_call / (pot + to_call)
        else:
            pot_odds = 0.0
        
        # Position encoding
        position_order = {
            "UTG": 0.0, "UTG+1": 0.1, "MP": 0.3, "LJ": 0.4,
            "HJ": 0.6, "CO": 0.8, "BTN": 1.0, "SB": 0.5, "BB": 0.5
        }
        position_value = position_order.get(state.position, 0.5)
        
        # Memory features
        memory_features = self._calculate_memory_features(
            state.history, state.me, acting_opponents
        )
        
        features = (
            [street, pot, to_call, hero_stack, eff_stack, n_players]
            + hole_enc + board_enc
            + [hand_strength, pot_odds, position_value]
            + memory_features
        )
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def _estimate_hand_strength(self, hole, board):
        """Hand strength estimate using eval_hand, normalised to [0.0, 1.0]."""
        if not hole or len(hole) < 2:
            return 0.0
        score = eval_hand(hole, board)
        if len(hole) + len(board) >= 5:
            return score / EVAL_HAND_MAX
        # Preflop / incomplete board: normalise the rank-sum heuristic.
        return min(1.0, score / _PREFLOP_EVAL_MAX)
    
    def _calculate_memory_features(self, history, me, opponents):
        """Calculate opponent behavior features from action history."""
        if not opponents or not history:
            return [0.5, 0.5, 0.5]

        # History entries written by the engine use top-level keys:
        # {"street": ..., "pid": ..., "type": "fold"|"call"|..., "amount": ...}
        opponent_actions = [
            entry
            for entry in history
            if isinstance(entry, dict) and entry.get("pid") in opponents
        ]

        if not opponent_actions:
            return [0.5, 0.5, 0.5]

        recent = opponent_actions[-10:]
        total = len(recent)
        if total == 0:
            return [0.5, 0.5, 0.5]

        aggressive = sum(1 for d in recent if d.get("type") in ("bet", "raise")) / total
        tightness = sum(1 for d in recent if d.get("type") == "fold") / total
        vpip = sum(1 for d in recent if d.get("type") in ("call", "bet", "raise", "check")) / total

        return [aggressive, tightness, vpip]
    
    def _update_memory(self, history, opponents):
        """Update opponent stats from hand history."""
        if not history or not opponents:
            return
        # Engine history entries: {"pid": ..., "type": "fold"|"call"|..., ...}
        for entry in history:
            if not isinstance(entry, dict):
                continue
            player = entry.get("pid")
            action_type = entry.get("type", "fold")
            if player not in opponents:
                continue

            if player not in self.opponent_stats:
                self.opponent_stats[player] = {
                    'action_count': 0, 'aggressive_count': 0,
                    'fold_count': 0, 'vpip_count': 0, 'last_action': action_type
                }
            stats = self.opponent_stats[player]
            stats['action_count'] += 1
            stats['last_action'] = action_type
            if action_type in ("bet", "raise"):
                stats['aggressive_count'] += 1
            if action_type == "fold":
                stats['fold_count'] += 1
            if action_type in ("call", "bet", "raise", "check"):
                stats['vpip_count'] += 1
    
    def _fallback_strategy(self, state):
        """Fallback to simple hand strength logic when model is not loaded."""
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
            # Raise with strong hands
            return self._raise_bucket(1, legal)  # Medium raise
        
        # No bet yet
        if strength > 0.60:
            return self._bet(pot, legal)
        return self._choose("check", legal)
    
    def _bet(self, pot, legal):
        """Make a bet."""
        for a in legal:
            if a["type"] == "bet":
                amt = max(a["min"], min(a["max"], pot * 0.5))
                return Action("bet", round(amt, 2))
        return self._choose("check", legal)

    def _legal_action_mask(self, legal):
        """
        Boolean mask of shape (1, 6) over the action head:
        True where the index maps to a currently legal engine action.
        Raise buckets (3-5) are legal whenever a "bet" or "raise" is legal.

        Fail-closed: an empty mask during training raises instead of
        fabricating an all-legal mask — training must never continue on
        invalid data.  In eval mode we stay permissive (all-True) so the
        fallback machinery can keep the bot playable.
        """
        types = {a["type"] for a in legal}
        aggressive = bool(types & {"bet", "raise"})
        mask = [
            "fold" in types,
            "check" in types,
            "call" in types,
            aggressive, aggressive, aggressive,
        ]
        if not any(mask):
            if self.training_mode:
                raise ValueError(
                    f"empty legal-action mask during training "
                    f"(legal_actions={legal!r}); refusing to assume all "
                    f"actions are legal"
                )
            mask = [True] * 6
        return torch.tensor(mask, dtype=torch.bool,
                            device=self.device).unsqueeze(0)

    def _masked_policy_dist(self, states, masks):
        """
        Masked policy distribution over the action head.

        Single source of truth for action probabilities: both rollout
        sampling (act) and PPO log-prob recomputation (_ppo_update) go
        through here, so with unchanged weights the importance ratio for a
        stored action is exactly 1 (the networks are deterministic — no
        dropout/batch-norm).
        """
        logits = self.policy_net(states)
        logits = logits.masked_fill(~masks, _MASK_FILL)
        return torch.distributions.Categorical(logits=logits)

    @staticmethod
    def _executed_action_idx(action, sampled_idx):
        """
        Action-head index corresponding to the action actually executed.

        Used as a consistency *detector*, not a repair: if this disagrees
        with the sampled index, act() raises rather than re-anchoring the
        stored probability.  Raise-bucket aliasing is intentional and
        allowed: indices 3/4/5 all map to type "bet"/"raise" (differing
        only in amount), so an aggressive executed action keeps the
        sampled bucket index.
        """
        idx = _TYPE_TO_IDX.get(action.type)
        if idx is not None:
            return idx
        # bet/raise: keep the sampled bucket; a non-aggressive sampled index
        # that produced an aggressive action is a mismatch (detected above).
        return sampled_idx if sampled_idx >= 3 else 3

    def act(self, state):
        """Choose action using policy network."""
        try:
            # Handle dict/PlayerView
            if isinstance(state, dict):
                class DictView:
                    def __init__(self, d):
                        for k, v in d.items():
                            setattr(self, k, v)
                state = DictView(state)
            
            legal = state.legal_actions
            
            # Update memory
            if hasattr(state, 'history') and state.history:
                self._update_memory(state.history, acting_opponents_for(state))
            
            # Use fallback if model not loaded and fallback enabled
            if not self.model_loaded and self.use_fallback and not self.training_mode:
                return self._fallback_strategy(state)
            
            # Get features and legal-action mask
            features = self._make_features(state)
            mask = self._legal_action_mask(legal)

            # Rollout probability computation never needs gradients — the
            # PPO update recomputes log-probs from the stored states through
            # the same _masked_policy_dist path.
            with torch.no_grad():
                dist = self._masked_policy_dist(features, mask)
                if self.training_mode:
                    value = self.value_net(features)   # shape: (1,)
                    # Sample from the masked policy (exploration comes from
                    # sampling + the entropy bonus; epsilon-greedy was removed
                    # because it breaks the PPO importance ratio)
                    action_idx = dist.sample()
                else:
                    # Eval mode: greedy over legal actions
                    action_idx = dist.probs.argmax(dim=1)
                log_prob = dist.log_prob(action_idx)

            # Convert to the actual engine action BEFORE storing the step so
            # the stored action always matches the executed one.
            action = self._action_idx_to_action(action_idx.item(), legal)

            # Store trajectory step for PPO update
            if self.training_mode:
                executed_idx = self._executed_action_idx(action, action_idx.item())
                if executed_idx != action_idx.item():
                    # Cannot happen under the mask.  Fail loudly: silently
                    # "repairing" the stored sample would record a probability
                    # that doesn't account for every label that falls back to
                    # this executed action.
                    raise RuntimeError(
                        f"sampled action index {action_idx.item()} converted "
                        f"to '{action.type}' (index {executed_idx}); fallback "
                        f"conversion would silently change the stored action "
                        f"(legal={legal!r})"
                    )
                self.current_episode.append({
                    'state':    features,
                    'action':   action_idx.item(),
                    'log_prob': log_prob,   # old π(a|s) — no-grad, frozen
                    'value':    value,       # V(s) old estimate — no-grad, frozen
                    'mask':     mask,        # legal-action mask, reused in updates
                    'legal_actions': legal,
                })

            return action

        except Exception as e:
            if self.training_mode:
                # Never silently substitute a fallback action during training:
                # the executed action would no longer match the stored
                # trajectory (or be missing from it entirely).
                raise
            # If anything goes wrong, use fallback
            if self.use_fallback:
                try:
                    return self._fallback_strategy(state)
                except:
                    # Last resort: just fold
                    legal = state.legal_actions if hasattr(state, 'legal_actions') else []
                    return self._choose("fold", legal)
            else:
                raise
    
    def _action_idx_to_action(self, idx, legal):
        """Convert action index to Action object."""
        # Ensure idx is valid
        idx = max(0, min(5, idx))
        
        if idx == 0:  # fold
            return self._choose("fold", legal)
        elif idx == 1:  # check
            return self._choose("check", legal)
        elif idx == 2:  # call
            return self._choose("call", legal)
        else:  # raise buckets (3, 4, 5)
            bucket = min(2, max(0, idx - 3))  # Clamp to 0-2
            return self._raise_bucket(bucket, legal)
    
    def _choose(self, typ, legal):
        """Choose legal action of given type."""
        for a in legal:
            if a["type"] == typ:
                return Action(typ)
        for fallback in ("call", "check", "fold"):
            for a in legal:
                if a["type"] == fallback:
                    return Action(fallback)
        a = legal[0]
        return Action(a["type"], a.get("min"))
    
    def _raise_bucket(self, bucket, legal):
        """Choose raise amount based on bucket (0=small, 1=medium, 2=large)."""
        raises = [a for a in legal if a["type"] == "raise"]
        bets = [a for a in legal if a["type"] == "bet"]
        
        if raises:
            a = raises[0]
        elif bets:
            a = bets[0]
        else:
            return self._choose("call", legal)
        
        lo, hi = a["min"], a["max"]
        amt = lo + (hi - lo) * (bucket / 2.0)
        return Action(a["type"], round(amt, 2))
    
    def record_reward(self, reward):
        """
        Record the reward for the hand that just finished.

        Call this once per hand with the hand's chip delta normalised by a
        stable baseline (big blinds or the initial stack — NOT the current
        stack).  The reward is assigned to the hand's final decision only;
        earlier decisions get 0 so the chip delta is counted exactly once.
        GAE/discounting propagates the credit backwards through the hand.

        No-decision hands (e.g. all-in from the blinds, so the bot never
        acted this hand) produce no new transition; their chip delta is
        added onto the most recent policy transition in the episode so the
        episode return still reflects the true chip outcome.  If the bot has
        made no decision in the entire episode there is nothing to credit
        and the reward is dropped (see sanity_rl_ppo.py section 11).
        """
        if self.training_mode and self.current_episode:
            # Tag only the steps that belong to the hand that just finished
            steps = self.current_episode[self._hand_step_start:]
            if steps:
                for step in steps[:-1]:
                    step['reward'] = 0.0
                steps[-1]['reward'] = reward
                # Advance the marker so the next hand won't overwrite these
                self._hand_step_start = len(self.current_episode)
            else:
                # No-decision hand: attach the chip delta to the most recent
                # policy transition (reward arrives after the last action).
                last = self.current_episode[-1]
                last['reward'] = last.get('reward', 0.0) + reward
            self.episode_rewards.append(reward)

    def record_terminal_bonus(self, bonus):
        """
        Add a terminal win/loss bonus onto the episode's final transition.

        Must be called after the last hand's ``record_reward`` and before
        ``end_episode``.  Unlike ``record_reward`` (which only tags steps of
        the not-yet-rewarded hand), this always reaches the last stored step.
        """
        if self.training_mode and self.current_episode:
            last = self.current_episode[-1]
            last['reward'] = last.get('reward', 0.0) + bonus

    def end_episode(self):
        """Buffer the completed episode; run PPO update every batch_size episodes."""
        if not self.training_mode or not self.current_episode:
            self.current_episode = []
            self._hand_step_start = 0
            return

        # Move the finished episode into the buffer and reset
        self.episode_buffer.append(list(self.current_episode))
        self.current_episode = []
        self._hand_step_start = 0

        # Only update when we have a full batch
        if len(self.episode_buffer) >= self.batch_size:
            self._ppo_update(self.episode_buffer)
            self.episode_buffer = []

    def flush_buffer(self):
        """Force a PPO update on any remaining buffered episodes (call at end of training)."""
        if not self.training_mode:
            return
        # Close out any in-progress episode first so the final episode of a
        # training run is never silently dropped.
        self.end_episode()
        if self.episode_buffer:
            self._ppo_update(self.episode_buffer)
            self.episode_buffer = []

    def save_model(self, path="models/rl_model.pt"):
        """Save policy and value network weights to disk."""
        import os
        dir_ = os.path.dirname(path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        torch.save({
            'policy': self.policy_net.state_dict(),
            'value':  self.value_net.state_dict(),
        }, path)

    def _ppo_update(self, episodes):
        """
        Run a PPO update over a batch of completed episodes.

        Uses clipped surrogate objective (PPO-clip) with:
          - GAE-lambda advantage estimation (gamma=0.99, lambda=0.95)
          - Shared normalisation of advantages across the batch
          - Entropy bonus to encourage exploration
          - Gradient clipping for stability

        Args:
            episodes: list of episode buffers, each a list of step dicts
                      with keys: state, action, log_prob, value, mask, reward
        """
        GAMMA     = 0.99
        LAMBDA    = 0.95
        CLIP_EPS  = 0.2
        VF_COEF   = 0.5
        ENT_COEF  = 0.01
        PPO_EPOCHS = 4

        all_states       = []
        all_actions      = []
        all_old_log_probs = []
        all_returns      = []
        all_advantages   = []
        all_masks        = []

        for ep in episodes:
            # Only keep steps that received a reward assignment
            steps = [s for s in ep if 'reward' in s]
            if not steps:
                continue

            rewards = [s['reward'] for s in steps]
            values  = [float(s['value'].item() if hasattr(s['value'], 'item')
                             else s['value']) for s in steps]

            # GAE advantage + discounted return computation (backward pass)
            gae        = 0.0
            next_value = 0.0
            advantages = [0.0] * len(steps)
            returns    = [0.0] * len(steps)

            for t in reversed(range(len(steps))):
                delta          = rewards[t] + GAMMA * next_value - values[t]
                gae            = delta + GAMMA * LAMBDA * gae
                advantages[t]  = gae
                returns[t]     = gae + values[t]
                next_value     = values[t]

            for step, adv, ret in zip(steps, advantages, returns):
                # Fail closed: a step without its acting-time legal mask
                # cannot be trained on — assuming "all actions legal" would
                # silently shift the recomputed distribution.
                step_mask = step.get('mask')
                if step_mask is None:
                    raise ValueError(
                        "trajectory step is missing its legal-action mask; "
                        "refusing to assume all actions were legal during "
                        "the PPO update"
                    )
                all_states.append(step['state'])
                all_actions.append(step['action'])
                all_old_log_probs.append(step['log_prob'])
                all_advantages.append(adv)
                all_returns.append(ret)
                all_masks.append(step_mask)

        if not all_states:
            return

        # Stack into tensors
        states_t       = torch.cat(all_states, dim=0).to(self.device)
        actions_t      = torch.tensor(all_actions, dtype=torch.long).to(self.device)
        # reshape(-1), not squeeze(): squeeze() on a single-step batch would
        # produce a 0-d tensor and break indexing below.
        old_log_probs_t = torch.stack(all_old_log_probs).to(self.device).reshape(-1)
        advantages_t   = torch.tensor(all_advantages, dtype=torch.float32).to(self.device)
        returns_t      = torch.tensor(all_returns,    dtype=torch.float32).to(self.device)
        masks_t        = torch.cat(all_masks, dim=0).to(self.device)

        # Normalise advantages across the whole batch
        if advantages_t.numel() > 1 and advantages_t.std() > 1e-8:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n = states_t.shape[0]

        for _ in range(PPO_EPOCHS):
            indices = torch.randperm(n, device=self.device)

            # ── Policy loss ──────────────────────────────────────────
            # Same masked-distribution path as rollout sampling, so the
            # ratio is exactly 1 for unchanged weights (epoch 1, step 0).
            dist         = self._masked_policy_dist(states_t[indices],
                                                    masks_t[indices])
            new_log_probs = dist.log_prob(actions_t[indices])
            entropy      = dist.entropy().mean()

            ratio  = torch.exp(new_log_probs - old_log_probs_t[indices].detach())
            adv_b  = advantages_t[indices]
            surr1  = ratio * adv_b
            surr2  = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean() - ENT_COEF * entropy

            self.optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer.step()

            # ── Value loss ───────────────────────────────────────────
            values_pred = self.value_net(states_t[indices])
            value_loss  = VF_COEF * torch.nn.functional.mse_loss(
                values_pred, returns_t[indices].detach()
            )

            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()
