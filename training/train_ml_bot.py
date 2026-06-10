import os
import sys
import json
import glob
import torch
from torch.utils.data import Dataset, DataLoader

# Add project root to path so imports work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from bots.poker_mlp import PokerMLP

import torch.nn as nn

# Feature logic is shared with bots/ml_bot.py inference — core/ml_features.py
# is the single source of truth for the 26-feature vector. Names re-exported
# for backwards compatibility.
from core.ml_features import (
    FEATURE_SCHEMA_VERSION,
    RANKS,
    SUITS,
    STREET_MAP,
    encode_card,
    build_features,
    OpponentMemory,
)
from core.logger import LOG_FORMAT_VERSION


def make_checkpoint(state_dict):
    """Wrap a state dict with the feature-schema marker MLBot requires.

    MLBot refuses raw state dicts (legacy, pre-parity feature semantics) and
    checkpoints whose feature_schema_version differs from the current one.
    """
    return {
        "state_dict": state_dict,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }




def load_decision_logs(root):
    """
    Recursively loads every .jsonl file inside logs/
    and returns a list of decision-dictionaries.
    """
    decisions = []

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.endswith(".jsonl"):
                continue

            full = os.path.join(dirpath, fname)
            try:
                with open(full, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)

                        # Skip final result rows
                        if "chosen_action" not in obj:
                            continue

                        decisions.append(obj)

            except Exception as e:
                print(f"[WARN] Could not read {full}: {e}")

    return decisions

class PokerDataset(Dataset):
    ACTION_MAP = {
        "fold": 0,
        "check": 1,
        "call": 2,
        # raises get bucketed into 3 bins
        "raise_small": 3,
        "raise_medium": 4,
        "raise_large": 5
    }

    def __init__(self, log_folder, filter_players=None, filter_winners_only=False,
                 starting_chips=500, allow_unmarked_sessions=False):
        """
        Args:
            log_folder: Directory containing log files
            filter_players: List of player IDs to include (e.g., ["P3"] for MonteCarloBot only)
            filter_winners_only: If True, only include decisions from hands that were won
            starting_chips: Used to normalise pot and stack features to [0, ~2] scale
            allow_unmarked_sessions: Accept files WITHOUT a {"session_start": ...}
                header (legacy per-hand logs). Off by default because memory
                features are cumulative per session: a per-hand file silently
                truncates opponent memory to a single hand, which does NOT
                match what a live bot saw. Generate proper logs with
                `run_local_match.py --log-session`.

        Features come from core.ml_features.build_features — the exact same
        builder MLBot uses at inference time. Memory features are produced by
        replaying each log file through OpponentMemory exactly as a live bot
        would have observed it (see _iter_file_samples).

        Raises ValueError when unmarked files are present and not allowed,
        and ALWAYS (not bypassable) when a file is corrupt: every non-empty
        row must be valid JSON and a JSON object, and a session header must
        be the first row, appear exactly once, and carry a nonempty
        session_id with log_format_version == LOG_FORMAT_VERSION. Corrupt
        rows are hard errors, never "legacy": a damaged line could be a
        second session_start, and skipping it would merge two tournaments
        into one cumulative-memory replay.
        """
        self.samples = []
        self.starting_chips = max(1, starting_chips)
        self.missing_position_rows = 0  # legacy logs without "position"
        self.unmarked_session_files = []  # files without a session_start header
        self.invalid_session_files = []   # [(path, reason)] — corrupt files

        files = sorted(glob.glob(f"{log_folder}/**/*.jsonl", recursive=True))
        parsed = []
        for path in files:
            decisions, hand_results, has_session_header, file_error = \
                self._load_file(path)
            if file_error is not None:
                self.invalid_session_files.append((path, file_error))
                continue
            if not has_session_header:
                self.unmarked_session_files.append(path)
            parsed.append((decisions, hand_results))

        if self.invalid_session_files:
            details = "; ".join(f"{os.path.basename(p)}: {reason}"
                                for p, reason in self.invalid_session_files[:5])
            more = len(self.invalid_session_files) - 5
            raise ValueError(
                f"{len(self.invalid_session_files)} log file(s) are corrupt "
                f"({details}{f'; +{more} more' if more > 0 else ''}). A "
                f"trainable log must contain only valid JSON object rows, "
                f"and a session header must be the first row, appear "
                f"exactly once, and contain a nonempty session_id with "
                f"log_format_version={LOG_FORMAT_VERSION}. Corrupt or "
                f"concatenated files can merge multiple sessions into one "
                f"cumulative-memory replay, so they are never trainable "
                f"(not bypassable with allow_unmarked_sessions). Regenerate "
                f"with `run_local_match.py --log-session`."
            )

        if self.unmarked_session_files and not allow_unmarked_sessions:
            shown = ", ".join(os.path.basename(p)
                              for p in self.unmarked_session_files[:5])
            more = len(self.unmarked_session_files) - 5
            raise ValueError(
                f"{len(self.unmarked_session_files)} log file(s) lack a session "
                f"header ({shown}{f', +{more} more' if more > 0 else ''}). "
                f"These are legacy/per-hand logs: cumulative opponent-memory "
                f"features cannot be reconstructed from them, so training on "
                f"them would not match live inference. Regenerate with "
                f"`run_local_match.py --log-session`, or pass "
                f"allow_unmarked_sessions=True / --allow-legacy-logs to "
                f"proceed anyway."
            )
        if self.unmarked_session_files:
            print(f"[WARN] {len(self.unmarked_session_files)} log file(s) lack a "
                  f"session header — treating each file as its own session. "
                  f"Opponent-memory features from these files are NOT "
                  f"trustworthy for cumulative-memory training.")

        for decisions, hand_results in parsed:
            self._iter_file_samples(
                decisions, hand_results, filter_players, filter_winners_only
            )

        if self.missing_position_rows:
            print(f"[WARN] {self.missing_position_rows} decision rows lack 'position' "
                  f"(legacy logs) — defaulted to MP. Regenerate logs with the "
                  f"current engine for true positions.")

    @staticmethod
    def _validate_session_header(row):
        """Return None if a session_start row is well-formed, else a reason."""
        info = row.get("session_start")
        if not isinstance(info, dict):
            return "malformed session_start row (payload is not an object)"
        session_id = info.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            return "empty or missing session_id"
        version = info.get("log_format_version")
        if version != LOG_FORMAT_VERSION:
            return (f"log_format_version {version!r} != required "
                    f"{LOG_FORMAT_VERSION}")
        return None

    @staticmethod
    def _load_file(path):
        """Read one .jsonl file ->
        (decision rows, hand winners, has_session_header, file_error).

        Session-scoped logs (DecisionLogger(session_scoped=True)) start with a
        {"session_start": {...}} row marking the file as one full tournament.
        Strict rules — violations set file_error (a reason string) and the
        caller rejects such files unconditionally:
          * the header must be the FIRST non-empty row, appear exactly once,
            and pass _validate_session_header;
          * every non-empty row must be valid JSON AND a JSON object.
        Malformed rows are NEVER skipped as noise: a corrupt line could be a
        damaged second session_start, so skipping it would silently merge
        two tournaments into one cumulative-memory replay. Unlike
        header-less legacy files, corrupt files are not opt-in loadable.
        """
        decisions = []
        hand_results = {}  # hand_id -> winner player_id (positive net)
        has_session_header = False
        file_error = None
        row_idx = 0  # counts non-empty lines, including unparseable ones
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        file_error = file_error or \
                            f"invalid JSON at row {row_idx}"
                        row_idx += 1
                        continue
                    if not isinstance(row, dict):
                        file_error = file_error or \
                            (f"row {row_idx} is not a JSON object "
                             f"({type(row).__name__})")
                        row_idx += 1
                        continue
                    if "session_start" in row:
                        if has_session_header:
                            file_error = file_error or \
                                "multiple session_start rows"
                        elif row_idx > 0:
                            file_error = file_error or \
                                "session_start is not the first row"
                        else:
                            file_error = file_error or \
                                PokerDataset._validate_session_header(row)
                        has_session_header = True
                    elif "chosen_action" in row:
                        decisions.append(row)
                    elif "result" in row and "hand_id" in row:
                        if row["result"].get("net", 0) > 0:
                            hand_results[row["hand_id"]] = row["result"].get("player")
                    row_idx += 1
        except OSError as e:
            print(f"[WARN] Could not read {path}: {e}")
        return decisions, hand_results, has_session_header, file_error

    def _iter_file_samples(self, decisions, hand_results,
                           filter_players, filter_winners_only):
        """Build samples for one session file, replaying opponent memory.

        Parity with live inference: a live MLBot only observes view.history
        at its own act() calls — i.e. each hand's action prefix up to its own
        decisions, accumulated across hands. We replay that exactly with one
        OpponentMemory per hero: when hero P acts in hand H, P's memory is
        fed every action of H that happened before this decision (and nothing
        after P's last action in H — P never saw those live).
        """
        memories = {}      # hero player_id -> OpponentMemory
        hand_entries = {}  # hand_id -> [(player, action_type), ...] in order
        fed = {}           # (hero, hand_id) -> count of entries already fed

        for row in decisions:
            me = row.get("player")
            chosen = row.get("chosen_action") or {}
            act_type = chosen.get("type")
            hand_id = row.get("hand_id")

            entries = hand_entries.setdefault(hand_id, [])

            # ---- replay memory the hero would have at this decision ----
            mem = memories.setdefault(me, OpponentMemory())
            start = fed.get((me, hand_id), 0)
            for actor, actor_action in entries[start:]:
                mem.observe(actor, actor_action)
            fed[(me, hand_id)] = len(entries)

            sample = self._make_sample(row, mem, filter_players,
                                       filter_winners_only, hand_results)
            if sample is not None:
                self.samples.append(sample)

            # Record this decision AFTER feature building — at act() time the
            # hero's own current action is not yet part of history.
            if act_type is not None:
                entries.append((me, act_type))

    def _make_sample(self, row, mem, filter_players, filter_winners_only,
                     hand_results):
        """Encode one logged decision -> (features, label), or None if filtered."""
        player_id = row.get("player")
        if filter_players and player_id not in filter_players:
            return None

        if filter_winners_only:
            hand_id = row.get("hand_id")
            if hand_id not in hand_results or hand_results[hand_id] != player_id:
                return None

        chosen = row["chosen_action"]
        act_type = chosen["type"]
        act_amt = chosen["amount"]

        # --------- LABEL ENCODING ---------
        if act_type in ("fold", "check", "call"):
            label = self.ACTION_MAP[act_type]
        elif act_type in ("raise", "bet"):
            legal = next(
                (la for la in row["legal"]
                 if la["type"] in ("raise", "bet")),
                None
            )
            if legal is None:
                label = self.ACTION_MAP["call"]
            else:
                lo, hi = legal["min"], legal["max"]
                if act_amt <= lo + (hi - lo) * 0.33:
                    label = self.ACTION_MAP["raise_small"]
                elif act_amt <= lo + (hi - lo) * 0.66:
                    label = self.ACTION_MAP["raise_medium"]
                else:
                    label = self.ACTION_MAP["raise_large"]
        else:
            return None

        # --------- FEATURE ENCODING (shared with MLBot inference) ---------
        position = row.get("position")
        if position is None:
            self.missing_position_rows += 1
            position = "MP"  # legacy logs only — engine now logs position

        acting_opponents = row.get("acting_opponents")
        memory_features = mem.features_for(
            acting_opponents if acting_opponents is not None
            else [pid for pid in row.get("opponents", [])
                  if float(row["stacks"].get(pid, 0)) > 0]
        )

        features = build_features(
            street=row["street"],
            pot=row["pot"],
            to_call=row["to_call"],
            stacks=row["stacks"],
            me=player_id,
            hole=row.get("hole", []),
            board=row["board"],
            position=position,
            opponents=row.get("opponents"),
            acting_opponents=acting_opponents,
            memory_features=memory_features,
            starting_chips=self.starting_chips,
        )

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(label),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def train_model(
    model,
    train_loader,
    val_loader,
    lr=1e-3,
    epochs=8,
    device="cpu",
    class_weights=None,
    feature_mean=None,
    feature_std=None,
):
    model = model.to(device)

    # Class-weighted loss
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Apply feature normalization
            if feature_mean is not None:
                x = (x - feature_mean.to(device)) / (feature_std.to(device) + 1e-8)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        # ---- Validation ----
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                if feature_mean is not None:
                    x = (x - feature_mean.to(device)) / (feature_std.to(device) + 1e-8)

                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        avg_val = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        accuracy = 100 * correct / total if total > 0 else 0.0

        # LR scheduling — manually log when the LR changes (replaces the
        # removed verbose=True which crashes on torch >= 2.2)
        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr != prev_lr:
            print(f"  [LR] Reduced from {prev_lr:.2e} → {current_lr:.2e}")

        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {avg_train:.4f} | "
              f"Val Loss: {avg_val:.4f} | "
              f"Val Acc:  {accuracy:.2f}% | "
              f"LR: {current_lr:.2e}")

        # Model checkpointing: save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            save_dir = os.path.join(project_root, "models")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(make_checkpoint(best_state),
                       os.path.join(save_dir, "ml_model_best.pt"))
            print(f"  -> New best model saved (val_loss={avg_val:.4f})")

    # Restore best weights for return
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def compute_confusion_matrix(model, loader, num_classes, device, feature_mean, feature_std):
    """Compute and print per-class accuracy (confusion matrix)."""
    ACTION_NAMES = ["fold", "check", "call", "raise_small", "raise_med", "raise_large"]
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if feature_mean is not None:
                x = (x - feature_mean.to(device)) / (feature_std.to(device) + 1e-8)
            preds = model(x).argmax(dim=1)
            for t, p in zip(y, preds):
                confusion[t.item(), p.item()] += 1

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX  (rows=true, cols=predicted)")
    print("=" * 60)

    header = f"{'':>14s}" + "".join(f"{n:>10s}" for n in ACTION_NAMES)
    print(header)
    for i, name in enumerate(ACTION_NAMES):
        row_total = confusion[i].sum().item()
        acc = 100 * confusion[i, i].item() / row_total if row_total > 0 else 0.0
        row_str = f"{name:>14s}" + "".join(f"{confusion[i,j].item():>10d}" for j in range(num_classes))
        row_str += f"  | acc {acc:5.1f}%  (n={row_total})"
        print(row_str)

    total_correct = sum(confusion[i, i].item() for i in range(num_classes))
    total_samples = confusion.sum().item()
    print(f"\nOverall accuracy: {100*total_correct/total_samples:.2f}% ({total_correct}/{total_samples})")


if __name__ == "__main__":
    import argparse
    import sys
    from torch.utils.data import DataLoader, random_split

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Folder containing JSONL logs")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--filter_players", type=str, nargs="+", default=None,
                        help="Only train on decisions from these players (e.g., --filter_players P3)")
    parser.add_argument("--filter_winners", action="store_true",
                        help="Only train on decisions from winning hands")
    parser.add_argument("--allow-legacy-logs", action="store_true",
                        help="Accept log files without a session header "
                             "(per-hand/legacy logs). Memory features from "
                             "such files are untrustworthy — prefer "
                             "regenerating with run_local_match.py --log-session")
    args = parser.parse_args()

    print("Loading logs...")
    dataset = PokerDataset(
        log_folder=args.log_dir,
        filter_players=args.filter_players,
        filter_winners_only=args.filter_winners,
        allow_unmarked_sessions=args.allow_legacy_logs,
    )

    if len(dataset) == 0:
        print(f"Error: No training data found in {args.log_dir}")
        sys.exit(1)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch)

    print(f"Training samples: {train_size}  |  Validation samples: {val_size}")
    if args.filter_players:
        print(f"Filtered to players: {args.filter_players}")
    if args.filter_winners:
        print("Filtered to winning hands only")

    sample_x, _ = dataset[0]
    input_size = sample_x.shape[0]

    model = PokerMLP(input_dim=input_size, hidden=128, num_classes=6)

    print("Starting training...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device
    )

    # ---- SAVE MODEL ----
    save_path = "models/ml_model.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(make_checkpoint(model.state_dict()), save_path)

    print(f"\nModel saved to: {save_path} "
          f"(feature schema v{FEATURE_SCHEMA_VERSION})")
