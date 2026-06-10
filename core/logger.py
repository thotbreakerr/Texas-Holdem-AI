# core/logger.py

import json
import os
from datetime import datetime

# Class-level counter so every DecisionLogger instance in this process
# gets a unique filename even when created within the same microsecond.
_logger_counter = 0

# Log-format version written into session headers. v2 = decision rows carry
# "position" and session-scoped files start with a {"session_start": ...} row.
LOG_FORMAT_VERSION = 2


class DecisionLogger:
    def __init__(self, enabled=True, directory="logs", session_scoped=False):
        """
        session_scoped=True marks this file as ONE full session/tournament:
        a {"session_start": ...} header row is written so ML training
        (PokerDataset) can trust the file for cumulative opponent-memory
        replay. Create one such logger per tournament and pass it to every
        Table.play_hand(..., logger=...) call of that tournament; close it
        exactly once at the end.

        Without session_scoped (the engine's internal per-hand fallback),
        no header is written and training treats the file as legacy data.
        """
        global _logger_counter
        self.enabled = enabled
        self.directory = directory
        self.file = None
        self.path = None
        self.hand_id = 0  # increments every hand
        self.session_id = None

        if enabled:
            os.makedirs(directory, exist_ok=True)
            # Include PID + monotonic counter to guarantee unique filenames
            # across processes and fast sequential hand creation.
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            pid = os.getpid()
            seq = _logger_counter
            _logger_counter += 1
            self.session_id = f"{ts}_p{pid}_{seq}"
            self.path = f"{directory}/session_{self.session_id}.jsonl"
            self.file = open(self.path, "a")
            if session_scoped:
                self.file.write(json.dumps({
                    "session_start": {
                        "session_id": self.session_id,
                        "log_format_version": LOG_FORMAT_VERSION,
                    }
                }) + "\n")
                self.file.flush()

    def start_hand(self, hand_id: int):
        """Set hand ID at start of each hand."""
        self.hand_id = hand_id

    def log_decision(self, entry: dict):
        if not self.enabled:
            return

        entry = dict(entry)  # copy
        entry["hand_id"] = self.hand_id

        self.file.write(json.dumps(entry) + "\n")
        self.file.flush()

    def log_result(self, pid: str, net: float):
        """Log the hand outcome for each player."""
        if not self.enabled:
            return

        entry = {
            "hand_id": self.hand_id,
            "result": {
                "player": pid,
                "net": net
            }
        }
        self.file.write(json.dumps(entry) + "\n")
        self.file.flush()

    def flush(self):
        if self.file:
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()
