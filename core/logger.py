# core/logger.py

import json
import os
from datetime import datetime

# Class-level counter so every DecisionLogger instance in this process
# gets a unique filename even when created within the same microsecond.
_logger_counter = 0

class DecisionLogger:
    def __init__(self, enabled=True, directory="logs"):
        global _logger_counter
        self.enabled = enabled
        self.directory = directory
        self.file = None
        self.hand_id = 0  # increments every hand

        if enabled:
            os.makedirs(directory, exist_ok=True)
            # Include PID + monotonic counter to guarantee unique filenames
            # across processes and fast sequential hand creation.
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            pid = os.getpid()
            seq = _logger_counter
            _logger_counter += 1
            self.file = open(
                f"{directory}/session_{ts}_p{pid}_{seq}.jsonl", "a"
            )

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
