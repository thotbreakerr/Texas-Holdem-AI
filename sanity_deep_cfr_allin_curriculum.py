"""Compatibility entry point for the schema-v2 Deep CFR regression gate.

The v1 shadow/staged/full all-in curriculum no longer exists. The validation
ladder still invokes this historical filename, so route it to the replacement
gate while accepting the old ``--mode`` argument.
"""
from __future__ import annotations

import argparse

from sanity_deep_cfr_v2 import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fast", "micro"), default="fast")
    parser.parse_args()
    raise SystemExit(0 if run() else 1)


if __name__ == "__main__":
    main()
