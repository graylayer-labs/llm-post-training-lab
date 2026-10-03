"""`lab` command line: one sub-command per stage, each driven by a YAML file."""

from __future__ import annotations

import argparse
import json

from post_training.config import SftConfig, load_config


def main() -> None:
    p = argparse.ArgumentParser(prog="lab")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("sft", "dpo", "eval"):
        s = sub.add_parser(name)
        s.add_argument("--config", required=True)
    a = p.parse_args()

    if a.cmd == "sft":
        from post_training.train.sft import run_sft

        summary = run_sft(load_config(a.config, SftConfig))
        summary.pop("log_history")
        print(json.dumps(summary, indent=1))
    else:
        raise SystemExit(f"{a.cmd}: not implemented yet")
