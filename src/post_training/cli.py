"""`lab` command line: one sub-command per stage, each driven by a YAML file."""

from __future__ import annotations

import argparse
import json

from post_training.config import EvalConfig, SftConfig, load_config


def main() -> None:
    p = argparse.ArgumentParser(prog="lab")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("sft", "dpo", "eval"):
        s = sub.add_parser(name)
        s.add_argument("--config", required=True)
        if name == "sft":
            s.add_argument(
                "--resume",
                action="store_true",
                help="continue from the last checkpoint in <output_dir>/checkpoints",
            )
            s.add_argument(
                "--scratch",
                action="store_true",
                help="allow resuming on a different commit or a dirty tree",
            )
    a = p.parse_args()

    if a.cmd == "sft":
        from post_training.train import sft

        try:
            summary = sft.run_sft(
                load_config(a.config, SftConfig), resume=a.resume, scratch=a.scratch
            )
        except sft.ResumeError as e:
            raise SystemExit(f"error: {e}") from e
        summary.pop("log_history")
        print(json.dumps(summary, indent=1))
    elif a.cmd == "eval":
        from post_training.eval import harness

        results = harness.run_eval(load_config(a.config, EvalConfig))
        print(harness.render_markdown(results))
    else:
        raise SystemExit(f"{a.cmd}: not implemented yet")
