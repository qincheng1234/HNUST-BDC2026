"""Run A1, A2, and A3 sequentially, then summarize their OOF metrics."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ("A1", "A2", "A3")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to run train.py (default: current interpreter)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Run later experiments even if an earlier training process fails",
    )
    return parser.parse_args()


def run_experiment(python_executable, experiment):
    environment = os.environ.copy()
    environment["FEATURE_EXPERIMENT"] = experiment
    command = [python_executable, "code/src/train.py"]
    print(f"\n===== {experiment}: {' '.join(command)} =====")
    return subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode


def main():
    args = parse_args()
    print(f"DATA_MODE={os.environ.get('DATA_MODE', 'local_split')}")
    failures = []
    for experiment in EXPERIMENTS:
        return_code = run_experiment(args.python, experiment)
        if return_code:
            failures.append((experiment, return_code))
            if not args.continue_on_error:
                break

    summary_command = [args.python, "scripts/summarize_feature_ablation.py"]
    subprocess.run(summary_command, cwd=ROOT, check=False)
    if failures:
        details = ", ".join(f"{experiment} (exit {code})" for experiment, code in failures)
        raise SystemExit(f"Feature ablation failed: {details}")


if __name__ == "__main__":
    main()
