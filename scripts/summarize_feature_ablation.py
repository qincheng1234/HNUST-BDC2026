"""Generate an OOF-only summary for completed A1/A2/A3 artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "code" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment_reporting import write_ablation_summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", default="model", help="Directory containing model artifacts")
    parser.add_argument(
        "--output",
        default="output/feature_ablation_oof_summary.csv",
        help="CSV path for the generated OOF summary",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = write_ablation_summary(args.model_root, args.output)
    for row in rows:
        print(f"{row['experiment']}: {row['status']}")
    print(f"OOF-only summary written to: {args.output}")


if __name__ == "__main__":
    main()
