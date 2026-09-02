"""Summarize the best validation epoch of each EV-UAV experiment run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "iou",
    "seg_acc",
    "trajectory_coverage",
    "trajectory_longest_ratio",
    "trajectory_fragmentation",
)


def best_validation(log_path: Path):
    records = []
    with log_path.open() as file:
        for line in file:
            record = json.loads(line)
            if "iou" in record and "test_iou" not in record:
                records.append(record)
    return max(records, key=lambda record: record["iou"]) if records else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs", type=Path)
    args = parser.parse_args()

    header = ("run", "epoch", *METRICS)
    print("\t".join(header))
    for log_path in sorted(args.runs.glob("train_*/metrics.jsonl")):
        record = best_validation(log_path)
        if record is None:
            continue
        values = [log_path.parent.name, str(record.get("epoch", ""))]
        values.extend(
            f"{float(record.get(metric, float('nan'))):.6f}"
            for metric in METRICS
        )
        print("\t".join(values))


if __name__ == "__main__":
    main()

