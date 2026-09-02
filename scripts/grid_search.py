#!/usr/bin/env python3
"""Sequential Cartesian-product grid search for EV-UAV training.

Examples:
    # 2 losses x 2 motion modes = 4 runs
    uv run python scripts/grid_search.py \
      --gpu 0 --epochs 50 --seeds 37 \
      --losses stc,mbtc --motion_gds none,shallow

    # Preview commands without launching training
    uv run python scripts/grid_search.py --dry_run \
      --losses stc,mbtc --trajectory_weights 0.1,0.25,0.5

Values are comma-separated. Every combination is run sequentially with
``subprocess`` (no shell), and a manifest is written for reproducibility.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _csv(text: str, cast):
    values = [x.strip() for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("value list cannot be empty")
    try:
        return [cast(x) for x in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid list {text!r}") from exc


def _flag(name: str, value) -> list[str]:
    return [name, str(value)]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run sequential Cartesian-product EV-UAV training sweeps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gpu", default="0")
    p.add_argument("--data_dir", default="/data/ev-uav")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--train_workers", type=int, default=0)
    p.add_argument("--seeds", type=lambda x: _csv(x, int), default=[37])
    p.add_argument("--losses", type=lambda x: _csv(x, str), default=["mbtc"])
    p.add_argument("--motion_gds", type=lambda x: _csv(x, str), default=["shallow"])
    p.add_argument("--patch_attentions", type=lambda x: _csv(x, str), default=["sequence"])
    p.add_argument("--trajectory_weights", type=lambda x: _csv(x, float), default=[0.25])
    p.add_argument("--support_weights", type=lambda x: _csv(x, float), default=[1.0])
    p.add_argument("--motion_direction_weights", type=lambda x: _csv(x, float), default=[0.1])
    p.add_argument("--motion_speed_weights", type=lambda x: _csv(x, float), default=[0.1])
    p.add_argument("--route_weights", type=lambda x: _csv(x, float), default=[0.05])
    p.add_argument("--route_temperatures", type=lambda x: _csv(x, float), default=[1.0])
    p.add_argument("--patience", type=lambda x: _csv(x, int), default=[10])
    p.add_argument("--max_events_num", type=lambda x: _csv(x, int), default=[700000])
    p.add_argument("--run_prefix", default="grid")
    p.add_argument("--output_dir", default="runs/grid_searches")
    p.add_argument("--max_runs", type=int, default=0,
                   help="limit combinations (0 means all)")
    p.add_argument("--dry_run", action="store_true",
                   help="print combinations and commands without training")
    p.add_argument("--continue_on_error", action=argparse.BooleanOptionalAction,
                   default=True)
    args = p.parse_args()

    fields = {
        "seed": args.seeds,
        "loss": args.losses,
        "motion_gd": args.motion_gds,
        "patch_attention": args.patch_attentions,
        "trajectory_weight": args.trajectory_weights,
        "support_weight": args.support_weights,
        "motion_direction_weight": args.motion_direction_weights,
        "motion_speed_weight": args.motion_speed_weights,
        "route_weight": args.route_weights,
        "route_temperature": args.route_temperatures,
        "early_stopping_patience": args.patience,
        "max_events_num": args.max_events_num,
    }
    keys = list(fields)
    combinations = list(itertools.product(*(fields[k] for k in keys)))
    if args.max_runs > 0:
        combinations = combinations[:args.max_runs]
    print(f"[grid] combinations: {len(combinations)}")
    if not combinations:
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_dir = ROOT / args.output_dir / f"{args.run_prefix}_{stamp}"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / "manifest.jsonl"

    for index, values in enumerate(combinations, 1):
        config = dict(zip(keys, values))
        tag = "_".join([
            f"{config['loss']}", f"gd-{config['motion_gd']}",
            f"attn-{config['patch_attention']}", f"tw-{config['trajectory_weight']}",
            f"s{config['seed']}",
        ]).replace(".", "p")
        run_name = f"{args.run_prefix}_{index:03d}_{tag}"
        command = [
            sys.executable, "train.py", "--gpu", args.gpu,
            "--data_dir", args.data_dir, "--epochs", str(args.epochs),
            "--train_workers", str(args.train_workers), "--run_name", run_name,
            "--seed", str(config["seed"]), "--loss", config["loss"],
            "--motion_gd", config["motion_gd"],
            "--patch_attention", config["patch_attention"],
            "--trajectory_weight", str(config["trajectory_weight"]),
            "--support_weight", str(config["support_weight"]),
            "--motion_direction_weight", str(config["motion_direction_weight"]),
            "--motion_speed_weight", str(config["motion_speed_weight"]),
            "--route_weight", str(config["route_weight"]),
            "--route_temperature", str(config["route_temperature"]),
            "--early_stopping_patience", str(config["early_stopping_patience"]),
            "--max_events_num", str(config["max_events_num"]),
        ]
        record = {"index": index, "total": len(combinations), "config": config,
                  "run_name": run_name, "command": command,
                  "started_at": datetime.now().isoformat()}
        print(f"\n[grid] ({index}/{len(combinations)}) {run_name}")
        print("[grid] command:", " ".join(command))
        if args.dry_run:
            record["status"] = "dry_run"
            with manifest.open("a") as f:
                f.write(json.dumps(record) + "\n")
            continue

        started = time.time()
        try:
            result = subprocess.run(command, cwd=ROOT, check=False)
            record["returncode"] = result.returncode
            record["status"] = "ok" if result.returncode == 0 else "failed"
        except KeyboardInterrupt:
            record["status"] = "interrupted"
            raise
        finally:
            record["elapsed_sec"] = round(time.time() - started, 2)
            with manifest.open("a") as f:
                f.write(json.dumps(record) + "\n")
        if record["status"] == "failed" and not args.continue_on_error:
            print("[grid] stopping after failed run")
            return int(record["returncode"])

    print(f"\n[grid] finished; manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
