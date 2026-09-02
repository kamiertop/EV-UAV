"""Shared argument parser — replaces the old configs/ YAML-based system.

Usage::

    from utils import args
    args.parse()
    # access args.cfg.gpu, args.cfg.data_dir, etc.
"""

import argparse
from typing import Any

# Module-level singleton — populated by parse()
cfg: argparse.Namespace | None = None


def parse(extra: list[tuple[str, dict[str, Any]]] | None = None) -> argparse.Namespace:
    """Parse CLI arguments; store result in ``utils.args.cfg`` and return it.

    ``cfg`` is a module-level attribute, so other modules can access it with
    ``from utils import args; args.cfg`` — this is safe because module
    attribute access resolves at runtime, not at import time.
    """
    global cfg

    parser = argparse.ArgumentParser(
        description="EV-UAV Event Point Cloud Segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── GPU ─────────────────────────────────────────────────────────
    parser.add_argument("--gpu", default="0", type=str, help="CUDA device index")

    # ── paths ───────────────────────────────────────────────────────
    parser.add_argument("--data_dir", default="/data/ev-uav",
                        help="path to dataset root directory")
    parser.add_argument("--model_path", default="",
                        help="path to pretrained checkpoint (for test)")
    parser.add_argument("--run_name", default="experiment",
                        help="short tag embedded in the run directory")

    # ── data ────────────────────────────────────────────────────────
    parser.add_argument("--input_channel", default=4, type=int)
    parser.add_argument("--whole_t", default=8000, type=int,
                        help="time window in ms")
    parser.add_argument("--res", nargs=2, default=[346, 260], type=int,
                        help="sensor resolution (width height)")
    parser.add_argument("--max_events_num", default=700000, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--train_workers", default=8, type=int)

    # ── model ───────────────────────────────────────────────────────
    parser.add_argument("--width", default=12, type=int,
                        help="network width multiplier")
    parser.add_argument("--block_residual", default=True, type=bool)
    parser.add_argument("--block_reps", default=2, type=int)
    parser.add_argument("--use_coords", default=True, type=bool)
    parser.add_argument("--patch_attention", default="sequence",
                        choices=["sequence", "legacy"],
                        help="cross-patch attention or released one-token behavior")
    parser.add_argument(
        "--motion_gd", default="shallow",
        choices=["none", "shallow", "encoder", "all"],
        help="where to replace fixed GDSC by motion-conditioned GDSC",
    )
    parser.add_argument("--route_temperature", default=1.0, type=float)
    parser.add_argument(
        "--motion_speed_scale", default=1.5, type=float,
        help="maximum predicted velocity magnitude in pixels/ms",
    )
    parser.add_argument(
        "--motion_temporal_dilations", nargs=4, default=[1, 2, 4, 8], type=int,
        metavar=("T1", "T2", "T3", "T4"),
        help="temporal dilation of the four motion-conditioned branches",
    )

    # ── training ────────────────────────────────────────────────────
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=0.01, type=float,
                        help="initial learning rate")
    parser.add_argument("--final_lr", default=0.001, type=float,
                        help="final learning rate for linear decay")
    parser.add_argument("--optim", default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--scheduler", default="linear",
                        choices=["linear", "step", "none"])
    parser.add_argument(
        "--loss", default="mbtc", choices=["stc", "tacl", "mbtc"],
        help="original STC, legacy TACL prototype, or proposed MBTC loss",
    )
    parser.add_argument("--k", default=3, type=int,
                        help="spatial correlation kernel size")
    parser.add_argument("--t", default=5, type=int,
                        help="temporal correlation kernel size in voxels")
    parser.add_argument("--seed", default=37, type=int, help="random seed")
    parser.add_argument("--val_start_epoch", default=0, type=int)
    parser.add_argument("--val_interval", default=1, type=int)
    parser.add_argument(
        "--early_stopping_patience", default=10, type=int,
        help="number of validation rounds without IoU improvement before stopping; 0 disables",
    )
    parser.add_argument(
        "--early_stopping_min_delta", default=1e-4, type=float,
        help="minimum validation IoU improvement counted as progress",
    )
    parser.add_argument(
        "--test_after_train", action=argparse.BooleanOptionalAction,
        default=True,
        help="evaluate the held-out test split after model selection",
    )

    # TACL: classification, instance continuity, and local motion priors.
    parser.add_argument("--embedding_dim", default=8, type=int)
    parser.add_argument("--focal_gamma", default=2.0, type=float)
    parser.add_argument("--tversky_alpha", default=0.6, type=float,
                        help="false-positive coefficient")
    parser.add_argument("--tversky_beta", default=0.4, type=float,
                        help="false-negative coefficient")
    parser.add_argument("--pos_balance_max", default=8.0, type=float)
    parser.add_argument("--hard_pos_weight", default=1.0, type=float)
    parser.add_argument("--hard_neg_weight", default=2.0, type=float)
    parser.add_argument("--tversky_weight", default=1.0, type=float)
    parser.add_argument("--instance_weight", default=0.1, type=float)
    parser.add_argument("--motion_weight", default=0.1, type=float)
    parser.add_argument("--embedding_margin", default=1.0, type=float)
    parser.add_argument("--background_margin", default=0.75, type=float)
    parser.add_argument("--motion_bin_ms", default=50.0, type=float)
    parser.add_argument("--max_background_embeddings", default=4096, type=int)

    # MBTC: local support disagreement, trajectory coverage, and route priors.
    parser.add_argument("--support_weight", default=1.0, type=float)
    parser.add_argument("--trajectory_weight", default=0.25, type=float)
    parser.add_argument("--route_weight", default=0.05, type=float)
    parser.add_argument("--motion_direction_weight", default=0.1, type=float)
    parser.add_argument("--motion_speed_weight", default=0.1, type=float)
    parser.add_argument("--trajectory_bin_ms", default=50.0, type=float)
    parser.add_argument("--min_trajectory_events", default=2, type=int)

    # ── evaluation ──────────────────────────────────────────────────
    parser.add_argument("--eval", default=True, type=bool)
    parser.add_argument("--vis", default=False, type=bool)
    parser.add_argument("--roc", default=True, type=bool)
    parser.add_argument("--pd_detT", default=50, type=int,
                        help="detection time window for ROC")
    parser.add_argument("--correct_thresh", default=0.0001, type=float)
    parser.add_argument(
        "--trajectory_correct_thresh", default=0.1, type=float,
        help="fraction of GT events recovered for a trajectory bin to count",
    )
    parser.add_argument("--prediction_thresh", default=0.9, type=float)
    parser.add_argument("--save", default=True, type=bool)

    if extra:
        for flag, kwargs in extra:
            parser.add_argument(flag, **kwargs)

    cfg = parser.parse_args()
    globals()["cfg"] = cfg
    return cfg
