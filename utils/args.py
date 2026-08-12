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


def parse(
    extra: list[tuple[str, dict[str, Any]]] | None = None,
) -> argparse.Namespace:
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
    parser.add_argument("--data_dir", default="data/EV-UAV-dataset",help="path to dataset root directory")
    parser.add_argument("--model_path", default="",help="path to pretrained checkpoint (for test)")

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

    # ── training ────────────────────────────────────────────────────
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--optim", default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--k", default=3, type=int,
                        help="STC loss k-nearest neighbours")
    parser.add_argument("--t", default=5, type=int,
                        help="STC loss time threshold")
    parser.add_argument("--seed", default=37, type=int)

    # ── evaluation ──────────────────────────────────────────────────
    parser.add_argument("--eval", default=True, type=bool)
    parser.add_argument("--vis", default=False, type=bool)
    parser.add_argument("--roc", default=False, type=bool)
    parser.add_argument("--pd_detT", default=50, type=int,
                        help="detection time window for ROC")
    parser.add_argument("--correct_thresh", default=0.0001, type=float)
    parser.add_argument("--save", default=True, type=bool)

    if extra:
        for flag, kwargs in extra:
            parser.add_argument(flag, **kwargs)

    cfg = parser.parse_args()

    return cfg
