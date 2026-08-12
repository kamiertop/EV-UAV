"""EV-UAV evaluation script (standalone).

Usage::

    uv run python test.py --model_path runs/train_xxx/checkpoints/best_iou_seed37.pt
"""

import os

from utils import args
from utils.eval import run_test


def main():
    args.parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cfg.gpu

    device = f"cuda:{args.cfg.gpu}"
    results = run_test(args.cfg.model_path, args.cfg, device=device)

    if "pd" in results:
        print(f"iou={results['iou']:.4f}  seg_acc={results['seg_acc']:.4f}  "
              f"pd={results['pd']:.4f}  fa={results['fa']:.4f}")
    else:
        print(f"iou={results['iou']:.4f}  seg_acc={results['seg_acc']:.4f}")


if __name__ == "__main__":
    main()
