"""EV-UAV training script.

Usage::

    uv run python train.py                          # GPU 0, defaults
    uv run python train.py --gpu 1 --epochs 100     # GPU 1, 100 epochs
    uv run python train.py --data_dir /path/to/data # custom dataset
"""

import torch
import torch.optim as optim
import tqdm

from dataset.ev_uav import EvUAV
from model.evspsegnet import evspsegnet
from utils import args
from utils.eval import evalute, run_test
from utils.run_manager import RunManager
from utils.seed import set_seed
from utils.stcloss import STCLoss
from utils.tacloss import TrajectoryAwareContrastiveLoss
from utils.mbtcloss import MotionBidirectionalTrajectoryLoss


def build_optimizer(net, cfg):
    parameters = filter(lambda parameter: parameter.requires_grad, net.parameters())
    if cfg.optim == "Adam":
        return optim.Adam(parameters, lr=cfg.lr)
    return optim.SGD(parameters, lr=cfg.lr, momentum=0.9, weight_decay=1e-4)


def build_scheduler(optimizer, cfg):
    if cfg.scheduler == "none":
        return None
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1,
        )
    if not 0 < cfg.final_lr <= cfg.lr:
        raise ValueError("linear scheduler requires 0 < final_lr <= lr")
    return torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=cfg.final_lr / cfg.lr,
        total_iters=max(cfg.epochs - 1, 1),
    )


def main():
    args.parse()

    device = f"cuda:{args.cfg.gpu}"
    torch.cuda.set_device(device)

    # ── run manager ─────────────────────────────────────────────────
    run = RunManager(root="runs", prefix=f"train_{args.cfg.run_name}")
    run.save_config(args.cfg)
    print(f"[train] Run directory: {run.run_dir}")

    # ── reproducibility ─────────────────────────────────────────────
    set_seed(args.cfg.seed)
    run.log_metric(0, None, seed=args.cfg.seed)

    # ── model ───────────────────────────────────────────────────────
    net = evspsegnet(args.cfg).train().to(device)

    # ── data ────────────────────────────────────────────────────────
    dataset = EvUAV(args.cfg, mode="train")
    train_sampler = torch.utils.data.sampler.RandomSampler(range(len(dataset)))
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.cfg.batch_size,
        collate_fn=dataset.custom_collate, sampler=train_sampler,
        num_workers=args.cfg.train_workers,
        pin_memory=True,
    )

    val_dataset = EvUAV(args.cfg, mode="val")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.cfg.batch_size,
        collate_fn=val_dataset.custom_collate,
        num_workers=args.cfg.train_workers,
        pin_memory=True,
    )
    evaluator = evalute(args.cfg)

    # ── loss / optim / scheduler ────────────────────────────────────
    if args.cfg.loss == "stc":
        criterion = STCLoss(k=args.cfg.k, t=args.cfg.t, cfg=args.cfg).to(device)
    elif args.cfg.loss == "tacl":
        criterion = TrajectoryAwareContrastiveLoss(args.cfg).to(device)
    else:
        criterion = MotionBidirectionalTrajectoryLoss(args.cfg).to(device)
    optimizer = build_optimizer(net, args.cfg)
    scheduler = build_scheduler(optimizer, args.cfg)

    # ── training loop ───────────────────────────────────────────────
    best_loss = 1e5
    best_iou = 0.0

    for epoch in range(args.cfg.epochs):
        net.train()
        epoch_loss = 0.0
        pbar = tqdm.tqdm(
            total=len(train_dataloader), unit="Batch", unit_scale=True,
            desc=f"Epoch: {epoch}", position=0, leave=True,
        )

        for batch_idx, ev in enumerate(train_dataloader):
            x = dataset.voxelize_to_sparse(ev, device)
            label = ev["seg_label"].float().to(device)
            p2v_map = ev["p2v_map"].long().to(device)
            ev_locs = ev["locs"].to(device)
            instance_ids = torch.as_tensor(
                ev["idx_label"], device=device, dtype=torch.long,
            )
            motion_target = ev["motion_target"].to(device)
            motion_valid = ev["motion_valid"].to(device)

            preds, voxel, auxiliary = net(x)
            if args.cfg.loss == "stc":
                loss = criterion(voxel, p2v_map, preds, label)
                components = {}
            else:
                loss, components = criterion(
                    voxel, p2v_map, preds, label, auxiliary,
                    ev_locs, instance_ids, motion_target, motion_valid,
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            pbar.set_postfix(
                loss=loss.item(), lr=optimizer.param_groups[0]["lr"],
            )
            pbar.update(1)

            with torch.no_grad():
                run.log_metric(
                    epoch, batch_idx, loss=loss.item(),
                    lr=optimizer.param_groups[0]["lr"], **components,
                )

        epoch_loss /= max(len(train_dataloader), 1)
        if epoch_loss < best_loss:
            run.save_checkpoint(net, f"best_loss_seed{args.cfg.seed}.pt")
            best_loss = epoch_loss
        if scheduler is not None:
            scheduler.step()

        # ---- validation ----
        if (
            epoch < args.cfg.val_start_epoch
            or (epoch - args.cfg.val_start_epoch) % args.cfg.val_interval != 0
        ):
            continue

        net.eval()
        evaluator.matches.clear()
        with torch.no_grad():
            for sample, ev in enumerate(val_dataloader):
                x = val_dataset.voxelize_to_sparse(ev, device)
                label = ev["seg_label"].float().to(device)
                p2v_map = ev["p2v_map"].long().to(device)

                preds, _, _ = net(x)
                preds = preds[p2v_map].squeeze().cpu()

                evaluator.matches[str(sample)] = {
                    "seg_pred": preds,
                    "seg_gt": label.cpu(),
                    "trajectory_ids": ev["idx_label"],
                    "trajectory_times": ev["locs"][:, 3].cpu(),
                    "trajectory_batches": ev["locs"][:, 0].cpu(),
                }

            iou = evaluator.evaluate_semantic_segmantation_miou(
                thresh=args.cfg.prediction_thresh,
            )
            seg_acc = evaluator.evaluate_semantic_segmantation_accuracy(
                thresh=args.cfg.prediction_thresh, device=device,
            )
            trajectory_metrics = evaluator.evaluate_trajectory_metrics(
                thresh=args.cfg.prediction_thresh,
                bin_ms=args.cfg.trajectory_bin_ms,
                correct_thresh=args.cfg.trajectory_correct_thresh,
            )
            run.log_metric(
                epoch, None, epoch_loss=epoch_loss, iou=iou.item(),
                seg_acc=seg_acc.item(), **trajectory_metrics,
            )

            if iou.item() > best_iou:
                run.save_checkpoint(net, f"best_iou_seed{args.cfg.seed}.pt")
                best_iou = iou.item()

    run.log_metric(args.cfg.epochs, None, best_loss=best_loss, best_iou=best_iou)
    print(f"[train] Done. Best loss={best_loss:.4f}, best IoU={best_iou:.4f}")

    # ── optional test on best checkpoint ───────────────────────────
    best_ckpt = run.ckpt_dir / f"best_iou_seed{args.cfg.seed}.pt"
    if args.cfg.test_after_train and best_ckpt.exists():
        print(f"[train] Running evaluation on best checkpoint...")
        results = run_test(str(best_ckpt), args.cfg, device=device)
        run.log_metric(
            args.cfg.epochs,
            None,
            test_iou=results.get("iou", float("nan")),
            test_seg_acc=results.get("seg_acc", float("nan")),
            test_pd=results.get("pd", float("nan")),
            test_fa=results.get("fa", float("nan")),
            test_trajectory_coverage=results.get(
                "trajectory_coverage", float("nan"),
            ),
            test_trajectory_longest_ratio=results.get(
                "trajectory_longest_ratio", float("nan"),
            ),
            test_trajectory_fragmentation=results.get(
                "trajectory_fragmentation", float("nan"),
            ),
        )
        if "pd" in results:
            print(f"[train] Test — iou={results['iou']:.4f}  "
                  f"seg_acc={results['seg_acc']:.4f}  "
                  f"pd={results['pd']:.4f}  fa={results['fa']:.6g}  "
                  f"coverage={results['trajectory_coverage']:.4f}  "
                  f"longest={results['trajectory_longest_ratio']:.4f}  "
                  f"fragmentation={results['trajectory_fragmentation']:.4f}")
        else:
            print(f"[train] Test — iou={results['iou']:.4f}  "
                  f"seg_acc={results['seg_acc']:.4f}")
    elif args.cfg.test_after_train:
        print(f"[train] No best-iou checkpoint found, skipping evaluation.")
    else:
        print("[train] Test split not evaluated (use --test_after_train explicitly).")

    print(f"[train] Results saved to: {run.run_dir}")
    run.close()


if __name__ == "__main__":
    main()
