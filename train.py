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


def main():
    args.parse()

    device = f"cuda:{args.cfg.gpu}"

    # ── run manager ─────────────────────────────────────────────────
    run = RunManager(root="runs", prefix="train")
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
    )

    val_dataset = EvUAV(args.cfg, mode="val")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.cfg.batch_size,
        collate_fn=val_dataset.custom_collate,
    )
    evaluator = evalute(args.cfg)

    # ── loss / optim / scheduler ────────────────────────────────────
    stc_criterion = STCLoss(k=args.cfg.k, t=args.cfg.t, cfg=args.cfg).to(device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()), lr=args.cfg.lr,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1,
    )

    # ── training loop ───────────────────────────────────────────────
    best_loss = 1e5
    best_iou = 0.0

    for epoch in range(args.cfg.epochs):
        pbar = tqdm.tqdm(
            total=len(train_dataloader), unit="Batch", unit_scale=True,
            desc=f"Epoch: {epoch}", position=0, leave=True,
        )

        for batch_idx, ev in enumerate(train_dataloader):
            x = ev["voxel_ev"]
            label = ev["seg_label"].float().to(device)
            p2v_map = ev["p2v_map"].long().to(device)
            ev_locs = ev["locs"].float().requires_grad_()

            preds, voxel = net(x)
            loss = stc_criterion(voxel, p2v_map, preds, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

            with torch.no_grad():
                run.log_metric(epoch, batch_idx, loss=loss.item())
                if loss.item() < best_loss:
                    run.save_checkpoint(net, f"best_loss_seed{args.cfg.seed}.pt")
                    best_loss = loss.item()

            torch.cuda.empty_cache()

        scheduler.step()

        # ---- validation ----
        if epoch < 40:
            continue

        with torch.no_grad():
            for sample, ev in enumerate(val_dataloader):
                x = ev["voxel_ev"]
                label = ev["seg_label"].float().to(device)
                p2v_map = ev["p2v_map"].long().to(device)

                preds, voxel = net(x)
                preds = preds[p2v_map].squeeze().cpu()

                evaluator.matches[str(sample)] = {
                    "seg_pred": preds,
                    "seg_gt": label,
                }

            iou = evaluator.evaluate_semantic_segmantation_miou()
            run.log_metric(epoch, None, iou=iou.item())

            if iou.item() > best_iou:
                run.save_checkpoint(net, f"best_iou_seed{args.cfg.seed}.pt")
                best_iou = iou.item()

    run.log_metric(args.cfg.epochs, None, best_loss=best_loss, best_iou=best_iou)
    print(f"[train] Done. Best loss={best_loss:.4f}, best IoU={best_iou:.4f}")

    # ── test on best checkpoint ─────────────────────────────────────
    best_ckpt = run.ckpt_dir / f"best_iou_seed{args.cfg.seed}.pt"
    if best_ckpt.exists():
        print(f"[train] Running evaluation on best checkpoint...")
        results = run_test(str(best_ckpt), args.cfg, device=device)
        run.log_metric(args.cfg.epochs, None, test_iou=results.get("iou", float("nan")),
                       test_seg_acc=results.get("seg_acc", float("nan")))
        if "pd" in results:
            print(f"[train] Test — iou={results['iou']:.4f}  "
                  f"seg_acc={results['seg_acc']:.4f}  "
                  f"pd={results['pd']:.4f}  fa={results['fa']:.4f}")
        else:
            print(f"[train] Test — iou={results['iou']:.4f}  "
                  f"seg_acc={results['seg_acc']:.4f}")
    else:
        print(f"[train] No best-iou checkpoint found, skipping evaluation.")

    print(f"[train] Results saved to: {run.run_dir}")
    run.close()


if __name__ == "__main__":
    main()
