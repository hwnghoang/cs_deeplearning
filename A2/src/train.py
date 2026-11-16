"""Train and evaluate one detector variant on VOC 2007.

    cd A2
    python -m src.train --run V2_detr_coco --arch detr --init coco --epochs 20 --lr_drop 15

After every epoch: VOC mAP@0.5 on the validation split; the best epoch is saved to
checkpoints/<run>.pt. At the end that checkpoint is evaluated once on the test split
(VOC and COCO metrics, FPS). Results: results/<run>.json, predictions: results/<run>_test.pt.
"""
import argparse, json, math, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .voc import VOCDetection, collate
from .models import build, count_params
from .evaluate import voc_ap, coco_ap

A2 = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda")

# Optimizer settings: Faster R-CNN as in torchvision's reference recipe (SGD), DETR as in its paper / main.py.
DEFAULTS = {"frcnn": dict(opt="sgd", lr=0.01, lr_backbone=0.01, wd=1e-4, clip=0.0, warmup=500),
            "detr": dict(opt="adamw", lr=1e-4, lr_backbone=1e-5, wd=1e-4, clip=0.1, warmup=0)}


def to_device(images, targets):
    images = [im.to(DEVICE, non_blocking=True) for im in images]
    targets = [{k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
    return images, targets


@torch.no_grad()
def predict_split(det, loader, amp, **kw):
    det.eval()
    preds, gts = [], []
    for images, targets in loader:
        images, _ = to_device(images, targets)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = det.predict(images, **kw)
        preds += [{k: v.float().cpu() for k, v in o.items()} for o in out]
        gts += list(targets)
    for p in preds:
        p["labels"] = p["labels"].long()
    return preds, gts


@torch.no_grad()
def measure_fps(det, dataset, amp, n=200):
    """Images per second at batch size 1, test resolution, including resize/normalize and post-processing."""
    det.eval()
    n = min(n, len(dataset))
    imgs = [dataset[i][0].to(DEVICE) for i in range(n)]
    for im in imgs[:10]:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            det.predict([im])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for im in imgs:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            det.predict([im])
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--arch", choices=["frcnn", "detr"], required=True)
    ap.add_argument("--init", choices=["coco", "imagenet"], required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--lr_drop", type=int, required=True, help="epoch after which the lr is divided by 10")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--no_aux", action="store_true", help="DETR without auxiliary decoding losses")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    # fp32 by default: on this GTX 1070 (Pascal) fp16 autocast was measured slower for both detectors
    # (Faster R-CNN 0.83 vs 0.71 s/step, DETR 0.60 vs 0.53 s/step at batch 4).
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only the first N images of each split")
    a = ap.parse_args()
    cfg = DEFAULTS[a.arch]
    amp = a.amp
    out_dir, ck_dir = A2 / "results", A2 / "checkpoints"
    out_dir.mkdir(exist_ok=True)
    ck_dir.mkdir(exist_ok=True)

    torch.manual_seed(a.seed)
    train_ds, val_ds, test_ds = VOCDetection("train", True), VOCDetection("val", False), VOCDetection("test", False)
    if a.limit:
        for ds in (train_ds, val_ds, test_ds):
            ds.ids = ds.ids[:a.limit]
    train_dl = DataLoader(train_ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, collate_fn=collate,
                          pin_memory=True, persistent_workers=a.workers > 0, generator=torch.Generator().manual_seed(a.seed))
    val_dl = DataLoader(val_ds, batch_size=8, num_workers=a.workers, collate_fn=collate,
                        pin_memory=True, persistent_workers=a.workers > 0)
    test_dl = DataLoader(test_ds, batch_size=8, num_workers=a.workers, collate_fn=collate, pin_memory=True)

    det = build(a.arch, a.init, aux_loss=not a.no_aux).to(DEVICE)
    groups = det.param_groups(cfg["lr"], cfg["lr_backbone"], cfg["wd"])
    opt = (torch.optim.SGD(groups, momentum=0.9) if cfg["opt"] == "sgd" else torch.optim.AdamW(groups))
    base_lrs = [g["lr"] for g in groups]
    scaler = torch.amp.GradScaler(enabled=amp)
    trainable, total = count_params(det)
    print(f"[{a.run}] {a.arch}/{a.init}: trainable {trainable:,} / {total:,}, {len(train_ds)} train images, "
          f"{a.epochs} epochs, lr drop after {a.lr_drop}, amp={amp}", flush=True)

    hist, best, step = [], None, 0
    torch.cuda.reset_peak_memory_stats()
    for ep in range(1, a.epochs + 1):
        det.train()
        t0, sums, n, skipped = time.perf_counter(), {}, 0, 0
        for images, targets in train_dl:
            images, targets = to_device(images, targets)
            # step decay after `lr_drop` epochs, plus a linear warm-up for Faster R-CNN
            factor = 0.1 if ep > a.lr_drop else 1.0
            if cfg["warmup"] and step < cfg["warmup"]:
                factor *= (step + 1) / cfg["warmup"]
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * factor
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                losses = det.loss(images, targets)                         # forward + loss
            step += 1
            if not math.isfinite(losses["loss"].item()):   # skip a diverged batch instead of corrupting the weights
                skipped += 1
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(losses["loss"]).backward()                        # backward
            if cfg["clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(det.parameters(), cfg["clip"])
            scaler.step(opt)                                               # update
            scaler.update()
            n += 1
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.item()
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        preds, gts = predict_split(det, val_dl, amp)
        val = voc_ap(preds, gts)
        hist.append({"epoch": ep, "epoch_time_s": secs, "skipped_steps": skipped,
                     **{f"train_{k}": v / max(n, 1) for k, v in sums.items()}, "val_map50": val["map_area"]})
        print(f"[{a.run}] ep {ep:02d}/{a.epochs} loss {sums.get('loss', 0) / max(n, 1):.3f}  val mAP@0.5 "
              f"{val['map_area']*100:.2f}  {secs:.0f}s" + (f"  ({skipped} non-finite steps skipped)" if skipped else ""),
              flush=True)
        if best is None or val["map_area"] > best[0]:
            best = (val["map_area"], ep)
            torch.save(det.state_dict(), ck_dir / f"{a.run}.pt")
    peak_mem = torch.cuda.max_memory_allocated() / 2**30

    det.load_state_dict(torch.load(ck_dir / f"{a.run}.pt", map_location=DEVICE))
    preds, gts = predict_split(det, test_dl, amp)
    voc = voc_ap(preds, gts)
    coco = coco_ap(preds, gts)
    fps = measure_fps(det, test_ds, amp)
    torch.save({"preds": preds, "ids": test_ds.ids}, out_dir / f"{a.run}_test.pt")
    result = {"run": a.run, "arch": a.arch, "init": a.init, "aux_loss": not a.no_aux, "epochs": a.epochs,
              "lr_drop": a.lr_drop, "batch_size": a.bs, "amp": amp, **cfg, "trainable_params": trainable,
              "total_params": total, "best_epoch": best[1], "best_val_map50": best[0],
              "test": {"map50_voc07_11pt": voc["map11"], "map50_area": voc["map_area"],
                       "ap50_per_class": voc["ap_area"].tolist(), **coco},
              "fps": fps, "train_time_s": sum(h["epoch_time_s"] for h in hist), "peak_mem_gb": peak_mem,
              "history": hist}
    (out_dir / f"{a.run}.json").write_text(json.dumps(result, indent=1))
    print(f"[{a.run}] TEST mAP@0.5 {voc['map_area']*100:.2f} (VOC07 11-pt {voc['map11']*100:.2f})  "
          f"AP {coco['AP']*100:.2f}  APs/m/l {coco['APs']*100:.1f}/{coco['APm']*100:.1f}/{coco['APl']*100:.1f}  "
          f"{fps:.1f} FPS  best epoch {best[1]}", flush=True)


if __name__ == "__main__":
    main()
