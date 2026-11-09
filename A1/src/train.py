"""Train and evaluate one (architecture, mode) configuration.

    cd A1
    python -m src.train --arch resnet50 --mode full --epochs 5

Writes results/<arch>_<mode>.json (history, metrics, costs) and
results/<arch>_<mode>_test.npz (test labels and predictions, for the error analysis).
The checkpoint with the best validation accuracy is the one evaluated on test.
"""
import argparse, copy, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from .data import loaders, class_names
from .models import build, frozen_bn_to_eval, param_groups, count_params, MODES

A1 = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda")

# Protocol A1.1, table "Hyperparameter": (lr head, lr backbone, weight decay)
HPARAMS = {"scratch": (1e-3, 1e-3, 0.05), "linear": (1e-3, None, 1e-4),
           "partial": (1e-3, 1e-4, 0.05), "full": (1e-3, 1e-4, 0.05)}


@torch.no_grad()
def evaluate(model, loader, amp):
    model.eval()
    loss, ys, preds, top5 = 0.0, [], [], 0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(x)
        loss += F.cross_entropy(out.float(), y, reduction="sum").item()
        top5 += (out.topk(5, dim=1).indices == y[:, None]).any(1).sum().item()
        ys.append(y.cpu()); preds.append(out.argmax(1).cpu())
    y, p = torch.cat(ys).numpy(), torch.cat(preds).numpy()
    return {"loss": loss / len(y), "top1": float((y == p).mean()), "top5": top5 / len(y),
            "macro_f1": float(f1_score(y, p, average="macro"))}, y, p


@torch.no_grad()
def throughput(model, amp, batch=64, iters=20):
    """Inference images/second on random input (excludes data loading)."""
    model.eval()
    x = torch.randn(batch, 3, 224, 224, device=DEVICE)
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            model(x)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            model(x)
    torch.cuda.synchronize()
    return batch * iters / (time.perf_counter() - t0)


def lr_factor(step, total, warmup):
    """Linear warm-up, then cosine decay to 0 (per optimizer step)."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["resnet50", "deit_s"], required=True)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--warmup_frac", type=float, default=0.1, help="share of steps used for warm-up")
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_amp", action="store_true")
    a = ap.parse_args()
    amp = not a.no_amp
    run = f"{a.arch}_{a.mode}"
    out_dir = A1 / "results"
    out_dir.mkdir(exist_ok=True)

    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    train_dl, val_dl, test_dl = loaders(a.bs, a.workers, a.seed)
    model = build(a.arch, a.mode).to(DEVICE)
    trainable, total = count_params(model)
    lr_head, lr_backbone, wd = HPARAMS[a.mode]
    opt = torch.optim.AdamW(param_groups(model, a.mode, lr_head, lr_backbone, wd))
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler(enabled=amp)
    steps = a.epochs * len(train_dl)
    warmup = int(a.warmup_frac * steps)
    print(f"[{run}] trainable {trainable:,} / {total:,} params, {len(train_dl.dataset)} train images, "
          f"{steps} steps, warm-up {warmup}, amp={amp}", flush=True)

    hist, best, step = [], None, 0
    torch.cuda.reset_peak_memory_stats()
    for ep in range(1, a.epochs + 1):
        model.train()
        frozen_bn_to_eval(model)
        t0, tl, tc, n = time.perf_counter(), 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * lr_factor(step, steps, warmup)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out = model(x)                                              # forward
                loss = F.cross_entropy(out, y, label_smoothing=a.label_smoothing)   # loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()                                   # backward
            scaler.step(opt); scaler.update()                               # update
            step += 1
            tl += loss.item() * len(y); tc += (out.argmax(1) == y).sum().item(); n += len(y)
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        val, _, _ = evaluate(model, val_dl, amp)
        hist.append({"epoch": ep, "train_loss": tl / n, "train_top1": tc / n, "epoch_time_s": secs,
                     **{f"val_{k}": v for k, v in val.items()}})
        print(f"[{run}] ep {ep:02d}/{a.epochs} train {tl/n:.3f}/{tc/n:.4f}  val {val['loss']:.3f}/"
              f"{val['top1']:.4f}  {secs:.0f}s", flush=True)
        if best is None or val["top1"] > best[0]:
            best = (val["top1"], ep, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    peak_mem = torch.cuda.max_memory_allocated() / 2**30

    model.load_state_dict(best[2])
    test, y, p = evaluate(model, test_dl, amp)
    result = {"run": run, "arch": a.arch, "mode": a.mode, "epochs": a.epochs, "batch_size": a.bs, "amp": amp,
              "train_images": len(train_dl.dataset), "trainable_params": trainable, "total_params": total,
              "best_epoch": best[1], "best_val_top1": best[0], "test": test,
              "train_time_s": sum(h["epoch_time_s"] for h in hist),
              "epoch_time_s": float(np.mean([h["epoch_time_s"] for h in hist])),
              "peak_mem_gb": peak_mem, "infer_img_per_s": throughput(model, amp), "history": hist}
    (out_dir / f"{run}.json").write_text(json.dumps(result, indent=1))
    np.savez_compressed(out_dir / f"{run}_test.npz", y=y, p=p, classes=np.array(class_names()))
    print(f"[{run}] TEST top1 {test['top1']:.4f} top5 {test['top5']:.4f} macro-F1 {test['macro_f1']:.4f} "
          f"(best val epoch {best[1]}), {result['train_time_s']:.0f}s train, {peak_mem:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
