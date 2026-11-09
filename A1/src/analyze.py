"""Tables and figures for the A1.2 report, from results/*.json and results/*_test.npz.

    cd A1
    python -m src.analyze

Writes into results/: summary.md, summary.json and fig_*.png.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .data import IMAGES, SPLITS

A1 = Path(__file__).resolve().parents[1]
RES = A1 / "results"
ARCHS = {"resnet50": "ResNet-50", "deit_s": "DeiT-S"}
MODES = {"scratch": "From-scratch", "linear": "Freeze backbone", "partial": "Freeze half", "full": "Full finetune"}


def load_runs():
    runs = {}
    for f in sorted(RES.glob("*.json")):
        if f.name != "summary.json":
            r = json.loads(f.read_text())
            runs[(r["arch"], r["mode"])] = r
    return runs


def confusion(y, p, k):
    cm = np.zeros((k, k), int)
    np.add.at(cm, (y, p), 1)
    return cm


def epoch_seconds(r):
    """Median epoch time: the first epoch also pays for DataLoader worker start-up and
    cuDNN autotuning, so the mean would overstate the steady-state cost."""
    return float(np.median([h["epoch_time_s"] for h in r["history"]]))


def table(runs):
    head = ("| Model | Mode | Trainable / total params | Epochs | Best val top-1 | Test top-1 | Test top-5 | "
            "Test macro-F1 | s/epoch (median) | Train time | Peak GPU mem | Inference img/s |")
    lines = [head, "|" + "---|" * 12]
    for arch in ARCHS:
        for mode in MODES:
            r = runs.get((arch, mode))
            if r is None:
                continue
            t = r["test"]
            lines.append(f"| {ARCHS[arch]} | {MODES[mode]} | {r['trainable_params']/1e6:.2f}M / {r['total_params']/1e6:.2f}M | "
                         f"{r['epochs']} | {r['best_val_top1']*100:.2f} | {t['top1']*100:.2f} | {t['top5']*100:.2f} | "
                         f"{t['macro_f1']*100:.2f} | {epoch_seconds(r):.0f} | {r['train_time_s']/60:.1f} min | "
                         f"{r['peak_mem_gb']:.1f} GB | {r['infer_img_per_s']:.0f} |")
    return "\n".join(lines)


def plot_curves(runs):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for col, arch in enumerate(ARCHS):
        for i, mode in enumerate(MODES):
            r = runs.get((arch, mode))
            if r is None:
                continue
            ep = [h["epoch"] for h in r["history"]]
            axes[0, col].plot(ep, [h["train_loss"] for h in r["history"]], f"C{i}--", lw=1)
            axes[0, col].plot(ep, [h["val_loss"] for h in r["history"]], f"C{i}-", label=MODES[mode])
            axes[1, col].plot(ep, [h["train_top1"] * 100 for h in r["history"]], f"C{i}--", lw=1)
            axes[1, col].plot(ep, [h["val_top1"] * 100 for h in r["history"]], f"C{i}-o", ms=3, label=MODES[mode])
        axes[0, col].set(title=f"{ARCHS[arch]}: loss (solid val, dashed train)", xlabel="epoch")
        axes[1, col].set(title=f"{ARCHS[arch]}: top-1 % (solid val, dashed train)", xlabel="epoch", ylim=(0, 100))
    for ax in axes.flat:
        ax.grid(alpha=.3); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(RES / "fig_curves.png", dpi=150); plt.close()


def plot_cost(runs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    markers = {"resnet50": "o", "deit_s": "s"}
    for (arch, mode), r in runs.items():
        i = list(MODES).index(mode)
        acc = r["test"]["top1"] * 100
        for ax, x in zip(axes, (r["trainable_params"] / 1e6, r["train_time_s"] / 60)):
            ax.scatter(x, acc, c=f"C{i}", marker=markers[arch], s=60)
            ax.annotate(f"{ARCHS[arch]}\n{MODES[mode]}", (x, acc), fontsize=6.5, xytext=(4, 4),
                        textcoords="offset points")
    axes[0].set(xscale="log", xlabel="trainable parameters (millions, log scale)", ylabel="test top-1 (%)",
                title="Accuracy vs trainable parameters")
    axes[1].set(xlabel="total training time on GTX 1070 (min)", ylabel="test top-1 (%)",
                title="Accuracy vs training time")
    for ax in axes:
        ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(RES / "fig_acc_vs_cost.png", dpi=150); plt.close()


def error_analysis(runs):
    """For the best run of each architecture: confusion matrix, top confused pairs,
    worst classes, and a grid of misclassified test images."""
    test_items = (SPLITS / "test.txt").read_text().split()
    out = {}
    for arch in ARCHS:
        cands = [r for (a, m), r in runs.items() if a == arch]
        if not cands:
            continue
        best = max(cands, key=lambda r: r["test"]["top1"])
        d = np.load(RES / f"{best['run']}_test.npz")
        y, p, classes = d["y"], d["p"], list(d["classes"])
        cm = confusion(y, p, len(classes))
        per_class = cm.diagonal() / cm.sum(1)
        off = cm.copy(); np.fill_diagonal(off, 0)
        pairs = sorted(((off[i, j], classes[i], classes[j]) for i, j in zip(*np.nonzero(off))), reverse=True)[:10]
        order = np.argsort(per_class)
        out[arch] = {"run": best["run"], "test_top1": best["test"]["top1"],
                     "top_confusions": [{"true": t, "pred": q, "count": int(c)} for c, t, q in pairs],
                     "worst_classes": [{"class": classes[i], "acc": float(per_class[i])} for i in order[:5]],
                     "best_classes": [{"class": classes[i], "acc": float(per_class[i])} for i in order[::-1][:5]]}

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        ax.set(title=f"{ARCHS[arch]} ({MODES[best['mode']]}): row-normalized confusion matrix, 101 classes",
               xlabel="predicted class index", ylabel="true class index")
        ax.title.set_fontsize(9)
        plt.colorbar(im, fraction=0.046); plt.tight_layout()
        plt.savefig(RES / f"fig_confusion_{arch}.png", dpi=150); plt.close()

        wrong = [i for i in range(len(y)) if y[i] != p[i]]
        rng = np.random.default_rng(0)
        pick = rng.choice(wrong, size=min(12, len(wrong)), replace=False)
        fig, axes = plt.subplots(2, 6, figsize=(13, 5))
        for ax, i in zip(axes.flat, pick):
            with Image.open(IMAGES / f"{test_items[i]}.jpg") as img:
                ax.imshow(img.convert("RGB"))
            ax.set_title(f"true: {classes[y[i]]}\npred: {classes[p[i]]}", fontsize=7, color="r")
            ax.axis("off")
        plt.suptitle(f"{ARCHS[arch]} ({MODES[best['mode']]}): random misclassified test images")
        plt.tight_layout(); plt.savefig(RES / f"fig_errors_{arch}.png", dpi=120); plt.close()
    return out


def plot_eda():
    """Sample images (one per class for 12 classes) and the size distribution of the original images."""
    items = (SPLITS / "train.txt").read_text().split()
    classes = sorted({i.split("/")[0] for i in items})
    raw = A1.parent / "data" / "food-101" / "images"
    first = {}
    for it in items:
        first.setdefault(it.split("/")[0], it)
    fig, axes = plt.subplots(2, 6, figsize=(13, 4.8))
    for ax, c in zip(axes.flat, classes[::9][:12]):
        with Image.open(IMAGES / f"{first[c]}.jpg") as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(c, fontsize=8); ax.axis("off")
    plt.suptitle("Food-101: one training image for 12 of the 101 classes")
    plt.tight_layout(); plt.savefig(RES / "fig_eda_samples.png", dpi=120); plt.close()

    sizes = []
    for it in items[::5]:                                 # every 5th train image is enough for the histogram
        with Image.open(raw / f"{it}.jpg") as img:
            sizes.append(img.size)
    w, h = np.array(sizes).T
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(np.maximum(w, h), bins=30, color="C0"); axes[0].set(title="long side (px)")
    axes[1].hist(w / h, bins=30, color="C1"); axes[1].set(title="aspect ratio w/h")
    plt.suptitle(f"Original image sizes ({len(sizes)} sampled training images)")
    plt.tight_layout(); plt.savefig(RES / "fig_eda_sizes.png", dpi=120); plt.close()
    return {"sampled": len(sizes), "long_side_max": int(np.maximum(w, h).max()),
            "long_side_median": float(np.median(np.maximum(w, h))), "share_square": float(np.mean(w == h))}


def main():
    runs = load_runs()
    md = table(runs)
    (RES / "summary.md").write_text(md + "\n")
    print(md)
    plot_curves(runs)
    plot_cost(runs)
    summary = {"errors": error_analysis(runs), "eda": plot_eda()}
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
