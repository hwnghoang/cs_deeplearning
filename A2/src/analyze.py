"""EDA, tables and figures for the A2.2 report.

    cd A2
    python -m src.analyze --eda      # dataset statistics only (no results needed)
    python -m src.analyze            # everything: EDA + results + ablation figures

Writes into results/: eda.json, summary.md, fig_*.png.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .voc import CLASSES, VOC, read_split, parse_annotation

A2 = Path(__file__).resolve().parents[1]
RES = A2 / "results"
NAMES = {"V1_frcnn_coco": "V1 Faster R-CNN (COCO)", "V2_detr_coco": "V2 DETR (COCO)",
         "V3_detr_imagenet": "V3 DETR (ImageNet)", "V5_frcnn_imagenet": "V5 Faster R-CNN (ImageNet)"}
COLORS = {"V1_frcnn_coco": "C0", "V2_detr_coco": "C1", "V3_detr_imagenet": "C3", "V5_frcnn_imagenet": "C2"}


# ----------------------------------------------------------------------------- EDA
def eda():
    stats = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for split, color in (("train", "C0"), ("test", "C1")):
        anns = [parse_annotation(i) for i in read_split(split)]
        labels = torch.cat([a["labels"] for a in anns]).numpy()
        diff = torch.cat([a["difficult"] for a in anns]).numpy()
        boxes = torch.cat([a["boxes"] for a in anns])
        area = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).numpy()
        per_img = np.array([len(a["labels"]) for a in anns])
        counts = np.bincount(labels, minlength=20)
        stats[split] = {"images": len(anns), "objects": int(len(labels)), "difficult": int(diff.sum()),
                        "objects_per_image_mean": float(per_img.mean()), "objects_per_image_max": int(per_img.max()),
                        "small_lt_32sq": float((area < 32 ** 2).mean()), "medium": float(((area >= 32 ** 2) & (area < 96 ** 2)).mean()),
                        "large_ge_96sq": float((area >= 96 ** 2).mean()),
                        "per_class": dict(zip(CLASSES, counts.tolist()))}
        x = np.arange(20) + (0.2 if split == "test" else -0.2)
        axes[0].bar(x, counts, width=0.4, color=color, label=split)
        axes[1].hist(np.clip(per_img, 0, 20), bins=np.arange(1, 22) - 0.5, alpha=0.6, color=color, label=split)
        axes[2].hist(np.sqrt(area), bins=50, alpha=0.6, color=color, label=split)
    axes[0].set_xticks(range(20)); axes[0].set_xticklabels(CLASSES, rotation=70, fontsize=7)
    axes[0].set(title="objects per class", yscale="log")
    axes[1].set(title="objects per image (clipped at 20)", xlabel="objects")
    axes[2].axvline(32, color="k", ls="--", lw=1); axes[2].axvline(96, color="k", ls="--", lw=1)
    axes[2].set(title="sqrt(box area) in px  (COCO small < 32 < medium < 96 < large)", xlabel="px")
    for ax in axes:
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(RES / "fig_eda.png", dpi=140); plt.close()
    (RES / "eda.json").write_text(json.dumps(stats, indent=1))
    return stats


# ----------------------------------------------------------------------------- results
def load_runs():
    return {f.stem: json.loads(f.read_text()) for f in sorted(RES.glob("V*.json"))}


def table(runs):
    lines = ["| Run | Trainable / total params | Epochs (best) | Val mAP@0.5 | Test mAP@0.5 | VOC07 11-pt | AP@[.5:.95] "
             "| AP75 | AP_S | AP_M | AP_L | Train time | Peak GPU mem | FPS |", "|" + "---|" * 14]
    for name, r in runs.items():
        t = r["test"]
        lines.append(f"| {NAMES.get(name, name)} | {r['trainable_params']/1e6:.1f}M / {r['total_params']/1e6:.1f}M | "
                     f"{r['epochs']} ({r['best_epoch']}) | {r['best_val_map50']*100:.1f} | {t['map50_area']*100:.1f} | "
                     f"{t['map50_voc07_11pt']*100:.1f} | {t['AP']*100:.1f} | {t['AP75']*100:.1f} | {t['APs']*100:.1f} | "
                     f"{t['APm']*100:.1f} | {t['APl']*100:.1f} | {r['train_time_s']/3600:.2f} h | {r['peak_mem_gb']:.1f} GB | "
                     f"{r['fps']:.1f} |")
    return "\n".join(lines)


def plot_curves(runs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, r in runs.items():
        ep = [h["epoch"] for h in r["history"]]
        axes[0].plot(ep, [h["train_loss"] for h in r["history"]], color=COLORS.get(name), marker=".", label=NAMES.get(name, name))
        axes[1].plot(ep, [h["val_map50"] * 100 for h in r["history"]], color=COLORS.get(name), marker="o", ms=3,
                     label=NAMES.get(name, name))
    axes[0].set(title="training loss (each model's own loss, not comparable across models)", xlabel="epoch", yscale="log")
    axes[1].set(title="validation mAP@0.5 (%)", xlabel="epoch", ylim=(0, 100))
    for ax in axes:
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    axes[0].title.set_fontsize(9)
    plt.tight_layout(); plt.savefig(RES / "fig_curves.png", dpi=140); plt.close()


def plot_per_class(runs):
    names = [n for n in runs if n in ("V1_frcnn_coco", "V2_detr_coco")]
    if len(names) < 2:
        return
    fig, ax = plt.subplots(figsize=(13, 4))
    order = np.argsort(runs["V1_frcnn_coco"]["test"]["ap50_per_class"])
    for k, n in enumerate(names):
        ap = np.array(runs[n]["test"]["ap50_per_class"])[order] * 100
        ax.bar(np.arange(20) + (k - 0.5) * 0.4, ap, width=0.4, color=COLORS[n], label=NAMES[n])
    ax.set_xticks(range(20)); ax.set_xticklabels(np.array(CLASSES)[order], rotation=60, fontsize=8)
    ax.set(ylabel="AP@0.5 (%)", title="Per-class AP@0.5 on VOC 2007 test (classes sorted by Faster R-CNN AP)", ylim=(0, 100))
    ax.grid(axis="y", alpha=.3); ax.legend()
    plt.tight_layout(); plt.savefig(RES / "fig_per_class.png", dpi=140); plt.close()


def plot_sizes(runs):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    keys = ("APs", "APm", "APl")
    for k, (n, r) in enumerate(runs.items()):
        vals = [r["test"][key] * 100 for key in keys]
        ax.bar(np.arange(3) + (k - (len(runs) - 1) / 2) * 0.2, vals, width=0.2, color=COLORS.get(n), label=NAMES.get(n, n))
    ax.set_xticks(range(3)); ax.set_xticklabels(["small (< 32²)", "medium", "large (> 96²)"])
    ax.set(ylabel="AP@[.5:.95] (%)", title="AP by object size (COCO definition)")
    ax.grid(axis="y", alpha=.3); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(RES / "fig_ap_by_size.png", dpi=140); plt.close()


def plot_examples(runs, n_images=4, thr=0.5):
    """The same test images: ground truth, then the detections (score >= thr) of V1 and V2."""
    show = [n for n in ("V1_frcnn_coco", "V2_detr_coco") if (RES / f"{n}_test.pt").exists()]
    if not show:
        return
    data = {n: torch.load(RES / f"{n}_test.pt", weights_only=False) for n in show}
    ids = data[show[0]]["ids"]
    rng = np.random.default_rng(3)
    multi = [i for i, x in enumerate(ids) if len(parse_annotation(x)["labels"]) >= 3]
    pick = rng.choice(multi, n_images, replace=False)
    fig, axes = plt.subplots(n_images, 1 + len(show), figsize=(4.2 * (1 + len(show)), 3.3 * n_images))
    for row, i in enumerate(pick):
        img = Image.open(VOC / "JPEGImages" / f"{ids[i]}.jpg").convert("RGB")
        gt = parse_annotation(ids[i])
        panels = [("ground truth", gt["boxes"], gt["labels"], None)]
        for n in show:
            p = data[n]["preds"][i]
            keep = p["scores"] >= thr
            panels.append((NAMES[n], p["boxes"][keep], p["labels"][keep], p["scores"][keep]))
        for col, (title, boxes, labels, scores) in enumerate(panels):
            ax = axes[row, col]; ax.imshow(img); ax.axis("off")
            for j, (b, l) in enumerate(zip(boxes.tolist(), labels.tolist())):
                ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, color=f"C{l % 10}", lw=1.5))
                txt = CLASSES[l] + ("" if scores is None else f" {scores[j]:.2f}")
                ax.text(b[0], b[1], txt, fontsize=6, color="w", bbox=dict(facecolor=f"C{l % 10}", pad=1, lw=0))
            if row == 0:
                ax.set_title(title + ("" if col == 0 else f" (score ≥ {thr})"), fontsize=9)
    plt.tight_layout(); plt.savefig(RES / "fig_examples.png", dpi=110); plt.close()


def plot_ablation():
    f = RES / "ablation.json"
    if not f.exists():
        return
    abl = json.loads(f.read_text())
    runs = [k for k in abl if k.endswith("_per_layer")]
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 3.8), squeeze=False)
    for ax, key in zip(axes[0], runs):
        rows = abl[key]
        for nms, ls in ((False, "-"), (True, "--")):
            r = [x for x in rows if x["nms"] == nms]
            ax.plot([x["layer"] for x in r], [x["AP"] * 100 for x in r], "C0" + ls, marker="o", label=f"AP{' + NMS' if nms else ''}")
            ax.plot([x["layer"] for x in r], [x["map50"] * 100 for x in r], "C1" + ls, marker="s", label=f"AP50{' + NMS' if nms else ''}")
        ax.set(title=f"{NAMES[key.replace('_per_layer', '')]}: AP after each decoder layer", xlabel="decoder layer",
               ylabel="%"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(RES / "fig_decoder_layers.png", dpi=140); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eda", action="store_true")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_class"} for k, v in eda().items()}, indent=1))
    if a.eda:
        return
    runs = load_runs()
    md = table(runs)
    (RES / "summary.md").write_text(md + "\n")
    print(md)
    plot_curves(runs); plot_per_class(runs); plot_sizes(runs); plot_examples(runs); plot_ablation()


if __name__ == "__main__":
    main()
