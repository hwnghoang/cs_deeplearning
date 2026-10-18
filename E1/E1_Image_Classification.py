# Import standard libraries
import argparse, json, time, logging
from pathlib import Path
import numpy as np

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Fashion-MNIST train-set statistics
MEAN, STD = 0.2860, 0.3530
# CUDA if available, else CPU
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# --------------------------------------------------------- Load data
def load_data(seed, bs=128):
    raw = ROOT.parent / "data"
    train = datasets.FashionMNIST(raw, train=True, download=True)
    test = datasets.FashionMNIST(raw, train=False, download=True)

    prep = lambda x: ((x.float() / 255) - MEAN) / STD
    x_train, y_train = prep(train.data).unsqueeze(1), train.targets
    x_test, y_test = prep(test.data).unsqueeze(1), test.targets
    full = TensorDataset(x_train, y_train)

    # split into train and validation
    train, val = random_split(full, [55000, 5000], generator = torch.Generator().manual_seed(seed))
    mk = lambda ds, sh: DataLoader(ds, batch_size=bs if sh else 1000, shuffle=sh)
    return mk(train, True), mk(val, False), mk(TensorDataset(x_test, y_test), False)

# --------------------------------------------------------- Models
# flatten -> linear -> (softmax inside the loss)
class Softmax(nn.Module):  
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x.flatten(1))

# 784->512->256->10, ReLU + dropout
class MLP(nn.Module): 
    def __init__(self, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(), nn.Dropout(p),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(p),
            nn.Linear(256, 10))

    def forward(self, x):
        return self.net(x.flatten(1))

# 3 x (conv-BN-ReLU-pool) + GAP-free small FC head
class CNN(nn.Module): 
    def __init__(self, p=0.3):
        super().__init__()
        blk = lambda i, o: [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                            nn.MaxPool2d(2)]
        self.features = nn.Sequential(*blk(1, 32), *blk(32, 64), *blk(64, 128)) # 28->14->7->3
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(p), nn.Linear(128 * 3 * 3, 128), nn.ReLU(),
                                  nn.Dropout(p), nn.Linear(128, 10))

    def forward(self, x):
        return self.head(self.features(x))

MODELS = {"Softmax": Softmax, "MLP": MLP, "CNN": CNN}
# lr per model: linear model likes a larger step, deeper gets a smaller one
LR = {"Softmax": 1e-2, "MLP": 1e-3, "CNN": 1e-3}

def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# --------------------------------------------------------- Training and evaluation
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss = correct = n = 0
    preds, ys = [], []
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV)
        out = model(x)
        loss += F.cross_entropy(out, y, reduction="sum").item()
        p = out.argmax(1)
        correct += (p == y).sum().item()
        n += len(y)
        preds.append(p.cpu()); ys.append(y.cpu())
    return loss / n, correct / n, torch.cat(preds), torch.cat(ys)

def train(name, seed, epochs, loaders, log):
    torch.manual_seed(seed)
    tr, va, _ = loaders
    model = MODELS[name]().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR[name], weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    best, best_state, t0 = -1, None, time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tl = tc = n = 0
        for x, y in tr:
            x, y = x.to(DEV), y.to(DEV)
            out = model(x)                  # forward
            loss = F.cross_entropy(out, y)  # loss (log-softmax + NLL)
            opt.zero_grad()
            loss.backward()                 # backward
            opt.step()                      # update
            tl += loss.item() * len(y); tc += (out.argmax(1) == y).sum().item(); n += len(y)
        sched.step()
        vl, vacc, _, _ = evaluate(model, va)
        for k, v in zip(hist, [tl / n, tc / n, vl, vacc]):
            hist[k].append(v)
        if vacc > best: # keep the best-on-validation weights (model selection never touches test)
            best, best_state = vacc, {k: v.clone() for k, v in model.state_dict().items()}
        log.info(f"{name} s{seed} ep {ep:02d}/{epochs} train {tl/n:.4f}/{tc/n:.4f} val {vl:.4f}/{vacc:.4f}")
    model.load_state_dict(best_state)
    return model, hist, time.time() - t0

# --------------------------------------------------------- Evaluation and plotting
def confusion(preds, ys):
    cm = np.zeros((10, 10), int)
    np.add.at(cm, (ys.numpy(), preds.numpy()), 1)
    return cm

def plot_all(res, models, test_loader):
    h = {n: res[n]["hist"] for n in MODELS}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for n, c in zip(MODELS, ["C0", "C1", "C2"]):
        ep = range(1, len(h[n]["train_loss"]) + 1)
        ax[0].plot(ep, h[n]["train_loss"], c, ls="--", label=f"{n} train"); ax[0].plot(ep, h[n]["val_loss"], c, label=f"{n} val")
        ax[1].plot(ep, h[n]["train_acc"], c, ls="--", label=f"{n} train"); ax[1].plot(ep, h[n]["val_acc"], c, label=f"{n} val")
    ax[0].set(title="Loss", xlabel="epoch", ylabel="cross-entropy"); ax[1].set(title="Accuracy", xlabel="epoch", ylim=(0.78, 1.0))
    for a in ax: a.legend(fontsize=7); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "curves.png", dpi=150); plt.close()

    xs = torch.cat([x for x, _ in test_loader])
    per_class = {}
    for n, m in models.items():
        _, _, p, y = evaluate(m, test_loader)
        cm = confusion(p, y)
        per_class[n] = (cm.diagonal() / cm.sum(1)).tolist()
        fig, a = plt.subplots(figsize=(7, 6))
        a.imshow(cm, cmap="Blues"); a.set(title=f"{n} confusion matrix (test)", xlabel="predicted", ylabel="true",
                                          xticks=range(10), yticks=range(10))
        a.set_xticklabels(CLASSES, rotation=60, ha="right", fontsize=7); a.set_yticklabels(CLASSES, fontsize=7)
        for i in range(10):
            for j in range(10):
                a.text(j, i, cm[i, j], ha="center", va="center", fontsize=6, color="w" if cm[i, j] > cm.max() / 2 else "k")
        plt.tight_layout(); plt.savefig(OUT / f"confusion_{n}.png", dpi=150); plt.close()
        # top confused pairs + 12 wrong samples
        off = cm.copy(); np.fill_diagonal(off, 0)
        pairs = sorted(((off[i, j], CLASSES[i], CLASSES[j]) for i in range(10) for j in range(10)), reverse=True)[:5]
        res[n]["top_confusions"] = [f"{t} -> {q}: {c}" for c, t, q in pairs]
        wrong = (p != y).nonzero().flatten()[:12]
        fig, axs = plt.subplots(2, 6, figsize=(12, 4.5))
        for a, i in zip(axs.flat, wrong):
            a.imshow(xs[i, 0] * STD + MEAN, cmap="gray"); a.axis("off")
            a.set_title(f"T:{CLASSES[y[i]]}\nP:{CLASSES[p[i]]}", fontsize=7, color="r")
        plt.suptitle(f"{n}: first 12 test errors"); plt.tight_layout(); plt.savefig(OUT / f"misclassified_{n}.png", dpi=150); plt.close()
    fig, a = plt.subplots(figsize=(10, 4)); w = .27
    for k, n in enumerate(MODELS):
        a.bar(np.arange(10) + (k - 1) * w, per_class[n], w, label=n)
    a.set(xticks=range(10), ylim=(0.5, 1.0), title="Per-class test accuracy"); a.set_xticklabels(CLASSES, rotation=40, ha="right", fontsize=8)
    a.legend(); a.grid(alpha=.3, axis="y"); plt.tight_layout(); plt.savefig(OUT / "per_class_acc.png", dpi=150); plt.close()
    for n in MODELS: res[n]["per_class_acc"] = dict(zip(CLASSES, per_class[n]))

# --------------------------------------------------------- Exploratory Data Analysis (EDA)
def eda(loaders, log):
    # Class balance, samples, per-class mean image, pixel stats, class-mean similarity.
    cat = lambda i: tuple(torch.cat(t) for t in zip(*[(x, y) for x, y in i]))
    (xtr, ytr), (xva, yva), (xte, yte) = (cat(l) for l in loaders)
    raw = xtr * STD + MEAN # back to [0,1]
    counts = {k: torch.bincount(y, minlength=10).tolist() for k, y in [("train", ytr), ("val", yva), ("test", yte)]}
    stats = {"n": {"train": len(ytr), "val": len(yva), "test": len(yte)}, "class_counts": counts,
             "pixel_mean": raw.mean().item(), "pixel_std": raw.std().item(),
             "min": raw.min().item(), "max": raw.max().item(),
             "background_frac(px<0.05)": (raw < 0.05).float().mean().item()}
    means = torch.stack([raw[ytr == c].mean(0)[0] for c in range(10)])
    v = F.normalize(means.flatten(1), dim=1)
    sim = v @ v.T                          # cosine similarity between class-mean images
    off = sim - 2 * torch.eye(10)
    top = sorted(((off[i, j].item(), CLASSES[i], CLASSES[j]) for i in range(10) for j in range(1, 10)), reverse=True)[:5]
    stats["most_similar_class_means"] = [f"{a} - {b}: {s:.3f}" for s, a, b in top]
    log.info(f"EDA: {json.dumps(stats, indent=1)}")
    json.dump(stats, open(OUT / "eda.json", "w"), indent=1)

    # 1) samples (8 per class) + class-mean image
    fig, axs = plt.subplots(10, 9, figsize=(9, 11))
    for c in range(10):
        idx = (ytr == c).nonzero().flatten()[:8]
        for j in range(8):
            axs[c, j].imshow(raw[idx[j], 0], cmap="gray")
            axs[c, 0].imshow(means[c], cmap="viridis")
        axs[c, 0].set_ylabel(CLASSES[c], rotation=0, ha="right", va="center", fontsize=8)
        for a in axs.flat: a.set_xticks([]); a.set_yticks([])
    axs[0, 0].set_title("mean", fontsize=8)
    plt.suptitle("Fashion-MNIST: 8 samples per class + class-mean image"); plt.tight_layout()
    plt.savefig(OUT / "eda_samples.png", dpi=150); plt.close()

    # 2) class balance per split, pixel histogram, class-mean similarity
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for k, (name, cn) in enumerate(counts.items()):
        ax[0].bar(np.arange(10) + (k - 1) * .27, np.array(cn) / sum(cn) * 100, .27, label=f"{name} (n={sum(cn)})")
    ax[0].axhline(10, c="k", ls=":", lw=1)
    ax[0].set(title="Class share per split (%; dotted = uniform 10%)", xticks=range(10)); ax[0].set_xticklabels(CLASSES, rotation=60, ha="right", fontsize=7)
    ax[0].legend(fontsize=7, loc="lower right")
    ax[1].hist(raw.flatten().numpy(), bins=50, log=True); ax[1].set(title="Pixel intensity histogram (log y)", xlabel="intensity")
    im = ax[2].imshow(sim, cmap="viridis", vmin=sim[~torch.eye(10, dtype=bool)].min().item(), vmax=1)
    ax[2].set(title="Cosine similarity of class-mean images", xticks=range(10), yticks=range(10))
    ax[2].set_xticklabels(CLASSES, rotation=60, ha="right", fontsize=7); ax[2].set_yticklabels(CLASSES, fontsize=7)
    plt.colorbar(im, ax=ax[2], fraction=.046); plt.tight_layout()
    plt.savefig(OUT / "eda_stats.png", dpi=150); plt.close()

# --------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--eda-only", action="store_true", help="only run the EDA section (seconds)")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(OUT / ("eda.log" if a.eda_only else "train.log"), "w", "utf-8"), logging.StreamHandler()])
    log = logging.getLogger("e1")
    log.info(f"device={DEV} torch={torch.__version__} epochs={a.epochs} seeds={a.seeds}")
    for n, m in MODELS.items(): log.info(f"{n}: {n_params(m())} params")

    res = {n: {"params": n_params(m()), "runs": []} for n, m in MODELS.items()}
    if a.eda_only:
        return eda(load_data(a.seeds[0]), log)
    for seed in a.seeds:
        loaders = load_data(seed)
        if seed == a.seeds[0]: eda(loaders, log)
        for n in MODELS:
            model, hist, secs = train(n, seed, a.epochs, loaders, log)
            t1, ta, _, _ = evaluate(model, loaders[2])
            log.info(f"[{n} s{seed}] TEST loss {t1:.4f} acc {ta:.4f} time {secs:.0f}s")
            res[n]["runs"].append({"seed": seed, "test_acc": ta, "test_loss": t1, "best_val_acc": max(hist["val_acc"]),
                                   "train_time_s": secs, "hist": hist})
            if seed == a.seeds[0]:
                res[n]["hist"] = hist; res[n]["model"] = model
                torch.save(model.state_dict(), OUT / f"{n}.pt")
    models = {n: res[n].pop("model") for n in MODELS}
    plot_all(res, models, load_data(a.seeds[0])[2])
    for n in MODELS:
        r = res[n]["runs"]
        res[n]["test_acc_mean"] = float(np.mean([x["test_acc"] for x in r]))
        res[n]["test_acc_std"] = float(np.std([x["test_acc"] for x in r]))
        res[n]["train_time_mean_s"] = float(np.mean([x["train_time_s"] for x in r]))
        log.info(f"{n}: params {res[n]['params']:,} test acc {res[n]['test_acc_mean']:.4f} +- {res[n]['test_acc_std']:.4f}"
                 f" time {res[n]['train_time_mean_s']:.0f}s")
    json.dump(res, open(OUT / "results.json", "w"), indent=1)

if __name__ == "__main__":
    main()