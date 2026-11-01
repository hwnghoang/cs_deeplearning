# E1 - Fashion-MNIST: Softmax vs MLP vs CNN, training loop

# Import standard libraries
import argparse, json, time
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
# train set Fashion-MNIST
MEAN, STD = 0.2860, 0.3530         
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(seed, bs=128):
    raw = ROOT.parent / "data"
    train = datasets.FashionMNIST(raw, train = True, download = True)
    test = datasets.FashionMNIST(raw, train = False, download = True)
    # Image pre-processing
    # convert the pixel values to [0,1] and normalize to zero-mean unit-variance
    # Shift the center to 0 and set the deviation level to 1    
    prep = lambda x: ((x.float() / 255) - MEAN) / STD

    # Split the training set into train and validation sets
    train, val = random_split(TensorDataset(prep(train.data).unsqueeze(1), train.targets),
                              [55000, 5000], generator=torch.Generator().manual_seed(seed))

    # [10000, 28, 28] -> [10000, 1, 28, 28] for conv2d
    test = TensorDataset(prep(test.data).unsqueeze(1), test.targets)
    mk = lambda ds, sh: DataLoader(ds, batch_size = bs if sh else 1000, shuffle = sh)
    return mk(train, True), mk(val, False), mk(test, False)

# flatten -> linear -> (softmax inside the loss)
class Softmax(nn.Module):
    def __init__(self):
        super().__init__()
        # 28x28=784 input features, 10 output classes
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        # input tensor x [batch, 1, 28, 28] -> flatten to [batch, 784] -> linear layer -> [batch, 10]
        return self.fc(x.flatten(1))
# 784->512->256->10, ReLU + dropout
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 28 x 28 = 784 input features, 512 hidden units, ReLU activation, dropout
        # 256 hidden units, ReLU activation, dropout
        # 10 output classes
        self.net = nn.Sequential(nn.Linear(784, 256), 
                                 nn.ReLU(), 
                                 nn.Dropout(0.2), 
                                 nn.Linear(256, 10))

    def forward(self, x):
        # input tensor x [batch, 1, 28, 28] -> flatten to [batch, 784] -> linear layer -> [batch, 10]
        return self.net(x.flatten(1))

# 2 x [conv-BN-ReLU-pool] + head
class CNN(nn.Module):                
    def __init__(self):
        super().__init__()
        # 28x28x1 -> 14x14x32 -> 7x7x64 -> flatten -> 10
        block = lambda i, o: [
                nn.Conv2d(i, o, 3, padding=1, bias=False), 
                nn.BatchNorm2d(o), 
                nn.ReLU(), 
                nn.MaxPool2d(2)]
        
        self.net = nn.Sequential(*block(1, 32), 
                                 *block(32, 64),
                                 nn.Flatten(), 
                                 nn.Dropout(0.2), 
                                 nn.Linear(64 * 7 * 7, 10))

    def forward(self, x):
        return self.net(x)

MODELS = {"Softmax": Softmax, "MLP": MLP, "CNN": CNN}
learning_rate = {"Softmax": 1e-2, "MLP": 1e-3, "CNN": 1e-3}
# total number of trainable model parameters
n_params = lambda m: sum(p.numel() for p in m.parameters())

@torch.no_grad()
def evaluate(model, loader):
    # loss, acc, preds, labels
    model.eval()
    loss = correct = n = 0
    preds, ys = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss += F.cross_entropy(out, y, reduction = "sum").item()
        p = out.argmax(1)
        correct += (p == y).sum().item()
        n += len(y)
        preds.append(p.cpu()); ys.append(y.cpu())
    return loss / n, correct / n, torch.cat(preds), torch.cat(ys)

def train(name, seed, epochs, loaders, log):
    # Training loop: forward -> loss -> backward() -> optimizer.step().
    torch.manual_seed(seed)
    tr, va, _ = loaders
    model = MODELS[name]().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr = learning_rate[name])
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tl = tc = n = 0
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)                   # forward
            loss = F.cross_entropy(out, y)   # loss = log-softmax + NLL
            opt.zero_grad()
            loss.backward()                  # backward
            opt.step()                       # update
            tl += loss.item() * len(y); tc += (out.argmax(1) == y).sum().item(); n += len(y)
        vl, va_acc, _, _ = evaluate(model, va)
        for k, v in zip(hist, (tl / n, tc / n, vl, va_acc)):
            hist[k].append(v)
        log(f"[{name}] ep {ep:02d}/{epochs} train {tl/n:.4f}/{tc/n:.4f}  val {vl:.4f}/{va_acc:.4f}")
    return model, hist, time.time() - t0

def confusion(preds, ys):
    cm = np.zeros((10, 10), int)
    np.add.at(cm, (ys.numpy(), preds.numpy()), 1)
    return cm

def plot_curves(hists, path, title = "E1"):
    fig, ax = plt.subplots(1, 2, figsize = (11, 4))
    for i, (n, h) in enumerate(hists.items()):
        ep = range(1, len(h["val_acc"]) + 1)
        ax[0].plot(ep, h["train_loss"], f"C{i}^", ls = "--", lw = .9, label = f"{n} train")
        ax[0].plot(ep, h["val_loss"], f"C{i}", label = f"{n} val")
        ax[1].plot(ep, h["train_acc"], f"C{i}^", ls = "--", lw = .9)
        ax[1].plot(ep, h["val_acc"], f"C{i}", label = n)
    ax[0].set(title=f"{title}: loss", xlabel="epoch"); ax[1].set(title = f"{title}: accuracy", xlabel = "epoch")
    for a in ax: a.legend(fontsize=7); a.grid(alpha = .3)
    plt.tight_layout(); plt.savefig(path, dpi = 150); plt.close()

def plot_errors(name, model, test, out_dir):
    # Confusion matrix + 8 images of common classification errors
    _, _, p, y = evaluate(model, test)
    cm = confusion(p, y)
    fig, a = plt.subplots(figsize=(6.5, 5.5))
    a.imshow(cm, cmap="Blues")
    a.set(title=f"{name}: confusion matrix (test)", xlabel="predicted", ylabel="true",
          xticks=range(10), yticks=range(10))
    a.set_xticklabels(CLASSES, rotation=60, ha="right", fontsize=7); a.set_yticklabels(CLASSES, fontsize=7)
    for i in range(10):
        for j in range(10):
            a.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                   color="w" if cm[i, j] > cm.max() / 2 else "k")
    plt.tight_layout(); plt.savefig(out_dir / f"confusion_{name}.png", dpi=150); plt.close()

    xs = torch.cat([x for x, _ in test])
    wrong = (p != y).nonzero().flatten()[:8]
    fig, axs = plt.subplots(1, 8, figsize=(13, 2.4))
    for a, i in zip(axs, wrong):
        a.imshow(xs[i, 0] * STD + MEAN, cmap="gray"); a.axis("off")
        a.set_title(f"true: {CLASSES[y[i]]}\npredicted: {CLASSES[p[i]]}", fontsize=6, color="r")
    plt.suptitle(f"{name}: 8 common classification errors"); plt.tight_layout()
    plt.savefig(out_dir / f"errors_{name}.png", dpi = 150); plt.close()
    off = cm.copy(); np.fill_diagonal(off, 0)
    pairs = sorted(((off[i, j], CLASSES[i], CLASSES[j]) for i in range(10) for j in range(10)), reverse = True)[:3]
    return {"per_class_acc": dict(zip(CLASSES, (cm.diagonal() / cm.sum(1)).round(4).tolist())),
            "top_confusions": [f"{t} -> {q}: {c}" for c, t, q in pairs]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type = int, default = 10)
    ap.add_argument("--seed", type = int, default = 0)
    a = ap.parse_args()
    OUT.mkdir(exist_ok = True)
    logf = open(OUT / "train.log", "w", encoding="utf-8")
    def log(s):
        print(s); logf.write(s + "\n"); logf.flush()

    log(f"device = {DEVICE} torch={torch.__version__} epochs = {a.epochs} seed = {a.seed}")
    loaders = load_data(a.seed)
    res = {}
    for name in MODELS:
        model, hist, secs = train(name, a.seed, a.epochs, loaders, log)
        tl, ta, _, _ = evaluate(model, loaders[2])
        torch.save(model.state_dict(), OUT / f"{name}.pt")
        res[name] = {"params": n_params(model), "test_acc": ta, "test_loss": tl,
                     "best_val_acc": max(hist["val_acc"]), "train_time_s": round(secs, 1), "hist": hist,
                     **plot_errors(name, model, loaders[2], OUT)}
        log(f"{name}: {res[name]['params']:,} params  test acc {ta:.4f}  loss {tl:.4f}  {secs:.0f}s")
    plot_curves({n: res[n]["hist"] for n in res}, OUT / "curves.png")
    json.dump(res, open(OUT / "results.json", "w"), indent = 1)
    logf.close()

if __name__ == "__main__":
    main()
