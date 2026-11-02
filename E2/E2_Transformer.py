# E2 - Fashion-MNIST: Transformer classifier with hand-written vs PyTorch MSA,
# and three tokenizations (patch, row, CNN-stem).

import argparse, json, math, time
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
MEAN, STD = 0.2860, 0.3530          # Fashion-MNIST train statistics (same as E1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- data
def load_data(seed):
    """Same split as E1: 55k train / 5k val (random_split with `seed`), 10k test."""
    raw = ROOT.parent / "data"
    train = datasets.FashionMNIST(raw, train=True, download=True)
    test = datasets.FashionMNIST(raw, train=False, download=True)
    prep = lambda x: ((x.float() / 255) - MEAN) / STD
    train, val = random_split(TensorDataset(prep(train.data).unsqueeze(1), train.targets),
                              [55000, 5000], generator=torch.Generator().manual_seed(seed))
    test = TensorDataset(prep(test.data).unsqueeze(1), test.targets)
    return train, val, test


# ---------------------------------------------------------------- attention
class ScratchMSA(nn.Module):
    """Multi-head self-attention from basic ops (Linear, einsum, softmax).

    Weight layout matches nn.MultiheadAttention: qkv.weight = [W_q; W_k; W_v] and,
    inside each of Q/K/V, head h uses features [h*dh, (h+1)*dh).
    """
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)      # one matmul for Q, K, V
        self.proj = nn.Linear(dim, dim)          # W_o, applied after concatenating heads

    def forward(self, x):                        # x: [B, N, D]
        B, N, D = x.shape
        # [B, N, 3D] -> [B, N, 3, h, dh] -> 3 tensors of [B, h, N, dh]
        q, k, v = self.qkv(x).view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        scores = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.dh)   # [B, h, N, N]
        attn = scores.softmax(dim=-1)            # each query's weights over keys sum to 1
        out = torch.einsum("bhij,bhjd->bhid", attn, v)                         # [B, h, N, dh]
        out = out.transpose(1, 2).reshape(B, N, D)                             # concat heads
        return self.proj(out)


class TorchMSA(nn.Module):
    """Reference version: same interface, backed by nn.MultiheadAttention."""
    def __init__(self, dim, heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x):
        return self.mha(x, x, x, need_weights=False)[0]


MSA = {"scratch": ScratchMSA, "torch": TorchMSA}

# ScratchMSA parameter name -> TorchMSA parameter name (identical shapes)
_KEY_MAP = {"qkv.weight": "mha.in_proj_weight", "qkv.bias": "mha.in_proj_bias",
            "proj.weight": "mha.out_proj.weight", "proj.bias": "mha.out_proj.bias"}

def scratch_to_torch_state(state):
    """Rename attention keys of a scratch state_dict (a whole model, or a single ScratchMSA)
    so the torch version can load it. Only keys under an `attn` module are renamed:
    tokenizers also have a `proj.weight` that must stay unchanged."""
    new = {}
    for key, val in state.items():
        for s, t in _KEY_MAP.items():
            if key == s or key.endswith("attn." + s):
                key = key[: -len(s)] + t
                break
        new[key] = val
    return new


# ---------------------------------------------------------------- tokenizers
# Each tokenizer maps an image [B, 1, 28, 28] to tokens [B, N, D].
class PatchTokens(nn.Module):
    """Non-overlapping p x p patches. Conv with kernel = stride = p is exactly a Linear
    layer applied to every flattened patch. 28/4 -> 7x7 = 49 tokens."""
    def __init__(self, dim, patch=4):
        super().__init__()
        self.proj = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        self.n_tokens = (28 // patch) ** 2

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)      # [B, D, 7, 7] -> [B, 49, D]


class RowTokens(nn.Module):
    """Each image row (28 pixels) is one token -> 28 tokens."""
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(28, dim)
        self.n_tokens = 28

    def forward(self, x):
        return self.proj(x[:, 0])                            # [B, 28, 28] -> [B, 28, D]


class CNNStemTokens(nn.Module):
    """Small CNN (same two blocks as the E1 CNN) -> 64 x 7 x 7 feature map;
    each spatial position is one token -> 49 tokens."""
    def __init__(self, dim):
        super().__init__()
        block = lambda i, o: [nn.Conv2d(i, o, 3, padding=1, bias=False),
                              nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2)]
        self.cnn = nn.Sequential(*block(1, 32), *block(32, 64))
        self.proj = nn.Linear(64, dim)
        self.n_tokens = 49

    def forward(self, x):
        return self.proj(self.cnn(x).flatten(2).transpose(1, 2))   # [B, 49, 64] -> [B, 49, D]


TOKENIZERS = {"patch": PatchTokens, "row": RowTokens, "cnn_stem": CNNStemTokens}


# ---------------------------------------------------------------- model
class EncoderBlock(nn.Module):
    """Pre-norm Transformer block: x + MSA(LN(x)), then x + FFN(LN(x))."""
    def __init__(self, dim, heads, msa, mlp_ratio=2, drop=0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = MSA[msa](dim, heads)
        self.drop = nn.Dropout(drop)
        self.ffn = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(mlp_ratio * dim, dim), nn.Dropout(drop))

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        return x + self.ffn(self.ln2(x))


class TransformerClassifier(nn.Module):
    """tokens -> [CLS] + learnable positional embedding -> `depth` blocks -> LN -> Linear on CLS."""
    def __init__(self, tokenizer, msa, dim=64, heads=4, depth=2):
        super().__init__()
        self.tok = TOKENIZERS[tokenizer](dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, self.tok.n_tokens + 1, dim) * 0.02)
        self.blocks = nn.Sequential(*[EncoderBlock(dim, heads, msa) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        t = self.tok(x)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], dim=1) + self.pos
        return self.head(self.norm(self.blocks(t))[:, 0])


def build_pair(tokenizer, seed, **kw):
    """Scratch and torch models with identical initial weights, so they differ only
    in the attention implementation."""
    torch.manual_seed(seed)
    scratch = TransformerClassifier(tokenizer, "scratch", **kw)
    ref = TransformerClassifier(tokenizer, "torch", **kw)
    ref.load_state_dict(scratch_to_torch_state(scratch.state_dict()))
    return scratch, ref


n_params = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------- correctness check
@torch.no_grad()
def check_msa(log, dim=64, heads=4):
    """ScratchMSA must give the same output as nn.MultiheadAttention with the same weights."""
    torch.manual_seed(0)
    a, b = ScratchMSA(dim, heads), TorchMSA(dim, heads)
    b.load_state_dict(scratch_to_torch_state(a.state_dict()))
    x = torch.randn(8, 50, dim)
    diff_msa = (a(x) - b(x)).abs().max().item()

    m1, m2 = build_pair("patch", 0)
    m1.eval(); m2.eval()
    img = torch.randn(8, 1, 28, 28)
    diff_model = (m1(img) - m2(img)).abs().max().item()
    log(f"check: max |scratch - torch| MSA output = {diff_msa:.2e}, full model logits = {diff_model:.2e}")
    assert diff_msa < 1e-5 and diff_model < 1e-4, "ScratchMSA does not match nn.MultiheadAttention"


# ---------------------------------------------------------------- train / eval
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss = correct = n = 0
    preds, ys = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss += F.cross_entropy(out, y, reduction="sum").item()
        p = out.argmax(1)
        correct += (p == y).sum().item(); n += len(y)
        preds.append(p.cpu()); ys.append(y.cpu())
    return loss / n, correct / n, torch.cat(preds), torch.cat(ys)


@torch.no_grad()
def throughput(model, loader):
    """Inference images/second over the whole loader (after one warm-up batch)."""
    model.eval()
    x0 = next(iter(loader))[0].to(DEVICE); model(x0)
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t0, n = time.perf_counter(), 0
    for x, _ in loader:
        model(x.to(DEVICE)); n += len(x)
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def train(name, model, data, a, log):
    """forward -> loss -> backward() -> optimizer.step(), with AdamW and a cosine LR schedule."""
    train_ds, val_ds, _ = data
    # Reset the global RNG (used by dropout) and seed the shuffling, so scratch vs torch
    # get identical dropout masks and identical batches.
    torch.manual_seed(a.seed)
    tr = DataLoader(train_ds, batch_size=a.bs, shuffle=True,
                    generator=torch.Generator().manual_seed(a.seed))
    va = DataLoader(val_ds, batch_size=1000)
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * len(tr))
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_s")}
    for ep in range(1, a.epochs + 1):
        model.train()
        tl = tc = n = 0
        t0 = time.perf_counter()
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)                                    # forward
            loss = F.cross_entropy(out, y)                    # loss
            opt.zero_grad()
            loss.backward()                                   # backward
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()                          # update
            tl += loss.item() * len(y); tc += (out.argmax(1) == y).sum().item(); n += len(y)
        secs = time.perf_counter() - t0
        vl, vacc, _, _ = evaluate(model, va)
        for k, v in zip(hist, (tl / n, tc / n, vl, vacc, secs)):
            hist[k].append(v)
        log(f"[{name}] ep {ep:02d}/{a.epochs} train {tl/n:.4f}/{tc/n:.4f}  "
            f"val {vl:.4f}/{vacc:.4f}  {secs:.1f}s")
    return hist


def benchmark(data, a, log, steps=50, repeats=3):
    """Timing on an otherwise idle machine, separate from training (training times can be
    disturbed by other programs). Median over `repeats` of: `steps` training steps and one
    full pass over the test set."""
    x_tr = torch.stack([data[0][i][0] for i in range(a.bs)]).to(DEVICE)
    y_tr = torch.stack([data[0][i][1] for i in range(a.bs)]).to(DEVICE)
    test = DataLoader(data[2], batch_size=1000)
    out = {}
    for tok in a.tokenizers:
        for msa, model in zip(MSA, build_pair(tok, a.seed)):
            model.to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
            def step():
                loss = F.cross_entropy(model(x_tr), y_tr)
                opt.zero_grad(); loss.backward(); opt.step()
            model.train()
            for _ in range(5): step()                      # warm-up
            t_step = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                for _ in range(steps): step()
                t_step.append((time.perf_counter() - t0) / steps)
            infer = [throughput(model, test) for _ in range(repeats)]
            name = f"{tok}/{msa}"
            out[name] = {"ms_per_step": round(1000 * float(np.median(t_step)), 1),
                         "est_s_per_epoch": round(float(np.median(t_step)) * math.ceil(55000 / a.bs), 1),
                         "infer_img_per_s": round(float(np.median(infer)))}
            log(f"bench {name}: {out[name]}")
    json.dump(out, open(OUT / "bench.json", "w"), indent=1)


# ---------------------------------------------------------------- plots
def plot_curves(res, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, tok in enumerate(TOKENIZERS):
        for msa, ls in (("scratch", "-"), ("torch", "--")):
            h = res[f"{tok}/{msa}"]["hist"]
            ep = range(1, len(h["val_acc"]) + 1)
            ax[0].plot(ep, h["train_loss"], f"C{i}", ls=ls, label=f"{tok} / {msa}")
            ax[1].plot(ep, h["val_acc"], f"C{i}", ls=ls, label=f"{tok} / {msa}")
    ax[0].set(title="E2: train loss", xlabel="epoch")
    ax[1].set(title="E2: validation accuracy", xlabel="epoch")
    for a in ax: a.legend(fontsize=7); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_confusion(name, preds, ys, path):
    cm = np.zeros((10, 10), int)
    np.add.at(cm, (ys.numpy(), preds.numpy()), 1)
    fig, a = plt.subplots(figsize=(6.5, 5.5))
    a.imshow(cm, cmap="Blues")
    a.set(title=f"{name}: confusion matrix (test)", xlabel="predicted", ylabel="true",
          xticks=range(10), yticks=range(10))
    a.set_xticklabels(CLASSES, rotation=60, ha="right", fontsize=7); a.set_yticklabels(CLASSES, fontsize=7)
    for i in range(10):
        for j in range(10):
            a.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                   color="w" if cm[i, j] > cm.max() / 2 else "k")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    off = cm.copy(); np.fill_diagonal(off, 0)
    pairs = sorted(((off[i, j], CLASSES[i], CLASSES[j]) for i in range(10) for j in range(10)),
                   reverse=True)[:3]
    return {"per_class_acc": dict(zip(CLASSES, (cm.diagonal() / cm.sum(1)).round(4).tolist())),
            "top_confusions": [f"{t} -> {q}: {c}" for c, t, q in pairs]}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--tokenizers", nargs="+", default=list(TOKENIZERS), choices=list(TOKENIZERS))
    ap.add_argument("--bench", action="store_true", help="only measure speed (writes bench.json)")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.bench:
        logf = open(OUT / "bench.log", "w", encoding="utf-8")
        def log(s):
            print(s, flush=True); logf.write(s + "\n"); logf.flush()
        log(f"device={DEVICE} torch={torch.__version__} threads={torch.get_num_threads()} bs={a.bs}")
        benchmark(load_data(a.seed), a, log)
        return
    logf = open(OUT / "train.log", "w", encoding="utf-8")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    log(f"device={DEVICE} torch={torch.__version__} threads={torch.get_num_threads()} "
        f"epochs={a.epochs} seed={a.seed} bs={a.bs} lr={a.lr} wd={a.wd}")
    check_msa(log)
    data = load_data(a.seed)
    test = DataLoader(data[2], batch_size=1000)
    res = {}
    for tok in a.tokenizers:
        for msa, model in zip(MSA, build_pair(tok, a.seed)):
            name = f"{tok}/{msa}"
            hist = train(name, model, data, a, log)
            tl, ta, p, y = evaluate(model, test)
            res[name] = {"tokenizer": tok, "msa": msa, "n_tokens": model.tok.n_tokens,
                         "params": n_params(model), "test_acc": ta, "test_loss": tl,
                         "best_val_acc": max(hist["val_acc"]),
                         "train_time_s": round(sum(hist["epoch_time_s"]), 1),
                         "epoch_time_s": round(float(np.mean(hist["epoch_time_s"])), 2),
                         "infer_img_per_s": round(throughput(model, test)), "hist": hist,
                         **plot_confusion(name, p, y, OUT / f"confusion_{tok}_{msa}.png")}
            torch.save(model.state_dict(), OUT / f"{tok}_{msa}.pt")
            r = res[name]
            log(f"{name}: {r['params']:,} params  {r['n_tokens']} tokens  test acc {ta:.4f}  "
                f"train {r['train_time_s']:.0f}s ({r['epoch_time_s']:.1f}s/epoch)  "
                f"infer {r['infer_img_per_s']} img/s")
            json.dump(res, open(OUT / "results.json", "w"), indent=1)
    if len(a.tokenizers) == len(TOKENIZERS):
        plot_curves(res, OUT / "curves.png")
    logf.close()


if __name__ == "__main__":
    main()
