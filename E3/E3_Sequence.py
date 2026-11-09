# Import libraries
import argparse, json, sys
from pathlib import Path
import torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT.parent / "E1"))
# Reuse E1 code for data loading, training, evaluation, plotting,..
import E1_Image_Classification as e1


REPS = {"row": (28, 28), "patch4": (49, 16)}   # ten: (T buoc thoi gian, D chieu moi buoc)

def to_seq(x, rep):
    # (B,1,28,28) -> (B,T,D). 
    # row: T = 28, D = 28; patch4: T = 49, D = 16
    x = x[:, 0]
    if rep == "row":
        return x
    return x.unfold(1, 4, 4).unfold(2, 4, 4).reshape(len(x), 49, 16)

class SeqNet(nn.Module):
    # LSTM/GRU -> Linear(10)
    def __init__(self, cell, rep, hidden = 128):
        super().__init__()
        self.rep = rep
        self.rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell](REPS[rep][1], hidden, batch_first = True)
        self.fc = nn.Linear(hidden, 10)

    def forward(self, x):
        out, _ = self.rnn(to_seq(x, self.rep))
        return self.fc(out[:, -1])

def register(cell, rep):
    name = f"{cell.upper()}-{rep}"
    e1.MODELS[name] = lambda: SeqNet(cell, rep)
    e1.learning_rate[name] = 2e-3
    return name

def load_baselines():
    # MLP/CNN checkpoint E1 (same seed, same split)
    r1 = json.load(open(ROOT.parent / "E1" / "outputs" / "results.json", encoding="utf-8"))
    models = {}
    for n in ("MLP", "CNN"):
        m = e1.MODELS[n]().to(e1.DEVICE)
        m.load_state_dict(torch.load(ROOT.parent / "E1" / "outputs" / f"{n}.pt", map_location = e1.DEVICE))
        models[n] = m
    return models, r1

def plot_params(res, path):
    fig, ax = plt.subplots(figsize = (7, 5))
    for i, n in enumerate(res):
        ax.plot(res[n]["params"], res[n]["test_acc"] * 100, "o", c = f"C{i}")
        ax.annotate(n, (res[n]["params"], res[n]["test_acc"] * 100), 
                    fontsize = 8, 
                    xytext = (4, 4),
                    textcoords = "offset points")
    ax.set(xscale = "log", xlabel = "parameters", ylabel = "Test accuracy (%)", title = "E3: Accuracy and Capacity")
    ax.grid(alpha = .3); plt.tight_layout(); plt.savefig(path, dpi = 150); plt.close()

def _check():
    # Check to_seq() for all reps
    x = torch.arange(2 * 784, dtype = torch.float).reshape(2, 1, 28, 28)
    for rep, (T, D) in REPS.items():
        s = to_seq(x, rep)
        assert s.shape == (2, T, D), (rep, s.shape)
        assert torch.equal(s.flatten().sort().values, x.flatten().sort().values), rep
    print("to_seq OK:", {r: tuple(to_seq(x, r).shape[1:]) for r in REPS})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type = int, default = 10)
    ap.add_argument("--seed", type = int, default = 0)
    ap.add_argument("--check", action = "store_true", help = "chi kiem tra to_seq roi thoat")
    a = ap.parse_args()
    if a.check:
        return _check()
    _check()

    OUT.mkdir(exist_ok=True)
    logf = open(OUT / "train.log", "w", encoding = "utf-8")
    def log(s):
        print(s); logf.write(s + "\n"); logf.flush()

    names = [register(c, r) for r in REPS for c in ("lstm", "gru")]
    log(f"device = {e1.DEVICE} epochs = {a.epochs} seed = {a.seed} models = {names}")
    loaders = e1.load_data(a.seed)
    res = {}
    for n in names:
        model, hist, secs = e1.train(n, a.seed, a.epochs, loaders, log)
        t1, ta, _, _ = e1.evaluate(model, loaders[2])
        torch.save(model.state_dict(), OUT / f"{n}.pt")
        res[n] = {"params": e1.n_params(model), 
                  "test_acc": ta, 
                  "test_loss": t1,
                  "best_val_acc": max(hist["val_acc"]), 
                  "train_time_s": round(secs, 1), 
                  "hist": hist,
                  **e1.plot_errors(n, model, loaders[2], OUT)}
        log(f"{n}: {res[n]['params']:,} params  test acc {ta:.4f}  {secs:.0f}s")

    # Evaluate baselines from E1
    base, r1 = load_baselines()
    for n, m in base.items():
        _, ta, _, _ = e1.evaluate(m, loaders[2])
        res[n] = {"params": r1[n]["params"], 
                  "test_acc": ta, 
                  "test_loss": r1[n]["test_loss"],
                  "train_time_s": r1[n]["train_time_s"], 
                  "hist": r1[n]["hist"], 
                  "source": "E1"}
        log(f"{n} (baseline E1): {res[n]['params']:,} params  test acc {ta:.4f}")

    e1.plot_curves({n: res[n]["hist"] for n in res}, OUT / "curves.png", title = "E3")
    plot_params(res, OUT / "acc_vs_params.png")
    json.dump(res, open(OUT / "results.json", "w"), indent = 1)
    logf.close()

if __name__ == "__main__":
    main()