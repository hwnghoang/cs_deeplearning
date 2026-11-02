# E2 — Transformer: MSA and tokenization (Fashion-MNIST)

Small Transformer classifier on Fashion-MNIST (same data and split as E1) with:

- **Two MSA implementations**
  - `ScratchMSA`: written from `nn.Linear`, `torch.einsum` and `softmax` (no `nn.MultiheadAttention`).
  - `TorchMSA`: `nn.MultiheadAttention`, used as the reference.
- **Three tokenizations**
  - `patch`: 4×4 non-overlapping patches → 49 tokens.
  - `row`: each image row is one token → 28 tokens.
  - `cnn_stem`: the two conv blocks of the E1 CNN → 7×7 feature map → 49 tokens.

All 6 combinations (3 tokenizers × 2 MSA) are trained with the same hand-written training loop.

## Environment

- Python 3.12, PyTorch 2.8, torchvision 0.23, numpy, matplotlib (see `../requirements.txt`).
- Results in `outputs/` were produced on CPU (16 threads).

## Run

```bash
cd E2
python E2_Transformer.py                      # all 6 runs, 10 epochs, seed 0
python E2_Transformer.py --tokenizers patch   # only one tokenizer
python E2_Transformer.py --bench              # speed only, on an idle machine -> outputs/bench.json
```

Speed numbers in the report come from `--bench`, run after training with nothing else running, because epoch times logged during training can be disturbed by other programs.

Options: `--epochs`, `--seed`, `--bs`, `--lr`, `--wd`, `--tokenizers`.

At start-up the script checks that `ScratchMSA` gives the same output as `nn.MultiheadAttention` when both use the same weights, and stops if they differ.

## Outputs (`outputs/`)

| File | Content |
|---|---|
| `train.log` | Per-epoch train/val loss and accuracy, epoch time, final summary |
| `results.json` | Params, test accuracy, timing, throughput, per-class accuracy, history for every run |
| `bench.json`, `bench.log` | ms per training step, estimated s/epoch, inference images/s (median of 3) |
| `curves.png` | Train loss and validation accuracy of all 6 runs |
| `confusion_<tokenizer>_<msa>.png` | Test confusion matrix of each run |
| `<tokenizer>_<msa>.pt` | Final weights of each run |
