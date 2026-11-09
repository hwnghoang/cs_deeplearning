# A1 — CNN vs Transformer on Food-101

ResNet-50 (CNN) and DeiT-Small (Transformer) on a controlled subset of Food-101, each in four training modes: from scratch, pretrained with the backbone frozen (linear probe), pretrained with the first half of the backbone frozen, and full fine-tuning. Protocol: [docs/A1.1_protocol.md](docs/A1.1_protocol.md).

## Layout

| Path | Content |
|---|---|
| `scripts/prepare_data.py` | Builds the stratified splits (`splits/*.txt`) and pre-resizes the selected images to short side 256 px |
| `src/data.py` | Dataset, train/eval transforms, DataLoaders |
| `src/models.py` | `timm` models, freezing modes, parameter groups |
| `src/train.py` | One run: hand-written training loop, evaluation, cost measurements |
| `run_all.py` | The 8 main runs in sequence |
| `src/analyze.py` | Tables and figures for the report |
| `results/` | Per-run JSON (history, metrics, costs), logs, test predictions, figures |

## Environment

- Windows 11, NVIDIA GeForce GTX 1070 (8 GB), Python 3.12.
- `torch==2.8.0+cu126` (CUDA 12.6 is the last build line that supports Pascal GPUs), `timm`, `scikit-learn`, `matplotlib`.
- `torchvision==0.23.0+cpu`: the CUDA build of torchvision's native extension was blocked by Windows Smart App Control on this machine. torchvision is only used for image transforms on the CPU, so the CPU build is sufficient; all model computation runs on the GPU through `torch`.

## Run

```bash
# 1. data: download and extract Food-101 into ../data/food-101 (about 5 GB)
curl -L -o ../data/food-101.tar.gz https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -xzf ../data/food-101.tar.gz -C ../data

# 2. splits + resized images (from the A1/ folder)
python scripts/prepare_data.py

# 3. all 8 runs, then tables and figures
python run_all.py
python -m src.analyze

# a single run
python -m src.train --arch deit_s --mode partial --epochs 5
```

## Changes from the A1.1 protocol

The A1.2 deadline left about two hours of GPU time, while the protocol needed an estimated 12–15 hours. The experiment design (2 models × 4 modes, same split, augmentation and hyperparameters for every run) is unchanged; only its size was reduced:

| | Protocol A1.1 | A1.2 (used) |
|---|---|---|
| Train images per class | 300 | 100 (10,100 images) |
| Validation images per class | 50 | 20 (2,020 images) |
| Test images per class | 250 (full official test split) | 50 (5,050 images, stratified from the official test split) |
| Epochs: from scratch / pretrained modes | 50 / 15 | 10 / 5 |
| Batch size | 64 × 2 accumulation = 128 | 64, no accumulation |
| Warm-up | 5 epochs (scratch), 1 epoch (others) | first 10% of steps for every run |

Everything else (models, checkpoints, freezing, learning rates, weight decay, label smoothing, augmentation, AMP, best-validation checkpoint selection, seed 42) follows the protocol.
