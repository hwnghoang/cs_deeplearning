"""Food-101 subset: datasets and transforms (protocol A1.1, section 2)."""
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

A1 = Path(__file__).resolve().parents[1]
IMAGES = A1.parent / "data" / "food-101-256"     # pre-resized by scripts/prepare_data.py
SPLITS = A1 / "splits"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # ImageNet statistics


def class_names():
    return sorted({line.split("/")[0] for line in (SPLITS / "train.txt").read_text().split()})


TRAIN_TF = T.Compose([
    T.RandomResizedCrop(224, scale=(0.35, 1.0)),
    T.RandomHorizontalFlip(),
    T.TrivialAugmentWide(),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
    T.RandomErasing(p=0.25),
])
# Images are already resized to short side 256, so Resize(256) is a no-op kept for clarity.
EVAL_TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])


class Food101Subset(Dataset):
    def __init__(self, split, transform):
        self.items = (SPLITS / f"{split}.txt").read_text().split()
        self.class_to_idx = {c: i for i, c in enumerate(class_names())}
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        with Image.open(IMAGES / f"{item}.jpg") as im:
            x = self.transform(im.convert("RGB"))
        return x, self.class_to_idx[item.split("/")[0]]


def loaders(batch_size, workers, seed):
    train = Food101Subset("train", TRAIN_TF)
    g = torch.Generator().manual_seed(seed)
    mk = lambda ds, shuffle: DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None,
                                        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                                        drop_last=shuffle)
    return (mk(train, True),
            mk(Food101Subset("val", EVAL_TF), False),
            mk(Food101Subset("test", EVAL_TF), False))
