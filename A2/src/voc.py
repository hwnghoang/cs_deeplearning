"""Pascal VOC 2007: annotation parsing, splits and the Dataset used by both detectors.

Boxes are returned as absolute (x1, y1, x2, y2) in the original image, labels as 0..19.
Resizing is NOT done here: each detector resizes with the same scale rules
(models.TRAIN_SCALES / TEST_SCALE), so both see the same augmentation.
"""
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

A2 = Path(__file__).resolve().parents[1]
VOC = A2.parent / "data" / "VOCdevkit" / "VOC2007"
SPLITS = A2 / "splits"
CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
           "train", "tvmonitor")


def parse_annotation(image_id):
    """-> dict(boxes [N,4] float, labels [N] int64, difficult [N] bool, size (h, w))."""
    root = ET.parse(VOC / "Annotations" / f"{image_id}.xml").getroot()
    size = root.find("size")
    h, w = int(size.find("height").text), int(size.find("width").text)
    boxes, labels, difficult = [], [], []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        # VOC pixel coordinates are 1-based and inclusive; shift to 0-based.
        x1, y1, x2, y2 = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append([x1 - 1, y1 - 1, x2, y2])
        labels.append(CLASSES.index(obj.find("name").text.strip()))
        d = obj.find("difficult")
        difficult.append(d is not None and d.text.strip() == "1")
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "difficult": torch.tensor(difficult, dtype=torch.bool), "size": (h, w)}


def make_splits(val_fraction=0.1, seed=42):
    """train/val from the official trainval list (shuffled with `seed`), test = official test list."""
    trainval = (VOC / "ImageSets" / "Main" / "trainval.txt").read_text().split()
    test = (VOC / "ImageSets" / "Main" / "test.txt").read_text().split()
    ids = sorted(trainval)
    random.Random(seed).shuffle(ids)
    n_val = round(len(ids) * val_fraction)
    splits = {"train": sorted(ids[n_val:]), "val": sorted(ids[:n_val]), "test": sorted(test)}
    SPLITS.mkdir(exist_ok=True)
    for name, items in splits.items():
        (SPLITS / f"{name}.txt").write_text("\n".join(items) + "\n")
    return splits


def read_split(name):
    return (SPLITS / f"{name}.txt").read_text().split()


class VOCDetection(Dataset):
    """Image as float tensor in [0, 1], shape [3, H, W], original size.
    Training applies a random horizontal flip (the only augmentation done on the CPU)."""

    def __init__(self, split, train):
        self.ids = read_split(split)
        self.train = train
        self.ann = {i: parse_annotation(i) for i in self.ids}      # ~10k small XML files, parsed once

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img = read_image(str(VOC / "JPEGImages" / f"{image_id}.jpg"), ImageReadMode.RGB).float() / 255
        a = self.ann[image_id]
        boxes = a["boxes"].clone()
        if self.train and random.random() < 0.5:
            img = img.flip(-1)
            w = img.shape[-1]
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        target = {"boxes": boxes, "labels": a["labels"], "difficult": a["difficult"],
                  "image_id": idx, "orig_size": torch.tensor(img.shape[-2:])}
        return img, target


def collate(batch):
    return tuple(zip(*batch))
