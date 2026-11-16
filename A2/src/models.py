"""Faster R-CNN (torchvision) and DETR (official code in third_party/detr) behind one interface:

    det.loss(images, targets)  -> dict of weighted loss terms, key "loss" = their sum (training)
    det.predict(images)        -> list of dict(boxes xyxy in original pixels, scores, labels 0..19)
    det.param_groups()         -> optimizer parameter groups (backbone / rest)

`images` is a list of [3, H, W] float tensors in [0, 1] at the ORIGINAL size; each detector
resizes them with the same rule: short side random from TRAIN_SCALES (training) or
TEST_SCALE (evaluation), long side at most MAX_SIZE.
"""
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import ResNet50_Weights

A2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(A2 / "third_party" / "detr"))      # the DETR repo imports `models` and `util` top-level
from models.detr import build as build_detr_official      # noqa: E402
from util.misc import nested_tensor_from_tensor_list      # noqa: E402
from util import box_ops                                  # noqa: E402

NUM_CLASSES = 20
TRAIN_SCALES = tuple(range(384, 609, 32))                 # 384, 416, ..., 608 (paper: 480..800)
TEST_SCALE = 512
MAX_SIZE = 1000
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def count_params(model):
    return (sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.parameters()))


# ----------------------------------------------------------------------------- Faster R-CNN
class FasterRCNNDetector(nn.Module):
    """torchvision Faster R-CNN, ResNet-50-FPN. Its internal transform does the resizing and
    normalization; its labels are 1..20 (0 = background), shifted here to/from 0..19."""

    def __init__(self, init):
        super().__init__()
        if init == "coco":
            m = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1, box_score_thresh=0.001)
            in_features = m.roi_heads.box_predictor.cls_score.in_features
            m.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES + 1)   # new 21-way head
        elif init == "imagenet":
            m = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
                                        num_classes=NUM_CLASSES + 1, box_score_thresh=0.001)
        else:
            raise ValueError(init)
        m.transform.max_size = MAX_SIZE
        self.model = m

    def train(self, mode=True):
        super().train(mode)
        # GeneralizedRCNNTransform draws a random min_size in training and uses min_size[-1] in eval.
        self.model.transform.min_size = TRAIN_SCALES if mode else (TEST_SCALE,)
        return self

    def loss(self, images, targets):
        t = [{"boxes": x["boxes"], "labels": x["labels"] + 1} for x in targets]
        losses = self.model(images, t)
        losses["loss"] = sum(losses.values())
        return losses

    @torch.no_grad()
    def predict(self, images):
        return [{"boxes": o["boxes"], "scores": o["scores"], "labels": o["labels"] - 1} for o in self.model(images)]

    def param_groups(self, lr, lr_backbone, wd):
        bb = [p for n, p in self.model.named_parameters() if p.requires_grad and n.startswith("backbone.body.")]
        rest = [p for n, p in self.model.named_parameters() if p.requires_grad and not n.startswith("backbone.body.")]
        return [{"params": bb, "lr": lr_backbone, "weight_decay": wd}, {"params": rest, "lr": lr, "weight_decay": wd}]


# ----------------------------------------------------------------------------- DETR
def detr_args(aux_loss=True):
    """The defaults of detr-main/main.py (ResNet-50, 6+6 layers, width 256, 100 queries)."""
    return Namespace(dataset_file="voc", device="cuda", backbone="resnet50", dilation=False,
                     position_embedding="sine", hidden_dim=256, lr_backbone=1e-5, masks=False,
                     enc_layers=6, dec_layers=6, dim_feedforward=2048, dropout=0.1, nheads=8,
                     num_queries=100, pre_norm=False, aux_loss=aux_loss, set_cost_class=1, set_cost_bbox=5,
                     set_cost_giou=2, bbox_loss_coef=5, giou_loss_coef=2, eos_coef=0.1, frozen_weights=None)


def resize(img, short, max_size=MAX_SIZE):
    """Same rule as torchvision's detection transform: scale so the short side = `short`,
    unless the long side would exceed `max_size`."""
    h, w = img.shape[-2:]
    s = min(short / min(h, w), max_size / max(h, w))
    return F.interpolate(img[None], size=(round(h * s), round(w * s)), mode="bilinear",
                         align_corners=False, antialias=True)[0]


class DETRDetector(nn.Module):
    """Official DETR model + SetCriterion + PostProcess. With init="coco" the COCO checkpoint
    is loaded except the 92-way class head, which is re-initialized for 20 classes + no-object."""

    def __init__(self, init, aux_loss=True):
        super().__init__()
        args = detr_args(aux_loss)
        # build() creates the ResNet-50 backbone with torchvision's ImageNet weights
        self.model, self.criterion, post = build_detr_official(args)
        self.postprocess = post["bbox"]
        if init == "coco":
            ckpt = torch.load(A2 / "weights" / "detr-r50-e632da11.pth", map_location="cpu", weights_only=False)["model"]
            ckpt = {k: v for k, v in ckpt.items() if not k.startswith("class_embed.")}
            missing, unexpected = self.model.load_state_dict(ckpt, strict=False)
            assert set(missing) == {"class_embed.weight", "class_embed.bias"} and not unexpected, (missing, unexpected)
        elif init != "imagenet":
            raise ValueError(init)

    def _prepare(self, images):
        scales = TRAIN_SCALES if self.training else (TEST_SCALE,)
        mean, std = MEAN.to(images[0].device), STD.to(images[0].device)
        picks = torch.randint(len(scales), (len(images),)).tolist()
        return nested_tensor_from_tensor_list([(resize(im, scales[k]) - mean) / std for im, k in zip(images, picks)])

    def loss(self, images, targets):
        out = self.model(self._prepare(images))
        out = {k: (v.float() if torch.is_tensor(v) else [{a: b.float() for a, b in d.items()} for d in v])
               for k, v in out.items()}                                      # loss and matching in fp32
        t = []
        for img, x in zip(images, targets):
            h, w = img.shape[-2:]
            boxes = x["boxes"] / torch.tensor([w, h, w, h], device=img.device)
            t.append({"labels": x["labels"], "boxes": box_ops.box_xyxy_to_cxcywh(boxes)})  # normalized cx,cy,w,h
        terms = self.criterion(out, t)
        wd = self.criterion.weight_dict
        losses = {k: terms[k] * wd[k] for k in terms if k in wd}
        losses["loss"] = sum(losses.values())
        return losses

    @torch.no_grad()
    def predict(self, images, decoder_layer=-1):
        """decoder_layer=-1: final output; 0..4: auxiliary head after that decoder layer (needs aux_loss)."""
        out = self.model(self._prepare(images))
        if decoder_layer != -1:
            out = out["aux_outputs"][decoder_layer]
        sizes = torch.stack([torch.tensor(im.shape[-2:], device=im.device) for im in images])
        res = self.postprocess({k: out[k].float() for k in ("pred_logits", "pred_boxes")}, sizes)
        return [{"boxes": r["boxes"], "scores": r["scores"], "labels": r["labels"]} for r in res]

    def param_groups(self, lr, lr_backbone, wd):
        bb = [p for n, p in self.model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
        rest = [p for n, p in self.model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
        return [{"params": bb, "lr": lr_backbone, "weight_decay": wd}, {"params": rest, "lr": lr, "weight_decay": wd}]


def build(arch, init, aux_loss=True):
    if arch == "frcnn":
        return FasterRCNNDetector(init)
    if arch == "detr":
        return DETRDetector(init, aux_loss)
    raise ValueError(arch)
