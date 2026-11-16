"""Evaluation-only experiments that test two claims of the DETR paper on VOC 2007 test
(no extra training; uses the best checkpoints written by src.train):

1. Paper Fig. 4: AP of the prediction made after each decoder layer (auxiliary heads), with and
   without NMS. Claim: NMS helps the first decoder layer and becomes useless or harmful later.
2. "DETR does not need NMS": the same question for Faster R-CNN, by switching its NMS off.

    cd A2
    python -m src.ablate                 # writes results/ablation.json
"""
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms

from .voc import VOCDetection, collate
from .models import build
from .evaluate import voc_ap, coco_ap
from .train import DEVICE, to_device

A2 = Path(__file__).resolve().parents[1]
NMS_IOU = 0.5          # Detectron2 default NMS threshold at test time, as used in the paper


def apply_nms(p, iou=NMS_IOU):
    keep = batched_nms(p["boxes"], p["scores"], p["labels"], iou)
    return {k: v[keep] for k, v in p.items()}


def metrics(preds, gts):
    v, c = voc_ap(preds, gts), coco_ap(preds, gts)
    return {"map50": v["map_area"], "AP": c["AP"], "APs": c["APs"], "APm": c["APm"], "APl": c["APl"]}


@torch.no_grad()
def detr_per_layer(run, loader):
    det = build("detr", "coco").to(DEVICE)          # init is irrelevant: the checkpoint overwrites all weights
    det.load_state_dict(torch.load(A2 / "checkpoints" / f"{run}.pt", map_location=DEVICE))
    det.eval()
    per_layer, gts = [[] for _ in range(6)], []
    for images, targets in loader:
        images, _ = to_device(images, targets)
        out = det.model(det._prepare(images))                               # one forward, 6 decoder outputs
        layers = out["aux_outputs"] + [{"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}]
        sizes = torch.stack([torch.tensor(im.shape[-2:], device=DEVICE) for im in images])
        for i, o in enumerate(layers):
            res = det.postprocess({k: o[k].float() for k in ("pred_logits", "pred_boxes")}, sizes)
            per_layer[i] += [{k: r[k].cpu() for k in ("boxes", "scores", "labels")} for r in res]
        gts += list(targets)
    rows = []
    for i, preds in enumerate(per_layer):
        rows.append({"layer": i + 1, "nms": False, **metrics(preds, gts)})
        rows.append({"layer": i + 1, "nms": True, **metrics([apply_nms(p) for p in preds], gts)})
        print(run, rows[-2], rows[-1], flush=True)
    return rows


@torch.no_grad()
def frcnn_without_nms(run, loader):
    det = build("frcnn", "coco").to(DEVICE)
    det.load_state_dict(torch.load(A2 / "checkpoints" / f"{run}.pt", map_location=DEVICE))
    det.eval()
    out = {}
    for name, thr in (("nms", det.model.roi_heads.nms_thresh), ("no_nms", 1.0)):
        det.model.roi_heads.nms_thresh = thr          # IoU 1.0: no box is ever suppressed
        preds, gts = [], []
        for images, targets in loader:
            images, _ = to_device(images, targets)
            preds += [{k: v.cpu() for k, v in o.items()} for o in det.predict(images)]
            gts += list(targets)
        out[name] = metrics(preds, gts)
        print(run, name, out[name], flush=True)
    return out


def main():
    loader = DataLoader(VOCDetection("test", False), batch_size=8, num_workers=6, collate_fn=collate)
    result = {"nms_iou": NMS_IOU}
    for run in ("V2_detr_coco", "V3_detr_imagenet"):
        if (A2 / "checkpoints" / f"{run}.pt").exists():
            result[f"{run}_per_layer"] = detr_per_layer(run, loader)
    if (A2 / "checkpoints" / "V1_frcnn_coco.pt").exists():
        result["V1_frcnn_coco_nms"] = frcnn_without_nms("V1_frcnn_coco", loader)
    (A2 / "results" / "ablation.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
