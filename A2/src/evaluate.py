"""Detection metrics.

voc_ap   Pascal VOC protocol, IoU >= 0.5, 'difficult' objects ignored (neither TP nor FP).
         Both the VOC2007 11-point interpolation and the area under the PR curve are returned.
coco_ap  COCO protocol via pycocotools: AP@[.5:.95], AP50, AP75, AP_S/M/L. 'difficult'
         objects are passed as crowd regions, so detections on them are ignored as well.
"""
import contextlib, io
import numpy as np
import torch
from torchvision.ops import box_iou


def voc_ap(preds, gts, num_classes=20, iou_thr=0.5):
    """preds/gts: lists (one entry per image) of dicts with boxes [N,4] xyxy, labels, (scores | difficult).
    Returns dict(ap11 [C], ap_area [C], map11, map_area, pr={c: (recall, precision)})."""
    ap11, ap_area, pr = np.zeros(num_classes), np.zeros(num_classes), {}
    for c in range(num_classes):
        scores, tp, n_pos = [], [], 0
        for p, g in zip(preds, gts):
            gmask = g["labels"] == c
            gb, gdiff = g["boxes"][gmask], g["difficult"][gmask]
            n_pos += int((~gdiff).sum())
            pmask = p["labels"] == c
            pb, ps = p["boxes"][pmask], p["scores"][pmask]
            order = ps.argsort(descending=True)
            pb, ps = pb[order], ps[order]
            used = torch.zeros(len(gb), dtype=torch.bool)
            ious = box_iou(pb, gb) if len(gb) and len(pb) else torch.zeros(len(pb), len(gb))
            for k in range(len(pb)):
                hit = False
                if len(gb) > 0:
                    best_iou, j = ious[k].max(0)
                    if best_iou >= iou_thr and gdiff[j]:
                        continue                  # matched a difficult object: ignored
                    if best_iou >= iou_thr and not used[j]:
                        used[j] = True
                        hit = True
                # not a hit = IoU too low, or a second detection of an object already found
                scores.append(ps[k].item())
                tp.append(int(hit))
        if n_pos == 0:
            continue
        order = np.argsort(-np.array(scores))
        tp = np.array(tp)[order]
        tpc, fpc = np.cumsum(tp), np.cumsum(1 - tp)
        recall = tpc / n_pos
        precision = tpc / np.maximum(tpc + fpc, 1e-9)
        pr[c] = (recall, precision)
        # VOC2007: mean of the max precision at recall >= t, t in {0, 0.1, ..., 1}
        ap11[c] = np.mean([precision[recall >= t].max() if (recall >= t).any() else 0 for t in np.linspace(0, 1, 11)])
        # area under the monotone (upper envelope) PR curve, as in VOC2010+
        mrec = np.concatenate([[0], recall, [1]])
        mpre = np.concatenate([[0], precision, [0]])
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap_area[c] = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return {"ap11": ap11, "ap_area": ap_area, "map11": float(ap11.mean()), "map_area": float(ap_area.mean()), "pr": pr}


def coco_ap(preds, gts, num_classes=20):
    """COCO-style metrics. Returns dict(AP, AP50, AP75, APs, APm, APl)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    images, anns, dets = [], [], []
    for i, (p, g) in enumerate(zip(preds, gts)):
        h, w = (int(x) for x in g["orig_size"])
        images.append({"id": i, "height": h, "width": w})
        for b, l, d in zip(g["boxes"].tolist(), g["labels"].tolist(), g["difficult"].tolist()):
            bw, bh = b[2] - b[0], b[3] - b[1]
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": l, "bbox": [b[0], b[1], bw, bh],
                         "area": bw * bh, "iscrowd": int(d)})
        for b, l, s in zip(p["boxes"].tolist(), p["labels"].tolist(), p["scores"].tolist()):
            dets.append({"image_id": i, "category_id": l, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": s})
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": c} for c in range(num_classes)]}
        gt.createIndex()
        dt = gt.loadRes(dets) if dets else COCO()
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return dict(zip(("AP", "AP50", "AP75", "APs", "APm", "APl"), (float(x) for x in ev.stats[:6])))
