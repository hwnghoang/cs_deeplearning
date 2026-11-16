# A2 — Object detection: Faster R-CNN vs DETR on Pascal VOC 2007

Faster R-CNN (ResNet-50-FPN, torchvision) and DETR (ResNet-50, official code) trained on VOC 2007 with the same data, resizing rules and evaluation, plus an analysis of the DETR paper (Carion et al., ECCV 2020). Plan: [docs/A2.1_proposal.md](docs/A2.1_proposal.md). Report: [docs/A2.2_report.pdf](docs/A2.2_report.pdf).

## Layout

| Path | Content |
|---|---|
| `src/voc.py` | VOC XML parsing, splits (`splits/*.txt`), Dataset (horizontal flip) |
| `src/models.py` | Both detectors behind one interface (`loss`, `predict`, `param_groups`), shared resize rule |
| `src/train.py` | Hand-written training loop, validation mAP per epoch, test evaluation, FPS |
| `src/evaluate.py` | VOC mAP@0.5 (difficult ignored; 11-point and area) and COCO metrics via pycocotools |
| `src/ablate.py` | Evaluation-only checks of two paper claims: AP per decoder layer with/without NMS, Faster R-CNN without NMS |
| `src/analyze.py` | EDA, tables and figures |
| `run_all.py` | The four training runs (V1, V2, V3, V5) |
| `third_party/detr/` | `models/` and `util/` of [facebookresearch/detr](https://github.com/facebookresearch/detr), unmodified (Apache-2.0, see its LICENSE) |
| `results/` | Per-run JSON (history, metrics, costs), logs, test predictions, figures |

## Variants

| Run | Detector | Initialization | Epochs (lr ÷10 after) |
|---|---|---|---|
| V1 | Faster R-CNN R50-FPN | COCO checkpoint (torchvision), new 21-class head | 10 (7) |
| V2 | DETR R50 | official COCO checkpoint, new 21-class head | 20 (15) |
| V3 | DETR R50 | ImageNet backbone only | 20 (15) |
| V5 | Faster R-CNN R50-FPN | ImageNet backbone only | 10 (7) |

## Environment

Windows 11, NVIDIA GeForce GTX 1070 (8 GB), Python 3.12, `torch==2.8.0+cu126`, `torchvision==0.23.0+cu126`, `scipy`, `pycocotools`, `matplotlib`. Training runs in fp32 (fp16 autocast was measured slower on this Pascal GPU).

## Run

```bash
# data (from the repository root): VOC 2007 trainval + test into data/VOCdevkit
cd data
curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar -xf VOCtrainval_06-Nov-2007.tar && tar -xf VOCtest_06-Nov-2007.tar
cd ..

# official DETR COCO checkpoint
mkdir -p A2/weights
curl -L -o A2/weights/detr-r50-e632da11.pth https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth

cd A2
python -c "from src.voc import make_splits; make_splits()"   # splits/*.txt (already committed)
python run_all.py                                             # 4 runs, about 11 hours on a GTX 1070
python -m src.ablate                                          # decoder-layer / NMS checks (about 20 min)
python -m src.analyze                                         # tables and figures

# a single run
python -m src.train --run V2_detr_coco --arch detr --init coco --epochs 20 --lr_drop 15
```

The torchvision COCO checkpoint of Faster R-CNN and the ImageNet ResNet-50 weights are downloaded automatically on first use. Checkpoints written by training go to `checkpoints/` (not committed).
