# Deep Learning Coursework (`cs_deeplearning`)

**Học sâu và ứng dụng trong thị giác máy tính (CO5085)** — HK261 (2026–2027), Trường Đại học Bách Khoa, Khoa Khoa học và Kỹ thuật Máy tính

|                        |                                                                       |
| ---------------------- | --------------------------------------------------------------------- |
| **Nhóm**        | Hummingbird                                                           |
| **Thành viên** | Nguyễn Lê Hoàng Hưng (2570214), Nguyễn Thị Linh Trang (2570344) |
| **Giảng viên** | Lê Thành Sách, Nguyễn Quốc Minh                                  |

## Bài tập

| Bài | Nội dung                               | Code                                                            | Báo cáo                                                                                              |
| ---- | --------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| E1   | Exercise 1                              | [E1/](https://github.com/hwnghoang/cs_deeplearning/tree/main/E1) | [exercises.pdf](https://github.com/hwnghoang/cs_deeplearning/blob/main/reports/exercises.pdf)           |
| E2   | Exercise 2                              | [E2/](https://github.com/hwnghoang/cs_deeplearning/tree/main/E2) | [exercises.pdf](https://github.com/hwnghoang/cs_deeplearning/blob/main/reports/exercises.pdf)           |
| E3   | Exercise 3                              | [E3/](https://github.com/hwnghoang/cs_deeplearning/tree/main/E3) | [exercises.pdf](https://github.com/hwnghoang/cs_deeplearning/blob/main/reports/exercises.pdf)           |
| A1   | CNN vs Transformer trên tập ảnh lớn | [A1/](https://github.com/hwnghoang/cs_deeplearning/tree/main/A1) | [A1.1 protocol (PDF)](https://github.com/hwnghoang/cs_deeplearning/blob/main/A1/docs/A1.1_protocol.pdf) |
| A2   | Phát hiện đối tượng: Faster R-CNN vs DETR | [A2/](https://github.com/hwnghoang/cs_deeplearning/tree/main/A2) | [A2.1 proposal (PDF)](https://github.com/hwnghoang/cs_deeplearning/blob/main/A2/docs/A2.1_proposal.pdf) |

---

## A1 — CNN vs Transformer trên tập ảnh lớn

So sánh một mô hình CNN và một mô hình Transformer trên tập ảnh Food-101, với các chế độ huấn luyện: from-scratch, pretrained + freeze toàn bộ backbone, pretrained + freeze một phần, và pretrained + full finetune.

|                       |                                                                                                                                                                                                           |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dataset**     | [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) — 101 lớp món ăn; subset 300 ảnh/lớp cho train, 50 ảnh/lớp cho validation, toàn bộ split test chính thức (25.250 ảnh) |
| **CNN**         | ResNet-50 (pretrained ImageNet-1k, `timm`)                                                                                                                                                              |
| **Transformer** | DeiT-Small / ViT-S/16 (pretrained ImageNet-1k, `timm`)                                                                                                                                                  |
| **Code**        | [A1/](https://github.com/hwnghoang/cs_deeplearning/tree/main/A1)                                                                                                                                           |

### Các mốc

| Mốc | Nội dung                         | Hạn       | Tài liệu                                                                                                                                                                                                 |
| ---- | --------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A1.1 | Dataset và protocol thí nghiệm | 26/10/2026 | [Protocol (PDF)](https://github.com/hwnghoang/cs_deeplearning/blob/main/A1/docs/A1.1_protocol.pdf) · [Protocol (Markdown)](https://github.com/hwnghoang/cs_deeplearning/blob/main/A1/docs/A1.1_protocol.md) |
| A1.2 | Kết quả so sánh cuối          | 09/11/2026 | Sắp cập nhật                                                                                                                                                                                            |

### Ma trận thí nghiệm

| Chế độ                                                  | ResNet-50 | DeiT-S |
| ---------------------------------------------------------- | --------- | ------ |
| From-scratch                                               | ✓        | ✓     |
| Pretrained + freeze toàn bộ backbone                     | ✓        | ✓     |
| Pretrained + freeze một phần (nửa đầu theo độ sâu) | ✓        | ✓     |
| Pretrained + full finetune                                 | ✓        | ✓     |

Nội dung sẽ cập nhật ở A1.2: EDA/dataset, DataLoader và augmentation, protocol thí nghiệm, kết quả so sánh, thảo luận.

---

## A2 — Phát hiện đối tượng: Faster R-CNN vs DETR

So sánh một detector CNN dùng anchor và NMS (Faster R-CNN) với một detector Transformer dự đoán tập hợp (DETR) trên Pascal VOC 2007, và phân tích paper DETR.

|                 |                                                                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Chủ đề**      | Phát hiện đối tượng (bounding box + lớp)                                                                                                  |
| **Paper**       | N. Carion et al., *End-to-End Object Detection with Transformers*, ECCV 2020 — [arXiv:2005.12872](https://arxiv.org/abs/2005.12872) · [code](https://github.com/facebookresearch/detr) |
| **Dataset**     | [Pascal VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) — 20 lớp; trainval 5.011 ảnh (chia 90/10 train/val), test 4.952 ảnh |
| **Metric**      | mAP@0.5 (chính), mAP@0.5:0.95, AP_S / AP_M / AP_L, FPS                                                                                     |
| **Mô hình**     | Faster R-CNN ResNet-50-FPN (`torchvision`) · DETR ResNet-50 (`detr-main`)                                                                 |
| **Code**        | [A2/](https://github.com/hwnghoang/cs_deeplearning/tree/main/A2)                                                                          |

### Các mốc

| Mốc  | Nội dung                         | Hạn        | Tài liệu |
| ---- | -------------------------------- | ---------- | -------- |
| A2.1 | Chủ đề, paper, dữ liệu, metric, kế hoạch | 02/11/2026 | [Proposal (PDF)](https://github.com/hwnghoang/cs_deeplearning/blob/main/A2/docs/A2.1_proposal.pdf) · [Proposal (Markdown)](https://github.com/hwnghoang/cs_deeplearning/blob/main/A2/docs/A2.1_proposal.md) |
| A2.2 | Pipeline, so sánh, phân tích paper | 16/11/2026 | Sắp cập nhật |

### Các biến thể

| #  | Mô hình | Khởi tạo |
| -- | ------- | -------- |
| V1 | Faster R-CNN R50-FPN | Checkpoint COCO |
| V2 | DETR R50 | Checkpoint COCO |
| V3 | DETR R50 | Chỉ backbone ImageNet |

---

## Cấu trúc thư mục

```text
cs_deeplearning/
+-- E1/                  # Exercise 1 PyTorch implementation
+-- E2/                  # Exercise 2 PyTorch implementation
+-- E3/                  # Exercise 3 PyTorch implementation
+-- A1/                  # Assignment 1: CNN vs Transformer
|   +-- docs/            # Protocol A1.1, báo cáo A1.2
+-- datasets/            # Datasets folder (ignored by Git)
+-- reports/             # PDF reports and documentation
|   +-- exercises.pdf    # Combined coursework report (E1-E3)
+-- .gitignore           # Git ignore rules (venv, kaggle.json, datasets)
+-- README.md            # Project documentation and instructions
+-- requirements.txt     # Python dependency configuration
```
