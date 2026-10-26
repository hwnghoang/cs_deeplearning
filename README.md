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
| A2   | Sắp cập nhật                         | —                                                              | —                                                                                                     |

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
