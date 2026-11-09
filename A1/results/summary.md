| Model | Mode | Trainable / total params | Epochs | Best val top-1 | Test top-1 | Test top-5 | Test macro-F1 | s/epoch (median) | Train time | Peak GPU mem | Inference img/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | From-scratch | 23.71M / 23.71M | 10 | 16.34 | 17.80 | 40.77 | 14.81 | 75 | 13.5 min | 6.5 GB | 491 |
| ResNet-50 | Freeze backbone | 0.21M / 23.71M | 5 | 48.61 | 52.85 | 80.91 | 51.00 | 24 | 2.4 min | 2.7 GB | 479 |
| ResNet-50 | Freeze half | 22.27M / 23.71M | 5 | 57.23 | 60.02 | 85.94 | 58.94 | 46 | 4.5 min | 5.7 GB | 453 |
| ResNet-50 | Full finetune | 23.71M / 23.71M | 5 | 57.13 | 60.44 | 85.82 | 59.37 | 78 | 7.5 min | 5.3 GB | 448 |
| DeiT-S | From-scratch | 21.70M / 21.70M | 10 | 9.95 | 11.01 | 30.32 | 7.99 | 89 | 15.3 min | 2.4 GB | 437 |
| DeiT-S | Freeze backbone | 0.04M / 21.70M | 5 | 57.77 | 60.69 | 86.46 | 59.87 | 27 | 2.6 min | 0.3 GB | 434 |
| DeiT-S | Freeze half | 10.69M / 21.70M | 5 | 69.70 | 75.39 | 93.72 | 75.10 | 60 | 5.5 min | 1.3 GB | 412 |
| DeiT-S | Full finetune | 21.70M / 21.70M | 5 | 72.08 | 76.20 | 93.82 | 75.97 | 94 | 8.3 min | 2.4 GB | 414 |
