| Model | Params | Test accuracy | Balanced acc. | TB AUC (95% CI) | Sensitivity | Specificity | Heatmap hits TB box | CPU ms |
|---|---|---|---|---|---|---|---|---|
| CNN from scratch | 2.35M | 96.9% | 97.0% | 0.9952 (0.9921–0.9979) | 98.0% | 94.6% | 34% | 246 |
| DenseNet-121 (ImageNet start) **(deployed)** | 6.96M | 99.2% | 98.4% | 0.9996 (0.9991–0.9999) | 100.0% | 95.9% | 36% | 313 |
| DenseNet-121 (chest X-ray start) | 6.95M | 98.3% | 97.0% | 0.9965 (0.9936–0.9984) | 98.0% | 95.8% | 57% | 294 |
