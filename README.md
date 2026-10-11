# A Bellman-Filtered, Self-Modulating Convolutional Framework with Divine Religions Algorithm Tuning for Automated Crohn's Disease Grading in Wireless Capsule Endoscopy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

This repository contains the official, publication-grade implementation of the research paper:

> **"A Bellman-Filtered, Self-Modulating Convolutional Framework with Divine Religions Algorithm Tuning for Automated Crohn's Disease Grading in Wireless Capsule Endoscopy"**  
> **Authors:** P. Shanmuga Sundaram, V. Saravanan, N. Santhiyakumari, and S. Kumar Ganesh  
> *Department of Electronics and Communication Engineering, Knowledge Institute of Technology, Salem, Tamil Nadu, India.*  
> *Target Journal: Machine Learning (Springer Nature)*

---

## Abstract

Wireless capsule endoscopy (WCE) serves as a primary non-invasive modality for evaluating mucosal inflammation across the small bowel in suspected or established Crohn's disease. However, a single examination routinely generates over 50,000 video frames characterized by peristaltic blur, non-uniform illumination, and fluid occlusions, rendering manual review tedious and observer-dependent. We formulate frame-level Crohn's lesion grading as a joint state-estimation and feature-modulated classification task, introducing an integrated four-stage analytical framework:
1. **Stage I:** An edge-adaptive recursive Bellman filter operating on scanline state-space representations to decouple photometric restoration from edge degradation.
2. **Stage II:** A multi-scale 131-dimensional descriptor front-end integrating directional edge histograms (85-D), dominant color centroids (16-D), and Gabor texture energies (30-D) with localized vicinity self-attention.
3. **Stage III:** A self-modulating convolutional neural network (SMCNN) employing feature-wise affine gating (FiLM).
4. **Stage IV:** A Divine Religions Algorithm (DRA) metaheuristic for global modulation parameter optimization.

On a seven-class Crohn's severity grading corpus (aphthoid ulceration, edema, erythema, normal mucosa, stenosis, 3--10 mm ulceration, and >10 mm ulceration), the framework achieves **84.4% hold-out test accuracy**, **86.3% macro-precision**, **81.7% macro-recall**, **82.4% macro-F1**, and **0.983 macro-AUC**, corroborated by five-fold cross-validation at **83.0 ± 4.0%**. A one-way ANOVA across ablation configurations confirms statistically significant variance across stages ($F(3,16) = 38.51$, $p = 1.5 	imes 10^{-7}$).

---

## Repository Structure

```
BF-VVT-SMCNN-DRA/
├── configs/
│   └── default_config.yaml           # Algorithmic hyperparameters matching Table 3
├── src/
│   ├── filtering/
│   │   ├── bellman_filter.py         # Stage I: 1D/2D Edge-adaptive Bellman filter
│   │   └── baseline_filters.py       # Bilateral, Gaussian, and Median baselines
│   ├── descriptors/
│   │   ├── edge_histogram.py         # 85-D Directional Edge Histogram Descriptor (EHD)
│   │   ├── dominant_color.py         # 16-D Dominant Color Descriptor via GLA (CIE-Lab)
│   │   ├── gabor_texture.py          # 30-D Multi-Scale Gabor Texture Filter Bank
│   │   ├── vicinity_attention.py     # Localized Vicinity Vision Transformer Attention
│   │   └── feature_pipeline.py       # Combined 131-D descriptor extractor
│   ├── models/
│   │   ├── smcnn.py                  # Stage III: Self-Modulating CNN with FiLM gating
│   │   └── baseline_cnn.py           # Unmodulated baseline CNN
│   ├── optimization/
│   │   └── dra_optimizer.py          # Stage IV: Divine Religions Algorithm metaheuristic
│   ├── dataset/
│   │   ├── crohn_dataset.py          # CrohnIPI dataset loader (7 classes, 5 folds)
│   │   └── augmentations.py          # Endoscopic augmentation protocol
│   └── utils/
│       ├── metrics.py                # PSNR, SSIM, Accuracy, F1, ROC-AUC, MCC, Cohen's kappa
│       ├── statistical_tests.py      # One-way ANOVA and Welch's t-test
│       └── visualization.py          # Confusion matrix, ROC curves, training plots
├── experiments/
│   ├── run_denoising_benchmark.py    # Reproduces Table 1 & Figure 2
│   ├── train_smcnn.py                # Reproduces Tables 4, 5 & Figures 7, 8, 9, 10
│   ├── run_ablation_study.py         # Reproduces Table 6 & Figure 12 (5-fold CV)
│   └── run_statistical_analysis.py   # Reproduces Tables 7 & 8 (ANOVA & Welch t-tests)
├── demo.py                           # Standalone 1-click end-to-end demo
├── requirements.txt                   # Dependency specifications
├── LICENSE                            # MIT License
└── README.md                          # Repository documentation
```

---

## Installation

### Prerequisites
- Python 3.10 or higher
- PyTorch 2.0+ (CUDA GPU support recommended, CPU supported)

### Setup
```bash
git clone https://github.com/your-username/BF-VVT-SMCNN-DRA-Crohns-Grading.git
cd BF-VVT-SMCNN-DRA-Crohns-Grading

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

---

## Quickstart Demo

To test the entire pipeline on a sample image in a single command:

```bash
python demo.py
```

Expected output:
```
======================================================================
QUICKSTART DEMO: BF-VVT-SMCNN-DRA FOR CROHN'S LESION GRADING
======================================================================

[Stage I] Running Edge-Adaptive Bellman Filter...
  -> Stage I completed in 12.4 ms

[Stage II] Extracting 131-D Multi-Scale Descriptors...
  -> Extracted descriptor of shape: (131,)
  -> Stage II completed in 4.1 ms

[Stage III] Executing SMCNN Forward Inference...
  -> Stage III completed in 1.9 ms

======================================================================
DIAGNOSTIC PREDICTION RESULTS:
  Predicted Category: Normal (Class ID: 0)
  Confidence:         88.42%
  End-to-End Latency: 18.4 ms (>50 FPS throughput)
======================================================================
Demo completed successfully!
```

---

## Reproducing Paper Experiments

### 1. Denoising Benchmark (Table 1 & Figure 2)
```bash
python experiments/run_denoising_benchmark.py
```

### 2. Multi-Class Hold-Out Evaluation (Tables 4 & 5)
```bash
python experiments/train_smcnn.py
```

### 3. Five-Fold Cross-Validation Ablation (Table 6 & Figure 12)
```bash
python experiments/run_ablation_study.py
```

### 4. Statistical Hypothesis Testing (Tables 7 & 8)
```bash
python experiments/run_statistical_analysis.py
```

---

## Experimental Results Summary

| Configuration | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |
|---|:---:|:---:|:---:|:---:|
| (d) Colour Cue Only | $59.9 \pm 3.8\%$ | $55.6 \pm 3.3\%$ | $53.9 \pm 4.0\%$ | $53.8 \pm 3.7\%$ |
| (c) w/o Bellman Filter | $76.8 \pm 1.5\%$ | $77.8 \pm 1.5\%$ | $74.7 \pm 1.4\%$ | $75.5 \pm 1.1\%$ |
| (b) w/o DRA Optimization | $83.6 \pm 4.3\%$ | $84.4 \pm 4.7\%$ | $82.4 \pm 4.5\%$ | $82.9 \pm 4.4\%$ |
| **(a) Full Proposed Pipeline** | **$83.0 \pm 4.0\%$** | **$84.8 \pm 3.6\%$** | **$81.7 \pm 3.9\%$** | **$82.4 \pm 3.6\%$** |

- **One-way ANOVA:** $F(3,16) = 38.51$, $p = 1.5 	imes 10^{-7}$ (statistically significant).
- **Hold-out Test Accuracy:** 84.4%, Macro-AUC: 0.983.
- **Inference Latency:** 18.4 ms per frame (>54 FPS).

---

## Citation

If you use this codebase or find our work helpful in your research, please cite:


```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
