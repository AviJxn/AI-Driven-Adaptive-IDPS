# AI-Driven Adaptive IDPS
### Transformer-Based Temporal Anomaly Detection for Network Intrusion Prevention

[![Status](https://img.shields.io/badge/Status-Active_Research-brightgreen?style=plastic)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=plastic&logo=python)](https://www.python.org)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=plastic&logo=pytorch)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-CIC--IDS2017%20%7C%20UNSW--NB15-orange?style=plastic)]()

---

## Table of Contents

- [Overview](#overview)
- [Research Foundation](#research-foundation)
- [System Architecture](#system-architecture)
- [Mathematical Formulation](#mathematical-formulation)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Roadmap](#roadmap)
- [References](#references)

---

## Overview

Traditional network security systems treat traffic telemetry as **static, independent tabular vectors**, a paradigm that fundamentally fails against modern Advanced Persistent Threats (APTs), which unfold across prolonged, multi-stage attack sequences.

This project reframes intrusion detection as a **genuine temporal sequence modeling task**. Using Transformer encoder architectures with masked self-attention, the system maps conversation-level flow metrics over time and isolates structural deviations that betray attack patterns invisible to classical classifiers.

**Key contributions:**

- Leakage-free evaluation via strict 5-tuple group splits (Moczkodan & Ragab, 2026)
- Masked zero-padding pipelines that neutralize artificial attention cues from repeat-last padding
- Reconstruction Trend Enhancement (RTE) loops to prevent overfitting to amplitude anomalies under distribution drift
- Real-time threshold adaptation via Peak-Over-Threshold (POT) extreme value theory

---

## Research Foundation

Our architecture directly addresses three structural gaps identified in 2025–2026 state-of-the-art literature:

### 1 · Leakage-Free Sequence Evaluation

Conventional random train/test splits allow sliding windows from the **same TCP/UDP conversation** to appear on both sides of the split boundary — a form of data leakage that inflates reported F1 scores by 15–30% without improving real-world generalization.

This project enforces **Group-by-Five-Tuple splits** (src IP, dst IP, src port, dst port, protocol), ensuring the model is evaluated against traffic conversations it has never encountered.

> *Reference: Moczkodan & Ragab (2026), arXiv:2606.11098v1*

### 2 · Padding Vulnerability Mitigation

Sequence models that use *repeat-last* padding inject artificial periodic signals into attention layers. These become unintended classification shortcuts — the model learns to detect padding artifacts rather than genuine attack behavior.

This framework enforces **strict masked zero-padding** with injected attention mask vectors to isolate authentic temporal self-attention weights.

### 3 · Reconstruction Trend Enhancement

Drawing from RTdetector (Liu et al., IJCAI-25), our decoder incorporates **self-conditioning trend loops (RTE)** to stabilize non-stationary data drift in live traffic, preventing the encoder from overfitting to transient amplitude spikes that do not generalize across network environments.

---

## System Architecture

The system processes raw network captures through a deterministic multi-stage pipeline:

```
┌─────────────────────────────────────────┐
│       Raw Network Capture (PCAP)        │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│   CIC-FlowMeter Feature Extraction      │
│         (76 statistical columns)        │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  5-Tuple Conversational Grouping &      │
│       Chronological Sorting             │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│   Temporal Sliding Window Generator     │
│      (Fixed Window Matrix: T = 20)      │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Zero-Pad & Attention Mask Injection    │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│     CNN Pre-Encoder Feature Stack       │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Multi-Head Self-Attention Transformer  │
│         Encoders  (h=4, L=2)            │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│   Dual-Adversarial Decoder (RTE Loop)   │
│      with Focus-Score Weighting         │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Real-Time Extreme Value Thresholding   │
│          (POT Selector Head)            │
└─────────────────────────────────────────┘
```

---

## Mathematical Formulation

### Scaled Dot-Product Attention

The core sequence alignment mechanism uses multi-head scaled dot-product attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

where $Q \in \mathbb{R}^{T \times d_k}$, $K \in \mathbb{R}^{T \times d_k}$, $V \in \mathbb{R}^{T \times d_v}$ are the query, key, and value projections of the input sequence, and $\sqrt{d_k}$ is the temperature scaling factor preventing vanishing gradients in high-dimensional spaces.

### Reconstruction Trend Enhancement (RTE)

To stabilize non-stationary distribution shifts in live traffic, input streams are normalized per-sequence before decoding and de-normalized on output:

**Input normalization:**

$$X' = \frac{1}{\sigma_x} \odot (X - \mu_x)$$

**Decoder trend de-normalization:**

$$O = \sigma_x \odot (O' + \mu_x)$$

where $\mu_x$ and $\sigma_x$ are the per-sequence mean and standard deviation, and $\odot$ denotes element-wise multiplication. This ensures the model learns relative deviation structure rather than absolute amplitude — a critical property for generalization across heterogeneous network environments.

---

## Repository Structure

```
AI-Driven-Adaptive-IDPS/
│
├── .github/
│   └── workflows/
│       └── ci.yml               # Linting, unit tests, and coverage checks
│
├── docs/
│   ├── literature_survey.md     # Annotated survey of 2024–2026 IDPS literature
│   └── architecture_notes.md    # Design decisions and ablation rationale
│
├── datasets/
│   ├── clean_pipeline.py        # NaN filtering, standardization, standard scaling
│   └── windowing_head.py        # 5-tuple grouping and T=20 sliding window logic
│
├── src/
│   ├── layers/
│   │   ├── attention.py         # Trend-aware temporal attention with mask support
│   │   └── embedding.py         # Sinusoidal positional encoders
│   │
│   ├── models/
│   │   ├── transformer.py       # CNN pre-encoder + Transformer stack (primary)
│   │   ├── baselines.py         # LSTM, GRU, and 1D-CNN baseline comparisons
│   │   └── classifiers.py       # Static baselines: Random Forest, MLP
│   │
│   └── train.py                 # AdamW optimizer loop with early stopping & logging
│
├── README.md
└── requirements.txt
```

---

## Getting Started

### Prerequisites

```bash
python >= 3.10
torch >= 2.2.0
scikit-learn >= 1.4
numpy, pandas, matplotlib
```

### Installation

```bash
git clone https://github.com/<your-org>/AI-Driven-Adaptive-IDPS.git
cd AI-Driven-Adaptive-IDPS
pip install -r requirements.txt
```

### Data Preparation

Download [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) or [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and place CSVs under `datasets/raw/`. Then run:

```bash
python datasets/clean_pipeline.py --dataset CIC-IDS2017
python datasets/windowing_head.py --window 20 --split group
```

### Training

```bash
python src/train.py \
  --model transformer \
  --epochs 50 \
  --batch_size 64 \
  --lr 1e-4
```

---

## Roadmap

| Week | Milestone | Status |
|------|-----------|--------|
| 1 | Literature survey | Complete |
| 2 | Problem definition & research gap analysis | Complete |
| 3 | IDS Dataset Preprocessing and Traffic Analysis | Complete |
| 4 |  | Pending |
| 5 |  | Pending |
| 6 |  | Pending |

---

## References

```bibtex
@article{moczkodan2026transformers,
  title     = {Do Transformers Actually Help Intrusion Detection?
               A Temporal Sequence Evaluation on CIC-IDS2017},
  author    = {Moczkodan, Z. and Ragab, H.},
  journal   = {arXiv preprint arXiv:2606.11098v1},
  year      = {2026}
}

@inproceedings{liu2025rtdetector,
  title     = {RTdetector: Deep Transformer Networks for Time Series
               Anomaly Detection Based on Reconstruction Trend},
  author    = {Liu, X. and Li, X. and Li, Y. and Tang, F. and Zhao, M.},
  booktitle = {Proceedings of the 34th International Joint Conference
               on Artificial Intelligence (IJCAI-25)},
  year      = {2025}
}
```
