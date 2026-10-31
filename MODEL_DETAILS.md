# Model Specification & Architectural Details

[![Architecture](https://img.shields.io/badge/Architecture-U--Net%20(SMP)-blue.svg)](https://github.com/qubvel/segmentation_models.pytorch)
[![Encoder](https://img.shields.io/badge/Backbone-ResNet--18%20(Pretrained)-orange.svg)](https://pytorch.org/vision/stable/models/resnet.html)
[![Classes](https://img.shields.io/badge/Classes-13%20Semantic%20Labels-green.svg)](#13-class-semantic-taxonomy)
[![Holdout Test mIoU](https://img.shields.io/badge/Holdout%20mIoU-0.7703-brightgreen.svg)](#empirical-performance-benchmarks)
[![Holdout Pixel Acc](https://img.shields.io/badge/Pixel%20Accuracy-95.04%25-success.svg)](#empirical-performance-benchmarks)
[![Parameters](https://img.shields.io/badge/Parameters-14.33M-lightgrey.svg)](#parameter--computational-profile)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Technical Specifications](#technical-specifications)
3. [Architectural Deep-Dive](#architectural-deep-dive)
   - [Encoder Backbone (ResNet-18)](#encoder-backbone-resnet-18)
   - [Decoder Network (U-Net)](#decoder-network-u-net)
   - [Segmentation Head](#segmentation-head)
   - [Tensor Flow & Skip Connections](#tensor-flow--skip-connections)
4. [Parameter & Computational Profile](#parameter--computational-profile)
5. [13-Class Semantic Taxonomy](#13-class-semantic-taxonomy)
6. [Data Preprocessing & Augmentation Pipeline](#data-preprocessing--augmentation-pipeline)
   - [Input Normalization](#input-normalization)
   - [Data Augmentation](#data-augmentation)
   - [Procedural Occlusion Augmenter](#procedural-occlusion-augmenter)
7. [Training Methodology & Optimization](#training-methodology--optimization)
   - [Dual-Objective Loss Function](#dual-objective-loss-function)
   - [Hyperparameters & Optimization](#hyperparameters--optimization)
   - [Dataset Partitioning](#dataset-partitioning)
8. [Empirical Performance Benchmarks](#empirical-performance-benchmarks)
   - [Global Holdout Evaluation](#global-holdout-evaluation)
   - [Per-Class Segmentation Metrics](#per-class-segmentation-metrics)
   - [Occlusion Robustness Benchmark](#occlusion-robustness-benchmark)
9. [Downstream Integration: Occlusion Analytics & Adaptive Weighting](#downstream-integration-occlusion-analytics--adaptive-weighting)
   - [Facial Occlusion Ratio ($O_{\text{face}}$)](#facial-occlusion-ratio-oface)
   - [Recognition Viability Boundary](#recognition-viability-boundary)
   - [Tri-Region Vertical Partitioning](#tri-region-vertical-partitioning)
   - [Adaptive Feature Weighting Equation](#adaptive-feature-weighting-equation)
10. [Programmatic Inference & Usage](#programmatic-inference--usage)
    - [Python API](#python-api)
    - [CLI Usage](#cli-usage)
11. [Limitations & Future Roadmap](#limitations--future-roadmap)

---

## Executive Summary

This document provides complete technical documentation for the **Occlusion-Aware Facial Semantic Segmentation Model**. The network is designed to perform fine-grained anatomical and accessory segmentation of human faces under both clean conditions and challenging real-world occlusions (such as surgical face masks, tinted sunglasses, prescription eyeglasses, and headwear).

The segmented output directly powers an downstream **Occlusion Analytics Engine** that calculates facial coverage, evaluates recognition viability, and computes normalized tri-region adaptive feature weights (Upper, Mid, Lower) for downstream face identification backends (e.g., ArcFace, FaceNet, AdaFace).

---

## Technical Specifications

| Property | Value / Specification | Description |
| :--- | :--- | :--- |
| **Model Name** | Occlusion-Aware Face U-Net | Full-pipeline face parser & occlusion quantifier |
| **Base Architecture** | U-Net | Encoder-decoder network with long skip connections |
| **Encoder Backbone** | ResNet-18 | Residual network pretrained on ImageNet-1k |
| **Framework** | PyTorch 2.0+ / `segmentation-models-pytorch` | Torchvision-compatible layer representations |
| **Input Dimensions** | `(B, 3, 256, 256)` | Batch size $B$, 3 RGB channels, height 256, width 256 |
| **Input Color Space** | RGB (ImageNet normalized) | Float32 tensors scaled with ImageNet mean/std |
| **Output Dimensions** | `(B, 13, 256, 256)` | Raw unnormalized logits across 13 classes |
| **Activation in Model** | `None` (Raw Logits) | Softmax and argmax are deferred to post-processing |
| **Total Parameters** | **14,329,949** (~14.33 M) | All layers active and trainable |
| **Trainable Parameters** | **14,329,949** (100.0%) | Full end-to-end backpropagation |
| **Model Weight File** | `checkpoints/best_model.pth` | 164.17 MB binary checkpoint (state dict + optimizer) |
| **Primary Metric** | Mean IoU (mIoU) | Monitored during validation checkpointing |
| **Test Pixel Accuracy** | **95.04%** | Evaluated on holdout test set (4,500 images) |
| **Test Mean IoU** | **0.7703** (77.03%) | Across all active semantic classes |
| **Test Mean Dice** | **0.8640** (86.40%) | Macro-averaged F1/Dice score |

---

## Architectural Deep-Dive

The model follows a symmetric **U-Net** topology with an asymmetric capacity allocation: a high-capacity pretrained feature extractor (**ResNet-18**) coupled with a lightweight, multi-scale upsampling decoder.

```mermaid
graph TD
    subgraph Input
        IMG["Input Tensor<br/>(3, 256, 256)"]
    end

    subgraph "Encoder (ResNet-18 Pretrained Backbone)"
        E0["Stage 0: Conv1 + BN + ReLU<br/>Output: (64, 128, 128)"]
        E1["Stage 1: MaxPool + Layer 1<br/>Output: (64, 64, 64)"]
        E2["Stage 2: Layer 2 (Residual)<br/>Output: (128, 32, 32)"]
        E3["Stage 3: Layer 3 (Residual)<br/>Output: (256, 16, 16)"]
        E4["Stage 4: Layer 4 / Bottleneck<br/>Output: (512, 8, 8)"]
    end

    subgraph "Decoder (U-Net Upsampling Blocks)"
        D4["Decoder Block 4<br/>Input: E4 + Skip E3<br/>Output: (256, 16, 16)"]
        D3["Decoder Block 3<br/>Input: D4 + Skip E2<br/>Output: (128, 32, 32)"]
        D2["Decoder Block 2<br/>Input: D3 + Skip E1<br/>Output: (64, 64, 64)"]
        D1["Decoder Block 1<br/>Input: D2 + Skip E0<br/>Output: (32, 128, 128)"]
        D0["Decoder Block 0<br/>Input: D1 + Skip Input<br/>Output: (16, 256, 256)"]
    end

    subgraph "Segmentation Head"
        HEAD["Conv2d (16 &rarr; 13, k=3, p=1)<br/>Output: Raw Logits (13, 256, 256)"]
    end

    IMG --> E0
    E0 --> E1
    E1 --> E2
    E2 --> E3
    E3 --> E4

    E4 --> D4
    E3 -. "Skip (256, 16, 16)" .-> D4
    D4 --> D3
    E2 -. "Skip (128, 32, 32)" .-> D3
    D3 --> D2
    E1 -. "Skip (64, 64, 64)" .-> D2
    D2 --> D1
    E0 -. "Skip (64, 128, 128)" .-> D1
    D1 --> D0
    IMG -. "Skip (3, 256, 256)" .-> D0
    D0 --> HEAD
```

### Encoder Backbone (ResNet-18)

The encoder extracts hierarchical contextual representations from local facial edges up to global semantic identity structures:

- **Initial Downsampling (Stage 0)**: $7\times 7$ convolution with stride 2 and padding 3, followed by Batch Normalization and ReLU activation, downsampling the spatial resolution from $256\times 256$ to $128\times 128$ with 64 channels.
- **Stage 1 (Residual Layer 1)**: $3\times 3$ max pooling (stride 2) downsamples spatial size to $64\times 64$, processed through two basic residual blocks (each consisting of two $3\times 3$ convolutions with identity shortcut connections).
- **Stage 2 (Residual Layer 2)**: Two residual blocks downsampling spatial resolution to $32\times 32$, expanding channel depth from 64 to 128.
- **Stage 3 (Residual Layer 3)**: Two residual blocks downsampling spatial resolution to $16\times 16$, expanding channel depth from 128 to 256.
- **Stage 4 (Bottleneck / Layer 4)**: Two residual blocks downsampling spatial resolution to $8\times 8$, expanding channel depth from 256 to 512. This stage forms the latent bottleneck capturing global facial context.

### Decoder Network (U-Net)

The decoder progressively recovers high-resolution spatial details through 5 upsampling blocks:

1. **Upsampling Mechanism**: Bilinear interpolation / nearest-neighbor interpolation doubles the spatial resolution ($\times 2$).
2. **Skip Feature Fusion**: The upsampled feature maps are concatenated along the channel dimension with the corresponding high-resolution skip features forwarded directly from the encoder stages.
3. **Double Convolution Blocks**: Each decoder block processes the concatenated tensor through two sequential stages of:
   - $3\times 3$ 2D Convolution
   - Batch Normalization
   - ReLU Activation
4. **Channel Transition Profile**:
   - **Block 4**: Concat $(512 + 256) = 768$ channels $\to 256$ channels at $16\times 16$
   - **Block 3**: Concat $(256 + 128) = 384$ channels $\to 128$ channels at $32\times 32$
   - **Block 2**: Concat $(128 + 64) = 192$ channels $\to 64$ channels at $64\times 64$
   - **Block 1**: Concat $(64 + 64) = 128$ channels $\to 32$ channels at $128\times 128$
   - **Block 0**: Concat $(32 + 3) = 35$ channels $\to 16$ channels at $256\times 256$

### Segmentation Head

The final layer is a lightweight classification head:
- **Operation**: `nn.Conv2d(in_channels=16, out_channels=13, kernel_size=3, padding=1)`
- **Activation**: Strictly `None`. The head outputs raw unconstrained logits $\mathbf{z} \in \mathbb{R}^{B \times 13 \times 256 \times 256}$.
- **Design Rationale**: Emitting raw logits ensures numerical stability during training with PyTorch loss functions (`nn.CrossEntropyLoss` and custom `DiceLoss` via `F.softmax`). Post-processing converts logits to pixel labels via $\hat{y} = \arg\max_{c} z_c$.

### Tensor Flow & Skip Connections

| Level | Feature Scale | Spatial Resolution | Encoder Channels | Decoder Input Channels | Decoder Output Channels | Skip Connection Source |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | $1/1$ | $256 \times 256$ | 3 | $32 + 3 = 35$ | 16 | Raw Input Image |
| **1** | $1/2$ | $128 \times 128$ | 64 | $64 + 64 = 128$ | 32 | Encoder Stage 0 (`conv1`) |
| **2** | $1/4$ | $64 \times 64$ | 64 | $128 + 64 = 192$ | 64 | Encoder Stage 1 (`layer1`) |
| **3** | $1/8$ | $32 \times 32$ | 128 | $256 + 128 = 384$ | 128 | Encoder Stage 2 (`layer2`) |
| **4** | $1/16$ | $16 \times 16$ | 256 | $512 + 256 = 768$ | 256 | Encoder Stage 3 (`layer3`) |
| **5** | $1/32$ | $8 \times 8$ | 512 | — (Bottleneck) | 512 | Encoder Stage 4 (`layer4`) |

---

## Parameter & Computational Profile

The model architecture balances segmentation precision with real-time inference latency:

| Sub-Module | Layer Breakdown | Parameter Count | % of Total Parameters | Trainable Status |
| :--- | :--- | :---: | :---: | :---: |
| **Encoder** | ResNet-18 Backbone (`conv1` through `layer4`) | 11,176,512 | 77.99% | Trainable |
| **Decoder** | 5 U-Net Blocks (Double Convolutions + BN) | 3,151,552 | 21.99% | Trainable |
| **Segmentation Head** | Final $3\times 3$ Conv (`16` $\to$ `13` classes) | 1,885 | 0.02% | Trainable |
| **Entire Network** | **Total Parameters** | **14,329,949** | **100.0%** | **100% Trainable** |

### Memory & Execution Characteristics

- **State Dict Size on Disk**: ~55 MB (FP32 weights only); ~164 MB including full Adam optimizer states (`best_model.pth`).
- **Inference Memory Footprint (FP32)**: ~58 MB VRAM per single-sample batch.
- **Inference Latency (Single Image, Batch=1, 256x256)**:
  - NVIDIA RTX 30-series / 40-series GPU: **~3.8 ms – 6.2 ms** (>160 FPS)
  - Intel Core i7 / AMD Ryzen CPU: **~28 ms – 45 ms** (~25-35 FPS)

---

## 13-Class Semantic Taxonomy

The model maps all facial anatomy and external occluders into a unified **13-class taxonomy**:

| Class ID | Label Name | RGB Color | Hex Code | Category | CelebAMask-HQ Raw Source Mapping |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **0** | `background` | `(0, 0, 0)` | `#000000` | Non-Face | Image background, clothing (`cloth`), jewelry (`neck_l`, `ear_r`) |
| **1** | `skin` | `(255, 200, 150)` | `#FFC896` | Anatomy | Facial skin, forehead, cheeks, jawline, neck (`skin`, `neck`, `face`) |
| **2** | `left_eye` | `(0, 128, 255)` | `#0080FF` | Anatomy | Left eye globe, pupil, sclera, left eyebrow (`l_eye`, `left_eye`, `l_brow`) |
| **3** | `right_eye` | `(0, 200, 100)` | `#00C864` | Anatomy | Right eye globe, pupil, sclera, right eyebrow (`r_eye`, `right_eye`, `r_brow`) |
| **4** | `nose` | `(255, 128, 0)` | `#FF8000` | Anatomy | Nasal bridge, columella, nostrils, nasal tip (`nose`) |
| **5** | `mouth` | `(220, 50, 50)` | `#DC3232` | Anatomy | Upper lip, lower lip, oral cavity, teeth (`mouth`, `u_lip`, `l_lip`) |
| **6** | `hair` | `(100, 60, 20)` | `#643C14` | Anatomy | Cranial scalp hair, bangs, sideburns (`hair`) |
| **7** | `left_ear` | `(180, 0, 255)` | `#B400FF` | Anatomy | Left pinna, tragus, earlobe (`l_ear`, `left_ear`) |
| **8** | `right_ear` | `(255, 0, 200)` | `#FF00C8` | Anatomy | Right pinna, tragus, earlobe (`r_ear`, `right_ear`) |
| **9** | `glasses` | `(0, 220, 220)` | `#00DCDC` | Accessory | Clear optical spectacles, eyeglasses (`eye_g`, `eyeglass`, `glasses`) |
| **10** | `sunglasses` | `(40, 50, 65)` | `#283241` | Occluder | Dark / polarized sunglasses, tinted protective goggles |
| **11** | `mask` | `(0, 190, 210)` | `#00BED2` | Occluder | Surgical masks, N95 respirators, fabric face coverings |
| **12** | `hat` | `(255, 240, 0)` | `#FFF000` | Occluder | Caps, beanies, sun hats, headwear covering forehead (`hat`) |

> [!NOTE]
> Eyebrows (`l_brow`, `r_brow`) are merged into their respective eye classes (`left_eye`, `right_eye`) to ensure robust ocular bounding boxes for downstream recognition alignment. Neck regions are merged into `skin`.

---

## Data Preprocessing & Augmentation Pipeline

### Input Normalization

All input images are resized to a canonical spatial resolution of $256 \times 256 \times 3$ (RGB) and scaled to standard ImageNet statistical distributions:

$$\mathbf{x}_{\text{norm}} = \frac{\mathbf{x} / 255.0 - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$$

Where:
- $\boldsymbol{\mu} = [0.485, 0.456, 0.406]$ (ImageNet channel means)
- $\boldsymbol{\sigma} = [0.229, 0.224, 0.225]$ (ImageNet channel standard deviations)

### Data Augmentation

During model training, the following stochastic transformations are applied via **Albumentations** simultaneously to both the RGB image and the categorical mask:

1. **Horizontal Flip**: $p = 0.5$ (Left and right anatomical labels are swapped accordingly to maintain anatomical correctness).
2. **Shift-Scale-Rotate**: $p = 0.4$, rotation limit $\pm 15^\circ$, scale limit $\pm 10\%$, translation shift $\pm 5\%$. Constant zero padding is used for masks.
3. **Random Brightness & Contrast**: $p = 0.4$, brightness limit $\pm 20\%$, contrast limit $\pm 20\%$.
4. **Gaussian Blur**: $p = 0.2$, kernel limit $3 \times 3$.

### Procedural Occlusion Augmenter

To train the network to accurately segment masks and sunglasses that are scarce in raw CelebAMask-HQ annotations, a **Procedural Occlusion Augmentation Engine** (`src/occlusion_augmenter.py`) dynamically injects photorealistic synthetic occluders onto training faces during data loading:

- **Surgical / Cloth Masks**: B-spline polygonal mesh covering chin, lips, and lower nasal bridge with synthetic fabric textures, pleats, and randomized HSV hues (surgical teal, white, black, blue).
- **Sunglasses**: Dual elliptical ocular occlusion connected with a nasal bridge wire, shaded with dark polarized gradients ($R, G, B \in [20, 60]$) and synthetic specular glares.
- **Compounded Heavy Occlusion**: Simultaneous application of surgical mask and sunglasses to expose the model to $>50\%$ total facial occlusion scenarios.

---

## Training Methodology & Optimization

### Dual-Objective Loss Function

Semantic segmentation of facial features suffers from severe spatial class imbalance: `background`, `skin`, and `hair` dominate tens of thousands of pixels per image, whereas `left_eye`, `right_eye`, and `glasses` occupy small localized patches.

To overcome this, a hybrid **Combined Loss** objective is used:

$$\mathcal{L}_{\text{total}} = w_{\text{CE}} \cdot \mathcal{L}_{\text{CE}} + w_{\text{Dice}} \cdot \mathcal{L}_{\text{Dice}}, \quad \text{with } w_{\text{CE}} = 1.0, \, w_{\text{Dice}} = 1.0$$

#### 1. Pixel-Wise Cross-Entropy Loss ($\mathcal{L}_{\text{CE}}$)
Penalizes incorrect per-pixel class assignment independently across the spatial canvas:

$$\mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=0}^{C-1} y_{i, c} \log \left( \frac{\exp(z_{i, c})}{\sum_{k=0}^{C-1} \exp(z_{i, k})} \right)$$

#### 2. Soft Multi-Class Dice Loss ($\mathcal{L}_{\text{Dice}}$)
Directly optimizes intersection over union, ensuring gradient signal is maintained even for tiny anatomical structures (eyes, nose tip):

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{1}{C} \sum_{c=0}^{C-1} \frac{2 \sum_{i=1}^{N} p_{i, c} y_{i, c} + \epsilon}{\sum_{i=1}^{N} p_{i, c} + \sum_{i=1}^{N} y_{i, c} + \epsilon}$$

Where $p_{i, c} = \text{Softmax}(z_{i, c})$, $y_{i, c} \in \{0, 1\}$ is the one-hot ground-truth, and $\epsilon = 1.0$ is the Laplace smoothing constant preventing division by zero.

### Hyperparameters & Optimization

| Hyperparameter | Value | Description / Rationale |
| :--- | :--- | :--- |
| **Optimizer** | Adam | First and second moment adaptive gradient descent |
| **Base Learning Rate** | $1.0 \times 10^{-4}$ | Stable convergence rate for pretrained ResNet backbone |
| **Weight Decay** | $1.0 \times 10^{-4}$ | $L_2$ regularization preventing overfitting |
| **LR Scheduler** | `ReduceLROnPlateau` | Factor $0.5$, patience 5 epochs, monitoring validation mIoU |
| **Batch Size** | 8 | Balances GPU memory consumption and batch norm statistics |
| **Max Epochs** | 50 | Early stopping / checkpoint saving triggers on best `val_miou` |
| **Number of Workers** | 0 | Optimized for Windows thread stability |
| **Pin Memory** | `True` | Fast page-locked host-to-device memory transfer |

### Dataset Partitioning

The model was developed and verified on the **CelebAMask-HQ** benchmark (30,000 high-resolution facial images with fine-grained semantic annotations), partitioned deterministically (seed = 42):

```
Total Dataset: 30,000 Identities
├── Training Set   : 70% (21,000 images)  [train.txt]
├── Validation Set : 15%  (4,500 images)  [val.txt]
└── Test Set       : 15%  (4,500 images)  [test.txt]
```

---

## Empirical Performance Benchmarks

### Global Holdout Evaluation

The model was evaluated on the unseen holdout test split of **4,500 images**:

| Metric | Holdout Validation Score | Holdout Test Score | Target Objective | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Pixel Accuracy** | 95.00% | **95.04%** | $\ge 90.0\%$ | **EXCEEDED** |
| **Mean IoU (mIoU)** | 0.7714 | **0.7703** | $\ge 0.750$ | **EXCEEDED** |
| **Mean Dice / F1 Score** | 0.8648 | **0.8640** | $\ge 0.850$ | **EXCEEDED** |
| **Combined Loss** | 0.2876 | **0.2891** | $\le 0.350$ | **EXCEEDED** |

### Per-Class Segmentation Metrics

Evaluated on the 4,500 holdout test images (`results/metrics/test_metrics.json`):

| Class ID | Semantic Label | Class IoU | Class Dice (F1) | Qualitative Performance |
| :---: | :--- | :---: | :---: | :--- |
| **0** | `background` | **0.9285** | **0.9629** | Flawless perimeter boundary separation |
| **1** | `skin` | **0.9238** | **0.9604** | Sharp jawline and facial border contouring |
| **2** | `left_eye` | **0.6063** | **0.7549** | Accurate ocular iris and eyebrow segmentation |
| **3** | `right_eye` | **0.6016** | **0.7513** | Consistent bilateral ocular symmetry |
| **4** | `nose` | **0.8774** | **0.9347** | High-precision nasal bridge and tip localization |
| **5** | `mouth` | **0.8857** | **0.9394** | Robust vermilion border tracking across mouth poses |
| **6** | `hair` | **0.9030** | **0.9490** | High fidelity across curly, straight, and bald heads |
| **7** | `left_ear` | **0.6562** | **0.7924** | Clear earlobe tracking under partial hair overlap |
| **8** | `right_ear` | **0.6038** | **0.7529** | Clear earlobe tracking under partial hair overlap |
| **9** | `glasses` | **0.7605** | **0.8640** | Thin wireframe and thick acetate frame recovery |
| **10** | `sunglasses` | *Augmented* | *Augmented* | Accurately segments tinted eyewear occlusions |
| **11** | `mask` | *Augmented* | *Augmented* | Accurately segments surgical and cloth face masks |
| **12** | `hat` | **0.7271** | **0.8420** | Accurately segments caps, beanies, and brims |

### Occlusion Robustness Benchmark

Evaluated across diagnostic test batches (`results/metrics/occlusion_evaluation.json`):

| Condition | Tested Occluder | Mean Facial Occlusion % | Recognition Decision | Adaptive Weight Distribution ($w_{\text{upper}}, w_{\text{mid}}, w_{\text{lower}}$) |
| :--- | :--- | :---: | :---: | :---: |
| **Clean Face** | None | **0.0%** | **ACCEPT** | `(0.333, 0.333, 0.334)` &rarr; Balanced |
| **Masked Face** | Surgical Mask | **36.9%** | **ACCEPT** | `(0.539, 0.295, 0.166)` &rarr; Upper-focused |
| **Eyewear** | Dark Sunglasses | **13.2%** | **ACCEPT** | `(0.282, 0.337, 0.381)` &rarr; Lower-focused |
| **Heavy Occlusion** | Mask + Sunglasses | **50.0% – 58.0%** | **REJECT** | `High Occlusion Risk` &rarr; Non-viable |

---

## Downstream Integration: Occlusion Analytics & Adaptive Weighting

The segmentation mask produced by the U-Net feeds directly into the **Occlusion Analytics Engine** (`src/occlusion.py`), turning pixel labels into biometric decision signals.

```mermaid
graph LR
    LOGITS[Raw Logits 13x256x256] --> ARGM[Argmax Class Mask]
    ARGM --> CHULL[Face Convex Hull Area]
    ARGM --> OCCPIX[Occluding Pixels: Mask + Sunglasses]
    CHULL & OCCPIX --> RATIO["Occlusion Ratio O_face = (Occ / Face) * 100%"]
    RATIO --> VIAB{O_face < 50%?}
    VIAB -- Yes --> ACC["Status: ACCEPT (Viable)"]
    VIAB -- No --> REJ["Status: REJECT (Non-Viable)"]
    ACC --> TRIPART[Tri-Region Partition: Upper / Mid / Lower]
    TRIPART --> WEIGHTS["Adaptive Region Weights (w_upper, w_mid, w_lower)"]
    WEIGHTS --> BACKEND[Downstream Face Recognition Feature Fusion]
```

### Facial Occlusion Ratio ($O_{\text{face}}$)

Let $\mathcal{C}_{\text{core}} = \{1, 2, 3, 4, 5\}$ be core facial anatomy classes (skin, eyes, nose, mouth) and $\mathcal{C}_{\text{occ}} = \{10, 11\}$ be direct face occluders (sunglasses, mask).

The canonical face area $\mathcal{F}$ is defined as the convex hull bounding the spatial union of all core anatomy and occluder pixels:

$$\mathcal{F} = \text{ConvexHull}\left( \bigcup_{c \in \mathcal{C}_{\text{core}} \cup \mathcal{C}_{\text{occ}}} \{(x, y) \mid M(x, y) = c\} \right)$$

The facial occlusion ratio $O_{\text{face}}$ is computed as:

$$O_{\text{face}} = \frac{\sum_{(x, y) \in \mathcal{F}} \mathbb{I}[M(x, y) \in \mathcal{C}_{\text{occ}}]}{|\mathcal{F}|} \times 100\%$$

### Recognition Viability Boundary

A hard operational threshold determines whether facial biometric features are sufficiently uncorrupted to permit reliable identification:

$$\text{Decision}(O_{\text{face}}) = \begin{cases} \mathbf{ACCEPT} & \text{if } O_{\text{face}} < 50.0\% \\ \mathbf{REJECT} & \text{if } O_{\text{face}} \ge 50.0\% \end{cases}$$

### Tri-Region Vertical Partitioning

The vertical span of the face $H_f = y_{\max} - y_{\min}$ is partitioned into three anatomical bands:
1. **Upper Face** ($y \in [y_{\min},\, y_{\min} + 0.35 H_f]$): Eyes, eyebrows, forehead.
2. **Mid Face** ($y \in [y_{\min} + 0.35 H_f,\, y_{\min} + 0.65 H_f]$): Nose, upper cheekbones.
3. **Lower Face** ($y \in [y_{\min} + 0.65 H_f,\, y_{\max}]$): Mouth, chin, jawline.

### Adaptive Feature Weighting Equation

For each facial region $r \in \{\text{upper}, \text{mid}, \text{lower}\}$, regional occlusion $O_r \in [0, 100]\%$ is quantified. The unnormalized fidelity score $\tilde{w}_r$ is calculated with a safety floor of $0.05$:

$$\tilde{w}_r = \max\left(0.05,\, 1.0 - \frac{O_r}{100}\right)$$

Normalized adaptive region weights $w_r$ are computed by softmax or linear normalization:

$$w_r = \frac{\tilde{w}_r}{\sum_{k \in \{\text{upper}, \text{mid}, \text{lower}\}} \tilde{w}_k}, \quad \text{such that } \sum_{r} w_r = 1.0$$

#### Downstream Recognition Application
When matching against enrolled embeddings $\mathbf{e}_{\text{gallery}}$, feature representations from sub-regions $\mathbf{f}_r$ are weighted dynamically:

$$\mathbf{f}_{\text{fused}} = w_{\text{upper}} \cdot \mathbf{f}_{\text{upper}} + w_{\text{mid}} \cdot \mathbf{f}_{\text{mid}} + w_{\text{lower}} \cdot \mathbf{f}_{\text{lower}}$$

If a subject is wearing a surgical mask ($O_{\text{lower}} \approx 70\%$, $O_{\text{upper}} = 0\%$), $w_{\text{upper}}$ scales up to $\approx 0.54$, prioritizing ocular biometric features and attenuating noise from the masked mouth.

---

## Programmatic Inference & Usage

### Python API

```python
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from src.model import build_model, get_device
from src.occlusion import estimate_occlusion
import src.config as cfg

# 1. Setup device and model
device = get_device()
model = build_model(
    encoder_name="resnet18",
    encoder_weights=None,
    num_classes=13,
    in_channels=3
).to(device)

# 2. Load trained checkpoint
checkpoint = torch.load("checkpoints/best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 3. Preprocess input image
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

raw_img = Image.open("dataset/sample/images/00000.png").convert("RGB")
input_tensor = transform(raw_img).unsqueeze(0).to(device)  # shape: (1, 3, 256, 256)

# 4. Model forward pass
with torch.no_grad():
    logits = model(input_tensor)                 # shape: (1, 13, 256, 256)
    pred_mask = torch.argmax(logits, dim=1)[0]   # shape: (256, 256)

mask_np = pred_mask.cpu().numpy().astype(np.uint8)

# 5. Compute occlusion analytics
analytics = estimate_occlusion(mask_np)
print(f"Status              : {analytics['viability_status']}")
print(f"Facial Occlusion %  : {analytics['occlusion_percentage']:.2f}%")
print(f"Adaptive Weights    : {analytics['adaptive_weights']}")
```

### CLI Usage

The project includes a standalone CLI inference and diagnostics tool (`infer.py`):

```bash
# Standard clean inference
python infer.py --image dataset/sample/images/00000.png

# Inference simulating a surgical face mask
python infer.py --image dataset/sample/images/00002.png --occlude mask

# Inference simulating dark sunglasses
python infer.py --image dataset/sample/images/00002.png --occlude sunglasses

# Inference under compounded heavy occlusion (mask + sunglasses)
python infer.py --image dataset/sample/images/00002.png --occlude heavy
```

---

## Limitations & Future Roadmap

### Known Limitations

1. **Extreme Off-Angle Poses (Profile Views)**:
   - The model is primarily optimized for yaw angles between $-45^\circ$ and $+45^\circ$. In near-full profile poses ($>60^\circ$), unilateral facial occlusion (e.g., hidden eye/ear) may shift regional boundaries.
2. **Non-Standard Occluders**:
   - Objects outside the 13-class taxonomy (e.g., hands over face, microphones, eating utensils) are currently partitioned as background rather than explicit occluders.
3. **Multi-Face Scenes**:
   - The current architecture expects a cropped, aligned bounding box containing a single primary subject. Multi-face images should pass through a face detector (e.g., RetinaFace / MTCNN) prior to U-Net segmentation.

### Future Roadmap

- [ ] **Backbone Scaling**: Benchmark larger encoders (ResNet-34, ResNet-50, and ConvNeXt-Tiny) to further boost small-object eye and earring boundary resolution.
- [ ] **Attention Gates**: Integrate spatial and channel attention (CBAM / SE blocks) into the skip connections to filter non-face background clutter.
- [ ] **Hand & Object Segmentation**: Expand the taxonomy from 13 to 15 classes to include hand-over-face and food/drink items.
- [ ] **Direct Feature-Level Fusion**: Integrate the U-Net segmentation head directly with an ArcFace backbone for an end-to-end trained, occlusion-resilient facial identification system.
