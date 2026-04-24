# High-Frequency Information Supported Domain Adaptation for Cross-Domain Object Detection(HIDA-YOLO)

This repository contains the implementation of **HIDA-YOLO**, a domain adaptive object detection framework with high-frequency detail enhancement.

---

## 🧠 Framework Overview

<p align="center"> <img src="https://github.com/1van-yyf/HIDA-YOLO/blob/main/data/HIDA.png" width="750"/> </p>

---

## 🔍 Internal Modules

### Domain Classifier
<p align="center">
  <img src="https://github.com/1van-yyf/HIDA-YOLO/blob/main/data/domain%20classifier.png" width="500"/>
</p>

The domain classifier in HIDA follows a standard design. It consists of four key components:

- **Gradient Reversal Layer (GRL):** Applies a negative gradient (−λ) to encourage the backbone to learn domain-invariant features.  
- **Convolutional Layers:** Two convolutional layers compress the feature channels to a single-channel representation.  
- **Global Average Pooling (GAP):** Reduces spatial dimensions to a scalar output.  
- **Domain Loss:** A binary cross-entropy (BCE) loss is applied on the sigmoid-activated output to measure inter-domain consistency.

### Multi-Label Classifier
<p align="center">
  <img src="https://github.com/1van-yyf/HIDA-YOLO/blob/main/data/multi-label%20classifier.png" width="500"/>
</p>

The multi-label classifier in HIDA consists of three key components:

- **Convolutional Layers:** Two convolutional layers compress the feature channels to a $C$-dimensional representation, where $C$ denotes the number of categories.  
- **Global Average Pooling (GAP):** Aggregates spatial features into a category-wise vector.  
- **Multi-Label Loss:** The classifier is optimized using Asymmetric Loss (ASL), a variant of focal loss, to address label imbalance.

## 📋 Requirements

Please install the required environment before running the code.  
Dependencies are listed in **requirements_S.txt**:

## 📂 Dataset Preparation

You need to prepare both the source and target datasets before training.
- **PASCAL VOC 2007/2012**  
  [Download VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/)  
- **Clipart1k Dataset**  
  [Download Clipart1k dataset](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets)

## 🔧 Step 1. Input-level Detail Refinement (IDR)

Run the **IDR_HF_scharr.py** script to perform high-frequency detail refinement on the dataset.

## 🔧 Step 2. Training HIDA-YOLO

### YOLOv9-S

Before training with YOLOv9, please ensure the pretrained weights are properly prepared.

Then run:

```bash
python train_hida_4ch_v9s.py \
  --epochs 201 \
  --device 0 \
  --batch 16 \
  --data xx \
  --img 640 \
  --cfg /models/detect/gelan-s_grl_mcls_4ch.yaml \
  --name xx
```

---

### YOLOv8

Before training with YOLOv8, please download the official pretrained weights:

- [YOLOv8-S](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt)
- [YOLOv8-M](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt)

Then run:

```bash
python train_hida_4ch_v8.py \
  --epochs 201 \
  --device 0 \
  --batch 16 \
  --data xx \
  --img 640 \
  --cfg /models/detect/yolov8s_hida.yaml \
  --name xx
```

You can replace the configuration file (e.g., `yolov8m_hida.yaml`) according to the selected model scale.



