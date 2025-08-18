# HIDA-YOLO

This repository contains the implementation of **HIDA-YOLO**, a domain adaptive object detection framework with high-frequency detail enhancement.

---

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

## 🔧 Step 2. Traning HIDA-YOLO
python train_HIDA_4ch_s.py



