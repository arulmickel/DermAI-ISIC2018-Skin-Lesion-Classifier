# DermAI-ISIC2018-Skin-Lesion-Classifier (PyTorch, Transfer Learning)

Skin lesion classification using the **ISIC 2018 Task 3** dataset. This project trains a CNN with **transfer learning**, applies leakage-aware splitting practices (patient/lesion-safe when metadata is available), handles **class imbalance**, and evaluates performance using **F1/AUC** (not only accuracy) plus confusion-matrix error analysis.

---

## Highlights
- Transfer learning with **PyTorch + torchvision** (ResNet backbone)
- Data cleaning: removes rows with missing image files
- Leakage-aware split (recommended: patient/lesion-safe group split if metadata is available)
- Class imbalance handling: **loss weighting** / **weighted sampling** (configurable)
- Evaluation: **macro/weighted F1**, optional **ROC-AUC (OvR)**, confusion matrix
- Optional interpretability: Grad-CAM / saliency maps (future improvement)

---

## Classes (ISIC 2018 Task 3)
`MEL, NV, BCC, AKIEC, BKL, DF, VASC`

---

## Dataset
This repository **does not include** the ISIC dataset (large + dataset usage terms).

### Download (Colab-friendly)
Your notebook downloads the training images zip and unzips locally:
- Images: `ISIC2018_Task3_Training_Input.zip`
- Labels CSV: `ISIC2018_Task3_Training_GroundTruth.csv`

> Tip: Keep dataset files out of GitHub. Put them in a local `data/` folder and ignore via `.gitignore`.

---

## Approach
### 1) Data Preparation
- Reads label CSV (one-hot columns for 7 classes)
- Removes rows where the corresponding `.jpg` image is missing
- Standard preprocessing:
  - Resize to **224×224**
  - Normalize using ImageNet mean/std

### 2) Split Strategy (Leakage-Aware)
Medical imaging can have data leakage if the same patient/lesion appears in both train and validation.
- Baseline: random split
- Recommended upgrade: **group split** using patient/lesion metadata (when available)

### 3) Model
- Backbone: **ResNet18** (pretrained)
- Final layer replaced for **7-class classification**
- Loss: CrossEntropyLoss (optionally class-weighted)
- Optimizer: Adam

### 4) Evaluation
- Classification report (precision/recall/F1)
- Confusion matrix heatmap
- Optional: ROC-AUC one-vs-rest (recommended)

---

## Results
> Replace these with your best run numbers once you train longer than the 3-epoch test run.

- Macro F1: `0.xx`
- Weighted F1: `0.xx`
- ROC-AUC (OvR): `0.xx`

Confusion Matrix:
![Confusion Matrix](assets/confusion_matrix.png)

---

## Disclaimer
- This project is for educational and research purposes only and is not a medical device. Do not use for clinical decisions.
## How to Run

-----

### Option 1 - Run in Google Colab
1. Open the notebook:
   - `notebooks/ISIC2018_Skin_Lesion_Classification.ipynb`
2. Run cells in order
3. Upload the ground truth CSV when prompted (or modify notebook to auto-download)

### Option 2 - Run Locally
#### 1) Create environment + install
```bash
python -m venv .venv
source .venv/bin/activate   # mac/linux
# .venv\Scripts\activate    # windows
pip install -r requirements.txt


