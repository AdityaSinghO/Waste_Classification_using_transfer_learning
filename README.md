# Waste Product Classification Using Transfer Learning & Fine-Tuning

A deep learning project that classifies waste images as **Organic (O)** or **Recyclable (R)** using transfer learning and fine-tuning on a pre-trained VGG16 model.

---

## Project Overview

This project tackles binary image classification for waste sorting — distinguishing between organic and recyclable materials. Two approaches are compared:

1. **Transfer Learning (Feature Extraction)** — VGG16 layers fully frozen; only the custom classifier head is trained.
2. **Fine-Tuning** — The top convolutional block (`block5_conv3` onwards) of VGG16 is unfrozen and trained alongside the classifier head.

---

## Dataset

- **Source:** [IBM Cloud Object Storage](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/kd6057VPpABQ2FqCbgu9YQ/o-vs-r-split-reduced-1200.zip)
- **Classes:** `O` (Organic), `R` (Recyclable)
- **Structure:**
  ```
  o-vs-r-split/
  ├── train/
  │   ├── O/
  │   └── R/
  └── test/
      ├── O/
      └── R/
  ```
- **Split:** 80% training / 20% validation (from training directory); separate test set

---

## Model Architecture

Both models are built on top of **VGG16** (pre-trained on ImageNet) with a custom head:

```
VGG16 Base (frozen or partially unfrozen)
  → Flatten
  → Dense(512, relu)
  → Dropout(0.3)
  → Dense(512, relu)
  → Dropout(0.3)
  → Dense(1, sigmoid)   ← binary output
```

---

## Training Configuration

| Parameter         | Transfer Learning Model | Fine-Tuned Model          |
|-------------------|-------------------------|---------------------------|
| Input size        | 150 × 150 × 3           | 150 × 150 × 3             |
| Batch size        | 32                      | 32                        |
| Epochs            | 10 (with early stopping)| 10 (with early stopping)  |
| Optimizer         | Adam (lr = 1e-5)        | RMSprop (lr = 1e-4)       |
| Loss              | Binary Crossentropy     | Binary Crossentropy       |
| Frozen layers     | All VGG16 layers        | All except block5_conv3+  |
| LR Schedule       | Exponential decay       | Exponential decay         |

**Callbacks used:**
- `EarlyStopping` — monitors `val_loss`, patience of 4 epochs
- `ModelCheckpoint` — saves the best model weights
- `LearningRateScheduler` — exponential decay schedule

---

## Data Augmentation

Applied to the training set:
- Horizontal flip
- Width & height shift (±10%)
- Rescaling (pixel values normalized to [0, 1])

---

## Results

Both trained models are evaluated on the held-out test set (50 Organic + 50 Recyclable images). Performance is reported via a **classification report** (precision, recall, F1-score).

Predictions are also visualized by overlaying the actual and predicted label on test images for quick qualitative inspection.

---

## Saved Models

| File | Description |
|------|-------------|
| `O_R_tlearn_vgg16.keras` | Best feature-extraction (transfer learning) model |
| `O_R_tlearn_fine_tune_vgg16.keras` | Best fine-tuned model |

---

## Requirements

```
tensorflow==2.17.0
numpy
scikit-learn==1.5.1
matplotlib==3.9.2
requests
tqdm
```

Install with:
```bash
pip install tensorflow==2.17.0 numpy scikit-learn==1.5.1 matplotlib==3.9.2 tqdm requests
```

---

## How to Run

1. Clone or download the notebook (`Final_Proj-Classify_Waste_Products_Using_TL-_FT-v1.ipynb`).
2. Install the dependencies listed above.
3. Run all cells in order — the dataset is downloaded and extracted automatically.
4. Training will begin and the best checkpoint for each model is saved to disk.
5. The final cells load both saved models and print the classification report and sample predictions.

---

## Project Structure

```
.
├── Final_Proj-Classify_Waste_Products_Using_TL-_FT-v1.ipynb
├── O_R_tlearn_vgg16.keras              # saved after training
├── O_R_tlearn_fine_tune_vgg16.keras    # saved after fine-tuning
├── o-vs-r-split/                       # downloaded & extracted automatically
│   ├── train/
│   └── test/
└── README.md
```

---

## Key Concepts Demonstrated

- **Transfer learning** with a frozen convolutional base
- **Fine-tuning** by selectively unfreezing upper layers
- **Image data augmentation** with `ImageDataGenerator`
- **Learning rate scheduling** with exponential decay
- **Model evaluation** with scikit-learn classification reports
- **Visual prediction inspection** on test images
