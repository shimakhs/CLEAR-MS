
# MS vs HC Classification from OCT Images using Autoencoder and Dense Classifier

This repository contains a Python implementation of an AI-based pipeline for classifying Multiple Sclerosis (MS) and Healthy Controls (HC) from Optical Coherence Tomography (OCT) images. The model leverages a custom autoencoder for feature extraction followed by a dense neural network for binary classification.

---

## 🧠 Project Overview

- **Goal**: Automatically distinguish MS patients from healthy individuals based on retinal OCT boundary maps.
- **Approach**:
  - Preprocessing OCT data into differential boundary layers
  - Feature extraction via an autoencoder
  - Classification using a shallow dense network
  - Evaluation through 5-fold Stratified Cross-Validation

---

## 📁 Dataset

The dataset includes:
- OCT boundary predictions from two datasets
- Labels indicating MS or HC
- `ScanPosition.pkl`, `XLayersBoundaryMap.pkl`, `TLayersBoundaryMap.pkl`, `HMlabels.pkl`

> **Note**: These `.pkl` files must be placed in the root directory before running the training script.

---

## 📦 Dependencies

Install the required libraries using:

```bash
pip install -r requirements.txt
```

**Main packages**:
- `TensorFlow`
- `scikit-learn`
- `opencv-python`
- `matplotlib`
- `numpy`
- `pandas`

---

## 🚀 Running the Code

Train the autoencoder and classifier by running:

```bash
python ms_oct_autoencoder.py
```

---

## 🧪 Evaluation Metrics

During training and evaluation, the following metrics are computed and logged:

- Accuracy
- Specificity
- Sensitivity
- G-mean
- Balanced Accuracy
- F1 Score
- Recall
- Precision

Results are saved to:

```
results.csv
```

---

## 📊 Visualization

- Confusion matrices are plotted per fold
- Training loss and accuracy curves are saved
- ROC curve for the final fold is displayed

---

## 🧬 Model Architecture

### Autoencoder

- Input: (3, 60, 256) image
- Encoder: Flatten → Dense(100) → Dense(50) → Dense(25)
- Decoder: Dense(50) → Dense(100) → Dense(original size) → Reshape

### Classifier

- Dense(50, tanh) → Dense(1, sigmoid)

---

## 📌 Notes

- The pipeline includes optional augmentation hooks (not active in the current script).
- Evaluation is done using Stratified K-Fold cross-validation for robustness.
- This code is modular and can be easily extended to other retinal OCT datasets or other disease classification tasks.

---

## 📬 Contact

For questions or collaboration opportunities, please reach out via [GitHub Issues](https://github.com/your_username/your_repo/issues) or email.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
