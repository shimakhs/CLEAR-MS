# Fine-tuning of pre-trained network like ResNet152V2 for MS vs HC OCT Classification

This project implements a deep learning pipeline using a pre-trained **ResNet152V2** model to classify **Multiple Sclerosis (MS)** vs **Healthy Controls (HC)** from **OCT (Optical Coherence Tomography)** retinal layers.

##  Dataset

The model uses segmented retinal layers:

- `XLayersBoundaryMap.pkl`: 4D numpy array of boundary maps (B-scans).
- `HMlabels.pkl`: binary labels (0 = HC, 1 = MS).
- `ScanPosition.pkl`: eye scan orientation metadata to normalize left/right eyes.

All data must be preprocessed using the provided functions.

---

##  Pipeline Overview

1. **Data Preprocessing**:
   - Thicknessmaps are computed to emphasize thickness of retinal layers.
   - Thicknessmaps are normalized and stacked for model input.
   - Left eye images are flipped to align with right eye format.

2. **Model Architecture**:
   - ResNet152V2 is used as a frozen feature extractor.
   - A custom classification head is added with fully connected layers.
   - Final output: 2-class softmax.

3. **Training**:
   - 5-fold **StratifiedKFold** cross-validation.
   - Image data is augmented (flip, rotation).
   - Best model checkpointed based on validation accuracy.

4. **Evaluation**:
   - Accuracy, Sensitivity, Specificity, Balanced Accuracy, and G-Mean.
   - ROC curve and Confusion Matrix plotted.

---

## Requirements

```txt
tensorflow>=2.8
keras
numpy
opencv-python
matplotlib
scikit-learn
seaborn
pandas
imgaug
sporco
```

Install them via:

```bash
pip install -r requirements.txt
```

---

##  Usage

1. **Prepare your environment**:
   - Place `XLayersBoundaryMap.pkl`, `HMlabels.pkl`, and `ScanPosition.pkl` in the root directory.

2. **Run training**:
   ```bash
   python train.py
   ```

3. **Visual Outputs**:
   - Training/Validation loss and accuracy plots.
   - Fold-specific confusion matrices.
   - Final ROC curve and classification report.
  
     
4. **Outputs**:
Training/validation loss and accuracy curves
Confusion matrix
Classification report
Balanced accuracy score
---

## Notes

- Input image size is fixed at `224x224x3`.
- You can change selected layers in the `ch = [0, 1, 2]` variable.
- `ImageDataGenerator` is used to perform augmentation on-the-fly.

---

## References

- ResNet152V2: [Keras Applications Docs](https://keras.io/api/applications/resnet/#resnet152v2-function)
- OCT Image Preprocessing adopted from domain-specific heuristics.
## Citation
If you use this code for your research, please cite the related paper:

CLEAR-MS: Comprehensive Evaluation of Artificial Intelligence Models for Diagnosis of Multiple Sclerosis Using Information from Retinal Layers Multicenter OCT Images

## Contact
For questions, feel free to contact shimakhodabandeh@gmail.com

