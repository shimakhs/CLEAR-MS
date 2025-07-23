# CNN-Based Classification of Multiple Sclerosis Using Retinal OCT Images from scratch

This repository contains the implementation of a Convolutional Neural Network (CNN) for the classification of Multiple Sclerosis (MS) using retinal Optical Coherence Tomography (OCT) data. The model was trained and evaluated on pre-processed OCT layer boundary maps using a K-fold cross-validation strategy.

## Project Structure

```
MS_OCT_CNN/
├── data/               # Place your .pkl input files here
├── models/             # Saved models after training
├── results/            # Evaluation results and plots
├── src/                # Source code for preprocessing, training, evaluation
│   ├── preprocess.py   # Loading and preprocessing of OCT data
│   ├── model.py        # CNN architecture
│   ├── train.py        # Model training with K-fold cross-validation
│   └── evaluate.py     # Evaluation metrics and visualization
├── main.py             # Entry point for training the model
└── requirements.txt    # List of required Python packages
```

## Method Overview

The pipeline includes:
- **Loading and Preprocessing** of OCT boundary maps from `.pkl` files.
- **Automatic feature extraction** using a deep CNN with multiple convolutional and normalization layers.
- **Training and Evaluation** using Stratified K-Fold cross-validation and data augmentation.
- **Performance Metrics** such as accuracy, confusion matrix, and balanced accuracy.

##  Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/MS_OCT_CNN.git
cd MS_OCT_CNN
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

##  Usage

1. Add your `.pkl` files (`ScanPosition.pkl`, `XLayersBoundaryMap.pkl`, `HMlabels.pkl`) into the `data/` directory.

2. Run the training pipeline:
```bash
python main.py
```

Model checkpoints and evaluation plots will be saved in the `models/` and `results/` folders.

##  Output

- Training/validation loss and accuracy curves
- Confusion matrix
- Classification report
- Balanced accuracy score

##  Citation

If you use this code for your research, please cite the related paper:

```
CLEAR-MS: Comprehensive Evaluation of Artificial Intelligence Models for Diagnosis of Multiple Sclerosis Using Information from Retinal Layers Multicenter OCT Images
```

##  Contact

For questions, feel free to contact shimakhodabandeh@gmail.com

