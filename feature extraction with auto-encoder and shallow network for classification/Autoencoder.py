# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 09:02:15 2022

@author: Shima
"""
import os
import gc
import cv2
import pickle
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, balanced_accuracy_score,
    f1_score, recall_score, precision_score
)
from sklearn.neighbors import NearestNeighbors

import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, models
from tensorflow.keras.models import load_model

# Constants
INPUT_SHAPE = (3, 60, 256)
EPOCHS = 300
BATCH_SIZE = 32

# Dataframes for result logging
df_results = pd.DataFrame(columns=["Method", 'Accuracy', 'Specificity', 'Sensitivity', 'GMean', 'Balanced_Accuracy', 'F1_Score', 'Recall', 'Precision'])


def load_data():
    with open('ScanPosition.pkl', 'rb') as f:
        scan_position = pickle.load(f)
    with open('XLayersBoundaryMap.pkl', 'rb') as f:
        X = pickle.load(f)
    with open('TLayersBoundaryMap.pkl', 'rb') as f:
        T = pickle.load(f)
    with open('HMlabels.pkl', 'rb') as f:
        Y = pickle.load(f)

    for i in range(len(scan_position)):
        if scan_position[i]:
            X[i] = X[i][:, :, ::-1, :]
            T[i] = T[i][:, :, ::-1, :]

    processed = []
    for j in range(len(X)):
        slices = []
        for i in range(3):
            diff = X[j][0, :, :, i] - X[j][0, :, :, i + 1]
            resized = cv2.resize(diff, (256, 60))
            slices.append(resized)
        processed.append(np.array(slices))

    return np.array(processed), np.array(Y)


def build_autoencoder(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    x = layers.Dense(100, activation='tanh', activity_regularizer=regularizers.l2(1e-5))(x)
    x = layers.Dense(50, activation='tanh', activity_regularizer=regularizers.l2(1e-5))(x)
    encoded = layers.Dense(25, activation='tanh', activity_regularizer=regularizers.l2(1e-5))(x)

    x = layers.Dense(50, activation='tanh', activity_regularizer=regularizers.l2(1e-5))(encoded)
    x = layers.Dense(100, activation='tanh', activity_regularizer=regularizers.l2(1e-5))(x)
    x = layers.Dense(np.prod(input_shape), activation='tanh')(x)
    decoded = layers.Reshape(input_shape)(x)

    autoencoder = models.Model(inputs, decoded)
    encoder = models.Model(inputs, encoded)

    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
    return autoencoder, encoder


def plot_conf_matrix(cm, labels):
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def run():
    X, y = load_data()
    autoencoder, encoder = build_autoencoder(INPUT_SHAPE)
    
    checkpoint = callbacks.ModelCheckpoint("ae_model.h5", monitor="loss", save_best_only=True)
    autoencoder.fit(X, X, epochs=80, callbacks=[checkpoint], verbose=1)

    encoder = models.load_model("ae_model.h5", compile=False)
    X_encoded = encoder.predict(X)
    del X, autoencoder
    gc.collect()

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds, all_labels = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_encoded, y)):
        print(f"\nRunning fold {fold+1}/5")
        X_train, X_test = X_encoded[train_idx], X_encoded[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = models.Sequential([
            layers.Dense(50, activation='tanh', input_shape=(X_train.shape[1],)),
            layers.Dense(1, activation='sigmoid')
        ])
        clf.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(3.4e-3), metrics=['accuracy'])

        clf.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test), verbose=1)

        preds = clf.predict(X_test).flatten()
        preds_binary = (preds > 0.5).astype(int)
        all_preds.extend(preds_binary)
        all_labels.extend(y_test)

        cm = confusion_matrix(y_test, preds_binary)
        plot_conf_matrix(cm, ["HC", "MS"])

    # Overall Evaluation
    acc = accuracy_score(all_labels, all_preds)
    spec = confusion_matrix(all_labels, all_preds)[0, 0] / sum(confusion_matrix(all_labels, all_preds)[0])
    sens = confusion_matrix(all_labels, all_preds)[1, 1] / sum(confusion_matrix(all_labels, all_preds)[1])
    gmean = np.sqrt(sens * spec)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)

    df_results.loc[len(df_results)] = [
        "Autoencoder + Dense Classifier", acc, spec, sens, gmean, bal_acc, f1, recall, precision
    ]

    df_results.to_csv("results.csv", index=False)
    print("\nFinal Evaluation Results:")
    print(df_results)


if __name__ == "__main__":
    run()












