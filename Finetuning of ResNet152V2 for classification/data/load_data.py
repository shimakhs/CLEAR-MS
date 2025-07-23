
import pickle
import numpy as np
import cv2

dim1, dim2 = 224, 224
ch = [0, 1, 2]

def load_pickle(filename):
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)

def preprocess_data(X, sp):
    dataset = np.zeros((len(X), dim1, dim2, 9))
    for i, img in enumerate(X):
        if sp[i]:
            img = img[:, :, ::-1, :]
        img = img.squeeze()
        img = cv2.resize(img, (dim2, dim1))
        def_img = np.zeros_like(img)
        for k in range(img.shape[2] - 1):
            def_img[:, :, k] = img[:, :, k + 1] - img[:, :, k]
        def_img[:, :, -1] = img[:, :, -1] - img[:, :, 0]
        dataset[i] = def_img
    return dataset

def normalize_dataset(dataset, ch):
    new_data = np.zeros((dataset.shape[0], dataset.shape[1], dataset.shape[2], len(ch)))
    for i in range(dataset.shape[0]):
        for j, c in enumerate(ch):
            layer = dataset[i, :, :, c]
            new_data[i, :, :, j] = (layer - layer.min()) / (layer.max() - layer.min())
    return new_data

def load_and_preprocess_data():
    sp = load_pickle("ScanPosition")
    X = load_pickle("XLayersBoundaryMap")
    Y = np.array(load_pickle("HMlabels")).squeeze()
    dataset = preprocess_data(X, sp)
    new_dataset = normalize_dataset(dataset, ch)
    return new_dataset, Y
