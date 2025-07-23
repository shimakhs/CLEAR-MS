import numpy as np
import cv2
import pickle

dim1 = 60
dim2 = 256

def load_data(data_path):
    with open(f'{data_path}/ScanPosition.pkl', 'rb') as f:
        sp = pickle.load(f)
    with open(f'{data_path}/XLayersBoundaryMap.pkl', 'rb') as f:
        X = pickle.load(f)
    with open(f'{data_path}/HMlabels.pkl', 'rb') as f:
        Y = pickle.load(f)

    for i in range(len(sp)):
        if sp[i]:
            X[i] = X[i][:, :, ::-1, :]
    return X, Y

def preprocess_data(X):
    dataset = np.zeros((len(X), dim1, dim2, 9))
    for j in range(len(X)):
        img = X[j].squeeze()
        img = cv2.resize(img, (dim2, dim1))
        num = img.shape[2]
        def_img = np.zeros_like(img)
        for k in range(num - 1):
            def_img[:, :, k] = img[:, :, k + 1] - img[:, :, k]
        def_img[:, :, num - 1] = img[:, :, num - 1] - img[:, :, 0]
        dataset[j] = def_img
    return dataset

def final_dataset(dataset, Y, ch=[0, 1, 2]):
    num_channel = len(ch)
    new_dataset = np.zeros((len(dataset), dataset[0].shape[0], dataset[0].shape[1], num_channel))
    for k in range(len(dataset)):
        for j in range(num_channel):
            I = dataset[k, :, :, ch[j]]
            new_dataset[k, :, :, j] = ((I - I.min()) / (I.max() - I.min())).astype('float32')
    Y = np.array(Y).squeeze()
    return new_dataset, Y
