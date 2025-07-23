
from data.load_data import load_and_preprocess_data
from models.model import classify_model
from utils.evaluation import evaluate_model
from utils.plotting import plot_loss
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def main():
    X, Y = load_and_preprocess_data()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=52)

    acc_arr = []
    bacc_arr = []
    yt, yp = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, Y), 1):
        print(f"Fold {fold}")
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]

        y_train_oh = to_categorical(y_train)
        y_test_oh = to_categorical(y_test)

        model = classify_model()
        datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=20)
        datagen.fit(x_train)
        train_gen = datagen.flow(x_train, y_train_oh, batch_size=12)

        history = model.fit(train_gen, epochs=200, validation_data=(x_test, y_test_oh), verbose=1)
        plot_loss(history, 200)

        acc, bacc, y_true, y_pred = evaluate_model(model, x_test, y_test)
        acc_arr.append(acc)
        bacc_arr.append(bacc)
        yt.extend(y_true)
        yp.extend(y_pred)

    evaluate_model(None, np.array(yt), np.array(yp), final=True)
    print("Average Accuracy:", np.mean(acc_arr))
    print("Average Balanced Accuracy:", np.mean(bacc_arr))

if __name__ == "__main__":
    main()
