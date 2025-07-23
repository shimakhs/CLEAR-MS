import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from .model import CNN_model
from .evaluate import plot_loss, evaluate_model

def train_model(data, labels, output_path="models/model_fold.h5", num_folds=5):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(skf.split(data, labels)):
        print(f"\n🧪 Fold {fold + 1}/{num_folds}")
        x_train, x_test = data[train_idx], data[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        model = CNN_model(input_shape=x_train.shape[1:])
        checkpoint = ModelCheckpoint(f"{output_path}_{fold}.h5", monitor="val_accuracy", save_best_only=True)

        datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=15)
        datagen.fit(x_train)

        history = model.fit(datagen.flow(x_train, y_train, batch_size=16),
                            epochs=50,
                            validation_data=(x_test, y_test),
                            callbacks=[checkpoint],
                            verbose=1)

        plot_loss(history)
        evaluate_model(model, x_test, y_test, class_names=["HC", "MS"])
