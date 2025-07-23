
import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_loss(history, epochs):
    plt.plot(range(epochs), history.history['loss'], 'r', label='Training Loss')
    plt.plot(range(epochs), history.history['val_loss'], 'b', label='Validation Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.show()

    plt.plot(range(epochs), history.history['accuracy'], 'r', label='Training Accuracy')
    plt.plot(range(epochs), history.history['val_accuracy'], 'b', label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.show()

def plot_confusion_matrix(cm, target_names, normalize=False, cmap='Blues', title='Confusion matrix'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names)
    plt.yticks(tick_marks, target_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] > 0:
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
