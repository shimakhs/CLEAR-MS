import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def plot_loss(history):
    plt.plot(history.history['loss'], 'r', label='Training loss')
    plt.plot(history.history['val_loss'], 'b', label='Validation loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], 'r', label='Training acc')
    plt.plot(history.history['val_accuracy'], 'b', label='Validation acc')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

def evaluate_model(model, X_test, y_test, class_names):
    from sklearn.metrics import balanced_accuracy_score

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)
    print(classification_report(y_true, y_pred_classes, digits=4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.show()

    bal_acc = balanced_accuracy_score(y_true, y_pred_classes)
    print(f"Balanced Accuracy: {bal_acc:.4f}")
