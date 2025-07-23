
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
from utils.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

def evaluate_model(model, x_test, y_test, final=False):
    if not final:
        y_pred = np.argmax(model.predict(x_test), axis=1)
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, target_names=["HC", "MS"])
        print(classification_report(y_test, y_pred, digits=4))
        acc = np.mean(y_pred == y_test)
        bacc = balanced_accuracy_score(y_test, y_pred)
        return acc, bacc, y_test.tolist(), y_pred.tolist()
    else:
        cm = confusion_matrix(x_test, y_test)
        plot_confusion_matrix(cm, target_names=["HC", "MS"], normalize=True)
        print(classification_report(x_test, y_test, digits=4))
        print("Final Accuracy:", np.mean(np.array(x_test) == np.array(y_test)))
