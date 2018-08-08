import pickle
import numpy as np

def build_confusion_matrix(true_label_path, predicted_label_path):
    true_labels = pickle.load(open(true_label_path, 'rb'))
    predicted_labels = pickle.load(open(predicted_label_path, 'rb'))

    cm = np.zeros(shape=(29, 29))

    for i in range(predicted_labels.shape[0]):
        for j in range(predicted_labels.shape[1]):
            if int(predicted_labels[i][j]):
                cm[j] += true_labels[i]
