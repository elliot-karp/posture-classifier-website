
import numpy as np
import pandas as pd
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

GOOD_DIR = "/Users/doot/Desktop/tensorflow_browser/models/data/angles/good_posture"
BAD_DIR = "/Users/doot/Desktop/tensorflow_browser/models/data/angles/bad_posture"


if __name__ == "__main__":
    scaler = StandardScaler()

    good_files = [os.path.join(GOOD_DIR, f) for f in os.listdir(GOOD_DIR) if f.endswith('.csv')]
    bad_files = [os.path.join(BAD_DIR, f) for f in os.listdir(BAD_DIR) if f.endswith('.csv')]

    X_list = []
    y_list = []

    for gf in good_files:
        df = pd.read_csv(gf, header=None, skiprows=1)
        features = df.iloc[:, 1:4].values  
        labels = np.ones(len(features))
        X_list.append(features)
        y_list.append(labels)

    for bf in bad_files:
        df = pd.read_csv(bf, header=None, skiprows=1)
        features = df.iloc[:, 1:4].values
        labels = np.zeros(len(features))
        X_list.append(features)
        y_list.append(labels)

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    def heuristic_classification(features, debug=False):
        shoulder_tilt, forward_lean, head_tilt = features

        is_bad = False
        if not ((-180 <= shoulder_tilt <= -174) or (174 <= shoulder_tilt <= 180)):
            is_bad = True

        if forward_lean < 120:
            is_bad = True

        if head_tilt < 50 or head_tilt > 70:
            is_bad = True

        return 0 if is_bad else 1

    y_pred = np.array([heuristic_classification(features) for features in X_test])
    accuracy = np.mean(y_pred == y_test)

    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    print(f"Test Accuracy of Heuristic: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    incorrect_predictions = []
    for idx, features in enumerate(X_test):
        label = y_test[idx]
        pred = heuristic_classification(features, debug=False)
        if pred != label:
            incorrect_predictions.append((features, label, pred))

    print(f"Number of Incorrect Predictions: {len(incorrect_predictions)}")