import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


GOOD_DIR_POINTS = "/Users/doot/Desktop/tensorflow_browser/models/data/points/good_posture"
BAD_DIR_POINTS = "/Users/doot/Desktop/tensorflow_browser/models/data/points/bad_posture"
MODEL_SAVE_DIR = "points_model"
MODEL_PATH = f"{MODEL_SAVE_DIR}/model.h5"
SCALING_PARAMS_PATH = "/Users/doot/Desktop/tensorflow_browser/docs/assets/models/points_model/scaling_params.json"
NUM_EPOCHS = 20
BATCH_SIZE = 32

def load_points_data(good_dir, bad_dir, feature_columns):
    """Load points data from CSV files."""
    good_files = [os.path.join(good_dir, f) for f in os.listdir(good_dir) if f.endswith('.csv')]
    bad_files = [os.path.join(bad_dir, f) for f in os.listdir(bad_dir) if f.endswith('.csv')]

    X_list = []
    y_list = []

    for gf in good_files:
        df = pd.read_csv(gf, header=None, skiprows=1)
        features = df.iloc[:, feature_columns].values
        labels = np.ones(len(features))
        X_list.append(features)
        y_list.append(labels)

    for bf in bad_files:
        df = pd.read_csv(bf, header=None, skiprows=1)
        features = df.iloc[:, feature_columns].values
        labels = np.zeros(len(features))
        X_list.append(features)
        y_list.append(labels)

    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y

def train_points_model(X, y, model_path, scaling_params_path, input_shape):
    """Train and save the points model."""
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    scaling_params = {
        "means": scaler.mean_.tolist(),
        "std_devs": scaler.scale_.tolist()
    }
    os.makedirs(os.path.dirname(scaling_params_path), exist_ok=True)
    with open(scaling_params_path, "w") as f:
        json.dump(scaling_params, f)

    print(f"Scaling parameters saved to {scaling_params_path}")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE
    )

    model.save(model_path)
    print(f"Model saved to {model_path}")

    print(f"Evaluating {model_path} on the test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")


    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, accuracy, (X_test, y_test)

def evaluate_model(model, X_test, y_test):
    """Perform additional evaluation on the test set."""
    print("Detailed Test Set Evaluation:")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    test_accuracy = np.mean(y_pred_classes == y_test)
    print(f"Calculated Test Accuracy: {test_accuracy:.4f}")

    confusion_matrix = tf.math.confusion_matrix(y_test, y_pred_classes).numpy()
    print("Confusion Matrix:")
    print(confusion_matrix)

  
    precision = precision_score(y_test, y_pred_classes, average='binary')
    recall = recall_score(y_test, y_pred_classes, average='binary')
    f1 = f1_score(y_test, y_pred_classes, average='binary')

    print("Metrics for Points Model:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print("\n")

if __name__ == "__main__":
    print("Training Points Model...")

    feature_columns = list(range(1, 16))  
    X_points, y_points = load_points_data(GOOD_DIR_POINTS, BAD_DIR_POINTS, feature_columns)

    model, test_accuracy, test_data = train_points_model(
        X_points, y_points, MODEL_PATH, SCALING_PARAMS_PATH, input_shape=(15,)
    )

    X_test, y_test = test_data
    evaluate_model(model, X_test, y_test)

    print("Converting model to TensorFlow.js format...")
    # os.system(f"tensorflowjs_converter --input_format=keras {MODEL_PATH} {MODEL_SAVE_DIR}")
    print("Model converted and saved to TensorFlow.js format.")