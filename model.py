import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
import json  # Added import for JSON handling
from tensorflow.keras.models import load_model

# Constants
GOOD_DIR = "angles/good_posture"
BAD_DIR = "angles/bad_posture"
MODEL_PATH = "posture_model.h5"
NUM_EPOCHS = 20
BATCH_SIZE = 32

if __name__ == "__main__":
    # Prepare the data
    scaler = StandardScaler()
    
    # Load good posture data
    good_files = [os.path.join(GOOD_DIR, f) for f in os.listdir(GOOD_DIR) if f.endswith('.csv')]
    bad_files = [os.path.join(BAD_DIR, f) for f in os.listdir(BAD_DIR) if f.endswith('.csv')]

    X_list = []
    y_list = []

    # Good posture data (label = 1)
    for gf in good_files:
        df = pd.read_csv(gf, header=None, skiprows=1)
        features = df.iloc[:, 1:4].values  # assuming columns 1,2,3 are features
        labels = np.ones(len(features))
        X_list.append(features)
        y_list.append(labels)

    # Bad posture data (label = 0)
    for bf in bad_files:
        df = pd.read_csv(bf, header=None, skiprows=1)
        features = df.iloc[:, 1:4].values
        labels = np.zeros(len(features))
        X_list.append(features)
        y_list.append(labels)

    # Combine data
    X = np.vstack(X_list)
    y = np.hstack(y_list)

    # Normalize data
    X = scaler.fit_transform(X)

    # Save scaling parameters to JSON
    scaling_params = {
        "means": scaler.mean_.tolist(),
        "std_devs": scaler.scale_.tolist()
    }

    with open("scaling_params.json", "w") as f:
        json.dump(scaling_params, f)

    print("Scaling parameters saved to scaling_params.json")

    # Split data into train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Build a simple Sequential model with explicit input_shape
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(batch_input_shape=(None, 3)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

    # Save the model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Test the model
    print("Evaluating on the test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Load the model (just to verify)
    loaded_model = load_model(MODEL_PATH)
    print("Model loaded from posture_model.h5")
    print(loaded_model.summary())