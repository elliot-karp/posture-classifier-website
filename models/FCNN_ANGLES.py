import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
import json   
from tensorflow.keras.models import load_model

# Constants
GOOD_DIR = "/Users/doot/Desktop/tensorflow_browser/models/data/angles/good_posture"
BAD_DIR = "/Users/doot/Desktop/tensorflow_browser/models/data/angles/bad_posture"


MODEL_PATH = "angles_model.h5"
NUM_EPOCHS = 20
BATCH_SIZE = 32

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

 
    X = scaler.fit_transform(X)

 
    scaling_params = {
        "means": scaler.mean_.tolist(),
        "std_devs": scaler.scale_.tolist()
    }

    with open("/Users/doot/Desktop/tensorflow_browser/docs/assets/models/angles_model/scaling_params.json", "w") as f:
        json.dump(scaling_params, f)

    print("Scaling parameters saved to scaling_params.json")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(batch_input_shape=(None, 3)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    print("Evaluating on the test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")

 
    loaded_model = load_model(MODEL_PATH)
    print("Model loaded from posture_model.h5")
    print(loaded_model.summary())

    # run this to convert to tensorflowjs format:
    #tensorflowjs_converter --input_format=keras angles_model.h5 /Users/doot/Desktop/tensorflow_browser/docs/assets/models/angles_model/

    