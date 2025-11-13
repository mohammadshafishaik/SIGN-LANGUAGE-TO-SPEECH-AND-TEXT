import os
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pose_extractor.utils import load_keypoints, generate_features, pad_or_truncate_sequence

# --- Constants ---
MAX_SEQ_LENGTH = 100  # Max number of frames per video
FEATURE_TYPE = '3d_velo' # Default feature type
MODEL_PATH = 'models/saved_models/baseline_lstm.h5'

def load_data(keypoints_dir, metadata_path, feature_type, max_len):
    """
    Loads the dataset by linking metadata with keypoint files.

    Args:
        keypoints_dir (str): Directory containing the .npz keypoint files.
        metadata_path (str): Path to the metadata.csv file.
        feature_type (str): The type of features to generate.
        max_len (int): The sequence length to pad/truncate to.

    Returns:
        A tuple (X, y, label_encoder) where X is the feature data, y is the encoded labels,
        and label_encoder is the fitted LabelEncoder instance.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    metadata = pd.read_csv(metadata_path)
    all_features = []
    all_labels = []

    print(f"Loading data from {keypoints_dir}...")
    for _, row in metadata.iterrows():
        video_id = row['video_id']
        phrase = row['phrase']
        npz_path = os.path.join(keypoints_dir, f"{video_id}.npz")

        if not os.path.exists(npz_path):
            print(f"Warning: Keypoint file not found for video_id {video_id}, skipping.")
            continue

        keypoint_data = load_keypoints(npz_path)
        if keypoint_data is None:
            continue
            
        features = generate_features(keypoint_data, feature_type)
        processed_sequence = pad_or_truncate_sequence(features, max_len)
        
        all_features.append(processed_sequence)
        all_labels.append(phrase)

    if not all_features:
        raise ValueError("No feature data was loaded. Check keypoints_dir and metadata.")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(all_labels)
    
    X = np.array(all_features)
    y = tf.keras.utils.to_categorical(y_encoded)

    return X, y, label_encoder

def build_lstm_model(input_shape, num_classes):
    """
    Builds and compiles the baseline LSTM model.

    Args:
        input_shape (tuple): The shape of the input data (max_len, num_features).
        num_classes (int): The number of unique classes (phrases).

    Returns:
        A compiled TensorFlow Keras model.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.5),
        LSTM(128, return_sequences=False),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    return model

def train_model(X_train, y_train, X_val, y_val, input_shape, num_classes):
    """
    Trains the LSTM model and saves the best version.
    """
    model = build_lstm_model(input_shape, num_classes)
    
    # Create directory for saved models if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(filepath=MODEL_PATH, save_best_only=True, monitor='val_loss')
    ]

    print("\n--- Starting Model Training ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks
    )
    print("--- Model Training Complete ---")
    return model, history

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluates the trained model and prints a classification report.
    """
    print("\n--- Evaluating Model ---")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}\n")

    report = classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
    print("Classification Report:")
    print(report)

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the baseline LSTM model.")
    parser.add_argument("--train", action="store_true", help="Flag to train the model.")
    parser.add_argument("--evaluate", action="store_true", help="Flag to evaluate the model.")
    parser.add_argument("--keypoints_dir", type=str, default="dataset/keypoints", help="Directory with .npz files.")
    parser.add_argument("--metadata", type=str, default="dataset/metadata.csv", help="Path to metadata CSV.")
    
    args = parser.parse_args()

    if not args.train and not args.evaluate:
        print("Error: Please specify either --train or --evaluate.")
        return

    # Load data
    try:
        X, y, label_encoder = load_data(args.keypoints_dir, args.metadata, FEATURE_TYPE, MAX_SEQ_LENGTH)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading data: {e}")
        print("Please ensure you have collected data with 'data_collector/collect.py' and extracted keypoints.")
        return

    num_classes = len(label_encoder.classes_)
    input_shape = (X.shape[1], X.shape[2])
    
    print(f"\nData loaded successfully.")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of classes: {num_classes}")
    print(f"Input shape: {input_shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    if args.train:
        train_model(X_train, y_train, X_val, y_val, input_shape, num_classes)
        print(f"\nModel trained and saved to {MODEL_PATH}")

    if args.evaluate:
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file not found at {MODEL_PATH}. Please train the model first using --train.")
            return
        
        model = tf.keras.models.load_model(MODEL_PATH)
        evaluate_model(model, X_test, y_test, label_encoder)

if __name__ == "__main__":
    # Example usage:
    # 1. Train the model:
    #    python models/baseline_lstm.py --train
    # 2. Evaluate the model:
    #    python models/baseline_lstm.py --evaluate
    main()
