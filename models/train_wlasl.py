"""
Train LSTM model on WLASL dataset for word-level sign language recognition
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
from pathlib import Path
from project_paths import (
    ROOT_DIR,
    DATASET_DIR,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    ensure_default_dirs,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Paths (portable)
ensure_default_dirs()
KEYPOINTS_DIR = DATASET_DIR / 'keypoints_wlasl'
MODEL_DIR = CHECKPOINTS_DIR

KEYPOINTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Hyperparameters
SEQUENCE_LENGTH = 50  # Number of frames per video
FEATURE_DIM = 144  # Hand + pose features
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

def load_dataset():
    """Load preprocessed keypoint sequences"""
    print("📂 Loading WLASL dataset...")
    
    X = []
    y = []
    labels = []
    
    # Load from keypoints directory
    if not KEYPOINTS_DIR.exists() or not list(KEYPOINTS_DIR.glob('*.npy')):
        print("❌ No keypoints found! Please run preprocessing first.")
        print("   Run: python data_prep/preprocess_wlasl.py")
        return None, None, None
    
    # Get all keypoint files
    keypoint_files = sorted(KEYPOINTS_DIR.glob('*.npy'))
    
    print(f"Found {len(keypoint_files)} keypoint files")
    
    # Extract labels from filenames
    label_set = set()
    for file in keypoint_files:
        label = file.stem.split('_')[0]  # e.g., "all_01912" -> "all"
        label_set.add(label)
    
    labels = sorted(list(label_set))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    print(f"Found {len(labels)} unique words: {labels[:10]}...")
    
    # Load data
    for file in keypoint_files:
        try:
            # Load keypoints
            keypoints = np.load(file)
            
            # Pad or truncate to SEQUENCE_LENGTH
            if len(keypoints) < SEQUENCE_LENGTH:
                # Pad with zeros
                padding = np.zeros((SEQUENCE_LENGTH - len(keypoints), FEATURE_DIM))
                keypoints = np.vstack([keypoints, padding])
            else:
                # Truncate
                keypoints = keypoints[:SEQUENCE_LENGTH]
            
            # Get label
            label = file.stem.split('_')[0]
            label_idx = label_to_idx[label]
            
            X.append(keypoints)
            y.append(label_idx)
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print(f"✓ Loaded {len(X)} samples")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    
    return X, y, labels

def create_model(num_classes):
    """Create LSTM model for temporal sequence recognition"""
    print("\n🏗️ Building LSTM model...")
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM)),
        
        # Masking layer (ignore padded zeros)
        layers.Masking(mask_value=0.0),
        
        # Bidirectional LSTM layers
        layers.Bidirectional(layers.LSTM(256, return_sequences=True)),
        layers.Dropout(0.4),
        
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.3),
        
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.3),
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✓ Model created with {num_classes} output classes")
    model.summary()
    
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """Train the model with callbacks"""
    print("\n🚀 Training model...")
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / 'wlasl_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(LOGS_DIR / 'wlasl'),
            histogram_freq=1
        )
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def plot_history(history):
    """Plot training history"""
    print("\n📊 Plotting training history...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'wlasl_training_history.png', dpi=150)
    print(f"✓ Saved plot to {MODEL_DIR / 'wlasl_training_history.png'}")
    
    plt.show()

def main():
    """Main training pipeline"""
    print("="*70)
    print("🎓 WLASL MODEL TRAINING")
    print("="*70)
    
    # Load dataset
    X, y, labels = load_dataset()
    
    if X is None:
        return
    
    # Save labels
    labels_path = MODEL_DIR / 'wlasl_labels.txt'
    with open(labels_path, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
    print(f"✓ Saved {len(labels)} labels to {labels_path}")
    
    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Dataset split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    
    # Class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\n📈 Training class distribution:")
    for idx, count in zip(unique[:10], counts[:10]):
        print(f"  {labels[idx]}: {count} samples")
    if len(unique) > 10:
        print(f"  ... and {len(unique)-10} more classes")
    
    # Create model
    model = create_model(num_classes=len(labels))
    
    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Save final model
    final_path = MODEL_DIR / 'wlasl_final.keras'
    model.save(final_path)
    print(f"\n✓ Saved final model to {final_path}")
    
    # Plot history
    plot_history(history)
    
    # Final evaluation
    print("\n📊 Final Evaluation:")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"  Training Accuracy: {train_acc*100:.2f}%")
    print(f"  Validation Accuracy: {val_acc*100:.2f}%")
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"📁 Best model: {MODEL_DIR / 'wlasl_best.keras'}")
    print(f"📁 Final model: {final_path}")
    print(f"📁 Labels: {labels_path}")
    print("="*70)

if __name__ == '__main__':
    main()
