"""
ISL (Indian Sign Language) Model Training
Train classification model on ISL fingerspelling (A-Z, 1-9)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import classification_report, confusion_matrix
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU enabled: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(f"GPU config error: {e}")

def load_data(base_dir):
    """Load train/val/test splits"""
    print("\n📂 Loading dataset splits...")
    
    train_data = np.load(f"{base_dir}/train.npz")
    val_data = np.load(f"{base_dir}/val.npz")
    test_data = np.load(f"{base_dir}/test.npz")
    
    with open(f"{base_dir}/label_mappings.json", 'r') as f:
        label_info = json.load(f)
    
    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = val_data['features']
    y_val = val_data['labels']
    X_test = test_data['features']
    y_test = test_data['labels']
    
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    print(f"  Classes: {len(label_info['label_to_idx'])}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), label_info

def create_model(input_shape, num_classes):
    """
    Create ISL classification model
    
    Architecture:
    - Input normalization
    - Dense layers with BatchNorm + Dropout
    - L2 regularization for stability
    """
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Normalize inputs
        layers.BatchNormalization(),
        
        # First dense block
        layers.Dense(512, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        
        # Second dense block
        layers.Dense(256, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        # Third dense block
        layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ], name='ISL_Classifier')
    
    return model

def plot_training_history(history, save_path):
    """Plot training metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Training plot saved: {save_path}")

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('ISL Classification - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Confusion matrix saved: {save_path}")

def main():
    print("\n" + "="*60)
    print("ISL MODEL TRAINING")
    print("="*60)
    
    # Paths
    base_dir = Path(__file__).parent.parent / "dataset" / "splits_isl"
    checkpoints_dir = Path(__file__).parent.parent / "checkpoints"
    logs_dir = Path(__file__).parent.parent / "logs"
    checkpoints_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), label_info = load_data(base_dir)
    
    num_classes = len(label_info['label_to_idx'])
    input_shape = (X_train.shape[1],)  # 144D
    
    # Create model
    print("\n🏗️  Building model...")
    model = create_model(input_shape, num_classes)
    model.summary()
    
    # Compile (use legacy Adam for M1/M4 Macs)
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(checkpoints_dir / 'isl_best.keras'),
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
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(logs_dir / 'isl_training'),
            histogram_freq=1
        )
    ]
    
    # Train
    print("\n🚀 Training model...")
    print(f"Batch size: 32")
    print(f"Max epochs: 100")
    print(f"Early stopping patience: 15")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history, logs_dir / 'isl_training_history.png')
    
    # Evaluate
    print("\n📊 Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Predictions
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    # Classification report
    idx_to_label = label_info['idx_to_label']
    class_names = [idx_to_label[str(i)] for i in range(num_classes)]
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, class_names, logs_dir / 'isl_confusion_matrix.png')
    
    # Save final model
    final_model_path = checkpoints_dir / 'isl_final.keras'
    model.save(final_model_path)
    print(f"\n💾 Final model saved: {final_model_path}")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Best model: {checkpoints_dir / 'isl_best.keras'}")
    print(f"Final model: {final_model_path}")
    print(f"Training plots: {logs_dir}/")
    print("="*60)

if __name__ == "__main__":
    main()
