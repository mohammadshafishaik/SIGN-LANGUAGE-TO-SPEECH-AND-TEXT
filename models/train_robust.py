"""
ROBUST Training Script for Sign Language Recognition
Features:
- Batch Normalization for numerical stability
- Lower learning rate (1e-6)
- Gradient clipping
- Data clipping to prevent extreme values
- Smaller model appropriate for dataset size
- L2 regularization

Usage:
    python models/train_robust.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import json

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ROBUST SIGN LANGUAGE TRAINING")
print("="*70)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print("="*70)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_preprocess_data(metadata_path, keypoints_dir, max_sequence_length=150):
    """Load keypoints with robust preprocessing."""
    
    df = pd.read_csv(metadata_path)
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Train: {len(df[df['split']=='train'])}")
    print(f"  Val: {len(df[df['split']=='val'])}")
    print(f"  Test: {len(df[df['split']=='test'])}")
    print(f"  Unique labels: {df['label'].nunique()}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    num_classes = len(label_encoder.classes_)
    
    print(f"\nLabel mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {i}: {label}")
    
    # Load sequences
    def load_sequences(split_df):
        sequences = []
        labels = []
        
        for _, row in split_df.iterrows():
            # Load keypoints
            kp_path = Path(row['keypoints_path'])
            data = np.load(kp_path)
            
            # Clip extreme values to prevent numerical instability
            data = np.clip(data, -10, 10)
            
            # Pad or truncate to max_sequence_length
            if len(data) < max_sequence_length:
                # Pad with zeros
                padding = np.zeros((max_sequence_length - len(data), data.shape[1]))
                data = np.vstack([data, padding])
            else:
                # Truncate
                data = data[:max_sequence_length]
            
            sequences.append(data)
            labels.append(row['label_encoded'])
        
        return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int32)
    
    # Load splits
    train_x, train_y = load_sequences(df[df['split']=='train'])
    val_x, val_y = load_sequences(df[df['split']=='val'])
    test_x, test_y = load_sequences(df[df['split']=='test'])
    
    print(f"\n🎯 Data shapes:")
    print(f"  Train: {train_x.shape}, Labels: {train_y.shape}")
    print(f"  Val:   {val_x.shape}, Labels: {val_y.shape}")
    print(f"  Test:  {test_x.shape}, Labels: {test_y.shape}")
    
    # Check for NaN
    if np.isnan(train_x).any():
        print("⚠️  WARNING: NaN values found in training data!")
        train_x = np.nan_to_num(train_x, 0)
    
    # Standardize per-feature across training set
    mean = np.mean(train_x, axis=(0, 1), keepdims=True)
    std = np.std(train_x, axis=(0, 1), keepdims=True) + 1e-8  # Avoid division by zero
    
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    test_x = (test_x - mean) / std
    
    # Clip after standardization
    train_x = np.clip(train_x, -5, 5)
    val_x = np.clip(val_x, -5, 5)
    test_x = np.clip(test_x, -5, 5)
    
    print(f"\n📈 Data statistics after standardization:")
    print(f"  Train mean: {train_x.mean():.6f}, std: {train_x.std():.6f}")
    print(f"  Train min: {train_x.min():.6f}, max: {train_x.max():.6f}")
    
    return {
        'train': (train_x, train_y),
        'val': (val_x, val_y),
        'test': (test_x, test_y),
        'label_encoder': label_encoder,
        'num_classes': num_classes
    }

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def create_robust_model(input_shape, num_classes):
    """
    Create a smaller, more robust model with:
    - Batch Normalization
    - L2 Regularization
    - Dropout
    - Appropriate for small datasets
    """
    
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Masking for variable-length sequences (padded with zeros)
    x = layers.Masking(mask_value=0.0)(inputs)
    
    # LSTM Encoder (smaller than before)
    x = layers.LSTM(
        128, 
        return_sequences=True,
        kernel_regularizer=keras.regularizers.l2(0.01),
        name='lstm1'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.LSTM(
        64,
        return_sequences=False,
        kernel_regularizer=keras.regularizers.l2(0.01),
        name='lstm2'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Dense layers
    x = layers.Dense(
        32,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.01),
        name='dense1'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='robust_sign_language_model')
    
    return model

# ============================================================================
# TRAINING
# ============================================================================

def main():
    # Paths
    metadata_path = 'dataset/metadata_split.csv'
    keypoints_dir = 'dataset/keypoints_wlasl/'
    output_dir = Path('models/saved_models/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_and_preprocess_data(metadata_path, keypoints_dir, max_sequence_length=150)
    
    train_x, train_y = data['train']
    val_x, val_y = data['val']
    test_x, test_y = data['test']
    num_classes = data['num_classes']
    label_encoder = data['label_encoder']
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_y),
        y=train_y
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    print(f"\n⚖️  Class weights: {class_weight_dict}")
    
    # Build model
    print("\n🏗️  Building model...")
    input_shape = (train_x.shape[1], train_x.shape[2])  # (seq_len, features)
    model = create_robust_model(input_shape, num_classes)
    
    # Compile with VERY LOW learning rate for stability
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            filename='logs/training_history.csv',
            append=False
        )
    ]
    
    # Training
    print("\n🚀 Starting training...")
    print(f"   Learning rate: 1e-5")
    print(f"   Batch size: 8")
    print(f"   Max epochs: 200")
    print("="*70)
    
    history = model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=200,
        batch_size=8,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    # Load best model
    best_model = keras.models.load_model(str(output_dir / 'best_model.h5'))
    
    # Test set evaluation
    test_loss, test_acc = best_model.evaluate(test_x, test_y, verbose=0)
    print(f"✅ Test Loss: {test_loss:.4f}")
    print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
    
    # Predictions
    test_preds = best_model.predict(test_x, verbose=0)
    test_pred_classes = np.argmax(test_preds, axis=1)
    
    print(f"\nSample predictions (first 10):")
    for i in range(min(10, len(test_y))):
        true_label = label_encoder.classes_[test_y[i]]
        pred_label = label_encoder.classes_[test_pred_classes[i]]
        confidence = test_preds[i][test_pred_classes[i]] * 100
        status = "✅" if test_y[i] == test_pred_classes[i] else "❌"
        print(f"  {status} True: {true_label:12s} | Pred: {pred_label:12s} ({confidence:.1f}%)")
    
    # Save label encoder
    label_map = {int(i): label for i, label in enumerate(label_encoder.classes_)}
    with open(output_dir / 'label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)
    
    print(f"\n💾 Model saved to: {output_dir / 'best_model.h5'}")
    print(f"💾 Label map saved to: {output_dir / 'label_map.json'}")
    print("="*70)

if __name__ == "__main__":
    main()
