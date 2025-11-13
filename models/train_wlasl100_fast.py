#!/usr/bin/env python3
"""
Fast WLASL 100-word training - optimized for speed on Mac M1/M2
Uses simpler architecture but effective regularization
Target: ~10-15 minutes per epoch instead of 4 hours
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION - Optimized for SPEED
# ============================================================================
DATASET_DIR = "dataset/wlasl_100_processed"
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"

# Fast training settings
BATCH_SIZE = 64  # Larger batch = faster (was 16)
EPOCHS = 40  # Fewer epochs
INITIAL_LR = 0.002  # Higher LR for faster convergence
MIN_LR = 1e-5

# Model size - SMALLER for speed
LSTM_UNITS = 64  # Reduced from 128
DENSE_UNITS = 128  # Reduced from 256
DROPOUT = 0.4

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("=" * 70)
print("🚀 FAST WLASL 100-WORD TRAINING")
print("=" * 70)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Batch size: {BATCH_SIZE} (large for speed)")
print(f"Epochs: {EPOCHS}")
print()

# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 70)
print("📦 LOADING DATA")
print("=" * 70)

train_data = np.load(f"{DATASET_DIR}/train_data.npz")
val_data = np.load(f"{DATASET_DIR}/val_data.npz")
test_data = np.load(f"{DATASET_DIR}/test_data.npz")

with open(f"{DATASET_DIR}/metadata.json", 'r') as f:
    metadata = json.load(f)

# Create label to word mapping
label_to_word = {}
for word_label, idx in metadata['label_to_idx'].items():
    word = word_label.split('\t')[1] if '\t' in word_label else word_label
    label_to_word[str(idx)] = word

n_classes = metadata['n_classes']

# Load and convert to float32
train_X = train_data['X'].astype(np.float32)
train_y = train_data['y'].astype(np.int32)
val_X = val_data['X'].astype(np.float32)
val_y = val_data['y'].astype(np.int32)
test_X = test_data['X'].astype(np.float32)
test_y = test_data['y'].astype(np.int32)

n_timesteps = train_X.shape[1]
n_features = train_X.shape[2]

print("✅ Training samples:", len(train_X))
print("✅ Validation samples:", len(val_X))
print("✅ Test samples:", len(test_X))
print("✅ Classes:", n_classes)
print("✅ Input shape:", train_X.shape[1:])
print()

# Clean data
train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)
val_X = np.nan_to_num(val_X, nan=0.0, posinf=0.0, neginf=0.0)
test_X = np.nan_to_num(test_X, nan=0.0, posinf=0.0, neginf=0.0)

# Convert labels
train_y_cat = keras.utils.to_categorical(train_y, n_classes)
val_y_cat = keras.utils.to_categorical(val_y, n_classes)
test_y_cat = keras.utils.to_categorical(test_y, n_classes)

# ============================================================================
# CLASS WEIGHTS
# ============================================================================
print("=" * 70)
print("⚖️  COMPUTING CLASS WEIGHTS")
print("=" * 70)

class_weights_array = compute_class_weight(
    'balanced',
    classes=np.unique(train_y),
    y=train_y
)
class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

unique, counts = np.unique(train_y, return_counts=True)
print(f"Min samples per class: {counts.min()}")
print(f"Max samples per class: {counts.max()}")
print(f"Mean samples per class: {counts.mean():.1f}")
print(f"Class weight range: {class_weights_array.min():.3f} - {class_weights_array.max():.3f}")
print()

# ============================================================================
# BUILD FAST MODEL - Simple but effective
# ============================================================================
print("=" * 70)
print("🏗️  BUILDING FAST MODEL")
print("=" * 70)

def build_fast_model():
    """Simple LSTM model optimized for speed"""
    inputs = layers.Input(shape=(n_timesteps, n_features), name='input')
    
    # Single Bidirectional LSTM (faster than 2 layers)
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS, return_sequences=False),  # return_sequences=False is faster
        name='bilstm'
    )(inputs)
    
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(DROPOUT)(x)
    
    # Dense layers
    x = layers.Dense(DENSE_UNITS, activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Dropout(DROPOUT)(x)
    
    # Output
    outputs = layers.Dense(n_classes, activation='softmax', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='WLASL_Fast')
    return model

model = build_fast_model()
model.summary()

print()
print(f"Total parameters: {model.count_params():,}")
print()

# ============================================================================
# COMPILE - Use legacy Adam for M1/M2
# ============================================================================
optimizer = keras.optimizers.legacy.Adam(learning_rate=INITIAL_LR)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy'),
        keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
    ]
)

print("✅ Model compiled successfully!")
print()

# ============================================================================
# CALLBACKS
# ============================================================================
print("=" * 70)
print("⚙️  CONFIGURING CALLBACKS")
print("=" * 70)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

callbacks = [
    keras.callbacks.ModelCheckpoint(
        f"{CHECKPOINT_DIR}/wlasl_100_fast.keras",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,  # Reduced from 15
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,  # Reduced from 7
        min_lr=MIN_LR,
        verbose=1
    ),
    keras.callbacks.CSVLogger(
        f"{LOG_DIR}/training_fast_{timestamp}.csv"
    )
]

print("✅ Callbacks configured")
print()

# ============================================================================
# TRAIN
# ============================================================================
print("=" * 70)
print("🎯 STARTING TRAINING")
print("=" * 70)
print(f"Estimated time per epoch: 10-15 minutes")
print(f"Total estimated time: {EPOCHS * 12 / 60:.1f} hours")
print()

history = model.fit(
    train_X, train_y_cat,
    validation_data=(val_X, val_y_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ============================================================================
# EVALUATE
# ============================================================================
print()
print("=" * 70)
print("📊 EVALUATING ON TEST SET")
print("=" * 70)

test_results = model.evaluate(test_X, test_y_cat, verbose=0)
test_loss = test_results[0]
test_acc = test_results[1]
test_top5 = test_results[2]
test_top3 = test_results[3]

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test Top-5 Accuracy: {test_top5:.4f} ({test_top5*100:.2f}%)")
print(f"Test Top-3 Accuracy: {test_top3:.4f} ({test_top3*100:.2f}%)")
print()

# Per-class accuracy
predictions = model.predict(test_X, verbose=0)
pred_classes = np.argmax(predictions, axis=1)

per_class_acc = {}
for class_id in range(n_classes):
    mask = test_y == class_id
    if mask.sum() > 0:
        class_acc = (pred_classes[mask] == test_y[mask]).mean()
        per_class_acc[class_id] = class_acc

# Top 5 best and worst
sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)

print("=" * 70)
print("🏆 TOP 5 BEST PERFORMING CLASSES")
print("=" * 70)
for class_id, acc in sorted_classes[:5]:
    word = label_to_word.get(str(class_id), f"Class_{class_id}")
    print(f"{word:20s}: {acc*100:6.2f}%")

print()
print("=" * 70)
print("⚠️  TOP 5 WORST PERFORMING CLASSES")
print("=" * 70)
for class_id, acc in sorted_classes[-5:]:
    word = label_to_word.get(str(class_id), f"Class_{class_id}")
    print(f"{word:20s}: {acc*100:6.2f}%")

# Save results
results = {
    'test_loss': float(test_loss),
    'test_accuracy': float(test_acc),
    'test_top5_accuracy': float(test_top5),
    'test_top3_accuracy': float(test_top3),
    'per_class_accuracy': {label_to_word.get(str(k), f"Class_{k}"): float(v) 
                          for k, v in per_class_acc.items()},
    'best_classes': [(label_to_word.get(str(c), f"Class_{c}"), float(a)) 
                     for c, a in sorted_classes[:5]],
    'worst_classes': [(label_to_word.get(str(c), f"Class_{c}"), float(a)) 
                      for c, a in sorted_classes[-5:]],
    'timestamp': timestamp
}

with open(f"{CHECKPOINT_DIR}/wlasl_100_fast_results.json", 'w') as f:
    json.dump(results, f, indent=2)

# Plot history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f"{LOG_DIR}/training_fast_history.png", dpi=150)
print()
print(f"✅ Training history plot saved to: {LOG_DIR}/training_fast_history.png")

print()
print("=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Best model saved to: {CHECKPOINT_DIR}/wlasl_100_fast.keras")
print(f"Results saved to: {CHECKPOINT_DIR}/wlasl_100_fast_results.json")
print()
