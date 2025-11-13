#!/usr/bin/env python3
"""
WLASL 100-word training optimized for 80%+ accuracy in ~1 hour
Uses proven techniques: Temporal Convolutional Network + LSTM + Strong Augmentation
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION - Optimized for 80% accuracy in 1 hour
# ============================================================================
DATASET_DIR = "dataset/wlasl_100_processed"
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"

# Fast training with high accuracy
BATCH_SIZE = 128  # Larger for speed
EPOCHS = 80  # More epochs but faster per epoch
INITIAL_LR = 0.003  # Higher initial LR
MIN_LR = 1e-6

# Model architecture - proven for temporal data
TCN_FILTERS = [128, 128, 64]  # Temporal Convolutional layers
LSTM_UNITS = 128
DENSE_UNITS = 256
DROPOUT = 0.5

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("=" * 70)
print("🚀 WLASL 100-WORD HIGH ACCURACY TRAINING")
print("=" * 70)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Target: 80%+ accuracy in ~1 hour")
print(f"Batch size: {BATCH_SIZE}")
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

# Load and convert
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
# STRONG DATA AUGMENTATION
# ============================================================================
@tf.function
def strong_augment(x):
    """Strong but controlled augmentation for better generalization"""
    # Random Gaussian noise
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.02, dtype=tf.float32)
    x = x + noise
    
    # Random scaling (0.95 to 1.05)
    scale = tf.random.uniform(shape=[], minval=0.95, maxval=1.05, dtype=tf.float32)
    x = x * scale
    
    # Random time shift (shift frames)
    shift = tf.random.uniform(shape=[], minval=-3, maxval=3, dtype=tf.int32)
    x = tf.roll(x, shift=shift, axis=0)
    
    # Random feature dropout (dropout 5% of features randomly)
    dropout_mask = tf.random.uniform(shape=tf.shape(x), dtype=tf.float32)
    dropout_mask = tf.where(dropout_mask > 0.05, 1.0, 0.0)
    x = x * dropout_mask
    
    return x

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
print()

# ============================================================================
# BUILD HIGH-ACCURACY MODEL - TCN + LSTM
# ============================================================================
print("=" * 70)
print("🏗️  BUILDING HIGH-ACCURACY MODEL (TCN + LSTM)")
print("=" * 70)

def build_high_accuracy_model():
    """
    Temporal Convolutional Network + LSTM
    Proven architecture for temporal sequence classification
    """
    inputs = layers.Input(shape=(n_timesteps, n_features), name='input')
    x = inputs
    
    # Temporal Convolutional layers (extract temporal patterns)
    for i, filters in enumerate(TCN_FILTERS):
        x = layers.Conv1D(
            filters=filters,
            kernel_size=3,
            padding='same',
            activation='relu',
            name=f'tcn_{i+1}'
        )(x)
        x = layers.BatchNormalization(name=f'bn_tcn_{i+1}')(x)
        x = layers.Dropout(0.3, name=f'dropout_tcn_{i+1}')(x)
    
    # Bidirectional LSTM (capture long-term dependencies)
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS, return_sequences=False),
        name='bilstm'
    )(x)
    
    x = layers.BatchNormalization(name='bn_lstm')(x)
    x = layers.Dropout(DROPOUT)(x)
    
    # Dense layers
    x = layers.Dense(DENSE_UNITS, activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(DROPOUT)(x)
    
    x = layers.Dense(128, activation='relu', name='dense_2')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output
    outputs = layers.Dense(n_classes, activation='softmax', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='WLASL_HighAccuracy')
    return model

model = build_high_accuracy_model()
model.summary()

print()
print(f"Total parameters: {model.count_params():,}")
print()

# ============================================================================
# COMPILE
# ============================================================================
# Use legacy optimizer for M1/M2 Mac
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

# Learning rate schedule: warmup + cosine decay
def lr_schedule(epoch, lr):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return INITIAL_LR * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (EPOCHS - warmup_epochs)
        return MIN_LR + (INITIAL_LR - MIN_LR) * 0.5 * (1 + np.cos(np.pi * progress))

callbacks = [
    keras.callbacks.ModelCheckpoint(
        f"{CHECKPOINT_DIR}/wlasl_100_best.keras",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=MIN_LR,
        verbose=1
    ),
    keras.callbacks.CSVLogger(
        f"{LOG_DIR}/training_best_{timestamp}.csv"
    )
]

print("✅ Callbacks configured")
print()

# ============================================================================
# CREATE AUGMENTED DATASETS
# ============================================================================
print("=" * 70)
print("🔄 CREATING AUGMENTED DATASETS")
print("=" * 70)

# Training with augmentation
train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y_cat))
train_dataset = train_dataset.map(
    lambda x, y: (strong_augment(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Validation without augmentation
val_dataset = tf.data.Dataset.from_tensor_slices((val_X, val_y_cat))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("✅ Datasets created with strong augmentation")
print()

# ============================================================================
# TRAIN
# ============================================================================
print("=" * 70)
print("🎯 STARTING TRAINING")
print("=" * 70)
print(f"Expected time: ~{EPOCHS * 8 / 60:.0f} minutes")
print()

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
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

# Check if 80% achieved
if test_acc >= 0.80:
    print("🎉" * 35)
    print("✅ TARGET ACHIEVED: 80%+ ACCURACY!")
    print("🎉" * 35)
else:
    print(f"⚠️  Close! Achieved {test_acc*100:.2f}% (target: 80%)")

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

sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)

print("=" * 70)
print("🏆 TOP 10 BEST PERFORMING CLASSES")
print("=" * 70)
for class_id, acc in sorted_classes[:10]:
    word = label_to_word.get(str(class_id), f"Class_{class_id}")
    print(f"{word:20s}: {acc*100:6.2f}%")

print()
print("=" * 70)
print("⚠️  TOP 10 WORST PERFORMING CLASSES")
print("=" * 70)
for class_id, acc in sorted_classes[-10:]:
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
                     for c, a in sorted_classes[:10]],
    'worst_classes': [(label_to_word.get(str(c), f"Class_{c}"), float(a)) 
                      for c, a in sorted_classes[-10:]],
    'timestamp': timestamp,
    'training_time_minutes': EPOCHS * 8 / 60
}

with open(f"{CHECKPOINT_DIR}/wlasl_100_best_results.json", 'w') as f:
    json.dump(results, f, indent=2)

# Plot history
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
plt.axhline(y=0.80, color='r', linestyle='--', label='80% Target')
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{LOG_DIR}/training_best_history.png", dpi=150, bbox_inches='tight')
print()
print(f"✅ Training history plot saved to: {LOG_DIR}/training_best_history.png")

print()
print("=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Best model saved to: {CHECKPOINT_DIR}/wlasl_100_best.keras")
print(f"Results saved to: {CHECKPOINT_DIR}/wlasl_100_best_results.json")
print(f"Final Test Accuracy: {test_acc*100:.2f}%")
print()
