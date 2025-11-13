#!/usr/bin/env python3
"""
Optimized WLASL 100-Word Sign Language Recognition Training
Goal: HIGH ACCURACY for each class with less overfitting
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, 
    TensorBoard, CSVLogger, LearningRateScheduler
)
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import matplotlib.pyplot as plt

print("=" * 70)
print("🚀 OPTIMIZED WLASL 100-WORD TRAINING")
print("=" * 70)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"TensorFlow version: {tf.__version__}")
print()

# ============================================================================
# CONFIGURATION - Optimized for High Accuracy
# ============================================================================
DATASET_DIR = "dataset/wlasl_100_processed"
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
BATCH_SIZE = 16  # Smaller batch for better generalization
EPOCHS = 60
INITIAL_LR = 0.001
MIN_LR = 1e-6

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("=" * 70)
print("📦 LOADING DATA")
print("=" * 70)

# Load training data
train_data = np.load(f"{DATASET_DIR}/train_data.npz")
train_X = train_data['X']
train_y = train_data['y']

# Load validation data
val_data = np.load(f"{DATASET_DIR}/val_data.npz")
val_X = val_data['X']
val_y = val_data['y']

# Load test data
test_data = np.load(f"{DATASET_DIR}/test_data.npz")
test_X = test_data['X']
test_y = test_data['y']

# Load metadata
with open(f"{DATASET_DIR}/metadata.json", 'r') as f:
    metadata = json.load(f)

# Create label to word mapping
label_to_word = {}
for word_label, idx in metadata['label_to_idx'].items():
    # word_label format is "classid\tword", extract just the word
    word = word_label.split('\t')[1] if '\t' in word_label else word_label
    label_to_word[str(idx)] = word

n_classes = metadata['n_classes']

# Load data and convert to float32
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

# Clean data (handle NaN/Inf)
train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)
val_X = np.nan_to_num(val_X, nan=0.0, posinf=0.0, neginf=0.0)
test_X = np.nan_to_num(test_X, nan=0.0, posinf=0.0, neginf=0.0)

# Convert labels to categorical
train_y_cat = keras.utils.to_categorical(train_y, n_classes)
val_y_cat = keras.utils.to_categorical(val_y, n_classes)
test_y_cat = keras.utils.to_categorical(test_y, n_classes)

# ============================================================================
# COMPUTE CLASS WEIGHTS - Handle class imbalance
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

# Show some class statistics
unique, counts = np.unique(train_y, return_counts=True)
print(f"Min samples per class: {counts.min()}")
print(f"Max samples per class: {counts.max()}")
print(f"Mean samples per class: {counts.mean():.1f}")
print(f"Class weight range: {class_weights_array.min():.3f} - {class_weights_array.max():.3f}")
print()

# ============================================================================
# DATA AUGMENTATION - Subtle, not aggressive
# ============================================================================
def subtle_augment(x):
    """Very subtle augmentation to reduce overfitting without distorting data"""
    # Only add small Gaussian noise (sigma=0.01)
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.01, dtype=tf.float32)
    x_aug = x + noise
    
    # Slight random scaling (0.98 to 1.02)
    scale = tf.random.uniform(shape=[], minval=0.98, maxval=1.02)
    x_aug = x_aug * scale
    
    return x_aug

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y_cat))
train_dataset = train_dataset.shuffle(buffer_size=len(train_X))
train_dataset = train_dataset.map(
    lambda x, y: (subtle_augment(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_X, val_y_cat))
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

print("=" * 70)
print("🏗️  BUILDING OPTIMIZED MODEL")
print("=" * 70)

# ============================================================================
# MODEL ARCHITECTURE - Deeper with better regularization
# ============================================================================
def build_optimized_model(input_shape, n_classes):
    """
    Optimized LSTM model with:
    - Deeper architecture for better feature learning
    - Strong regularization to prevent overfitting
    - Batch normalization for stable training
    - Attention mechanism for temporal focus
    """
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Layer 1: Bidirectional LSTM with L2 regularization
    x = layers.Bidirectional(
        layers.LSTM(
            128, 
            return_sequences=True,
            kernel_regularizer=regularizers.l2(0.001),
            recurrent_regularizer=regularizers.l2(0.001),
            dropout=0.3,
            recurrent_dropout=0.2
        ),
        name='bilstm_1'
    )(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    
    # Layer 2: Bidirectional LSTM
    x = layers.Bidirectional(
        layers.LSTM(
            64,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(0.001),
            recurrent_regularizer=regularizers.l2(0.001),
            dropout=0.3,
            recurrent_dropout=0.2
        ),
        name='bilstm_2'
    )(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    
    # Attention mechanism - focus on important frames
    attention = layers.Dense(1, activation='tanh', name='attention_scores')(x)
    attention = layers.Flatten(name='attention_flatten')(attention)
    attention = layers.Activation('softmax', name='attention_weights')(attention)
    attention = layers.RepeatVector(128, name='attention_repeat')(attention)
    attention = layers.Permute([2, 1], name='attention_permute')(attention)
    
    # Apply attention to LSTM output
    x = layers.Multiply(name='attention_apply')([x, attention])
    
    # Global pooling
    x_avg = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
    x_max = layers.GlobalMaxPooling1D(name='global_max_pool')(x)
    x = layers.Concatenate(name='pool_concat')([x_avg, x_max])
    
    # Dense layers with strong dropout
    x = layers.Dense(
        256, 
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),
        name='dense_1'
    )(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.Dropout(0.5, name='dropout_1')(x)
    
    x = layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),
        name='dense_2'
    )(x)
    x = layers.BatchNormalization(name='bn_4')(x)
    x = layers.Dropout(0.4, name='dropout_2')(x)
    
    # Output layer
    outputs = layers.Dense(n_classes, activation='softmax', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='WLASL_Optimized')
    return model

# Build model
model = build_optimized_model((n_timesteps, n_features), n_classes)

# Model summary
print()
model.summary()
print()
print(f"Total parameters: {model.count_params():,}")
print()

# ============================================================================
# COMPILE MODEL
# ============================================================================
# Use legacy optimizer for M1/M2 Macs (runs much faster)
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
# CALLBACKS - Learning rate schedule with warmup
# ============================================================================
print("=" * 70)
print("⚙️  CONFIGURING CALLBACKS")
print("=" * 70)

# Learning rate schedule with warmup
def lr_schedule(epoch, lr):
    """
    Learning rate schedule:
    - Warmup for first 5 epochs
    - Then cosine decay
    """
    warmup_epochs = 5
    total_epochs = EPOCHS
    
    if epoch < warmup_epochs:
        # Linear warmup
        return INITIAL_LR * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return MIN_LR + (INITIAL_LR - MIN_LR) * 0.5 * (1 + np.cos(np.pi * progress))

# Callbacks
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

callbacks = [
    ModelCheckpoint(
        filepath=f"{CHECKPOINT_DIR}/wlasl_100_optimized.keras",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    LearningRateScheduler(lr_schedule, verbose=0),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=MIN_LR,
        verbose=1
    ),
    TensorBoard(
        log_dir=f"{LOG_DIR}/wlasl100_optimized_{timestamp}",
        histogram_freq=1
    ),
    CSVLogger(
        f"{LOG_DIR}/training_optimized_{timestamp}.csv"
    )
]

print("✅ Callbacks configured:")
print("   - ModelCheckpoint (save best model)")
print("   - EarlyStopping (patience=15)")
print("   - LearningRateScheduler (warmup + cosine decay)")
print("   - ReduceLROnPlateau (factor=0.5, patience=7)")
print("   - TensorBoard logging")
print("   - CSV logging")
print()

# ============================================================================
# TRAIN MODEL
# ============================================================================
print("=" * 70)
print("🎯 STARTING TRAINING")
print("=" * 70)
print()

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

print()
print("=" * 70)
print("✅ TRAINING COMPLETED!")
print("=" * 70)

# ============================================================================
# EVALUATE ON TEST SET
# ============================================================================
print()
print("=" * 70)
print("📊 EVALUATING MODEL ON TEST SET")
print("=" * 70)
print()

print("📦 Loading best model...")
best_model = keras.models.load_model(f"{CHECKPOINT_DIR}/wlasl_100_optimized.keras")
print("   ✅ Loaded: wlasl_100_optimized.keras")
print()

print("🧪 Testing on test set...")
test_results = best_model.evaluate(test_X, test_y_cat, verbose=1, batch_size=BATCH_SIZE)

print()
print("=" * 70)
print("📈 FINAL RESULTS")
print("=" * 70)
test_loss = test_results[0]
test_acc = test_results[1]
test_top5 = test_results[2]
test_top3 = test_results[3]

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test Top-5 Accuracy: {test_top5:.4f} ({test_top5*100:.2f}%)")
print(f"Test Top-3 Accuracy: {test_top3:.4f} ({test_top3*100:.2f}%)")
print()

# ============================================================================
# PER-CLASS ACCURACY ANALYSIS
# ============================================================================
print("=" * 70)
print("📊 PER-CLASS ACCURACY ANALYSIS")
print("=" * 70)
print()

# Get predictions
test_pred = best_model.predict(test_X, batch_size=BATCH_SIZE, verbose=0)
test_pred_classes = np.argmax(test_pred, axis=1)

# Calculate per-class accuracy
per_class_correct = {}
per_class_total = {}

for true_label, pred_label in zip(test_y, test_pred_classes):
    if true_label not in per_class_total:
        per_class_total[true_label] = 0
        per_class_correct[true_label] = 0
    
    per_class_total[true_label] += 1
    if true_label == pred_label:
        per_class_correct[true_label] += 1

# Calculate and display per-class accuracies
per_class_acc = {}
for class_id in sorted(per_class_total.keys()):
    total = per_class_total[class_id]
    correct = per_class_correct[class_id]
    accuracy = correct / total if total > 0 else 0
    per_class_acc[class_id] = accuracy
    
    word = label_to_word[str(class_id)]
    print(f"Class {class_id:3d} ({word:15s}): {accuracy*100:5.1f}% ({correct:3d}/{total:3d})")

print()
print(f"Average per-class accuracy: {np.mean(list(per_class_acc.values()))*100:.2f}%")
print()

# Find best and worst classes
sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
print("🏆 Top 5 Best Performing Classes:")
for class_id, acc in sorted_classes[:5]:
    word = label_to_word[str(class_id)]
    print(f"   {word:15s}: {acc*100:.1f}%")

print()
print("⚠️  Top 5 Worst Performing Classes:")
for class_id, acc in sorted_classes[-5:]:
    word = label_to_word[str(class_id)]
    total = per_class_total[class_id]
    print(f"   {word:15s}: {acc*100:.1f}% ({total} test samples)")

print()

# ============================================================================
# SAVE RESULTS
# ============================================================================
results = {
    'test_loss': float(test_loss),
    'test_accuracy': float(test_acc),
    'test_top5_accuracy': float(test_top5),
    'test_top3_accuracy': float(test_top3),
    'per_class_accuracy': {
        label_to_word[str(k)]: float(v) 
        for k, v in per_class_acc.items()
    },
    'training_history': {
        'epochs': len(history.history['loss']),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'best_val_accuracy': float(max(history.history['val_accuracy']))
    },
    'model_info': {
        'total_parameters': int(best_model.count_params()),
        'n_classes': int(n_classes),
        'input_shape': [int(n_timesteps), int(n_features)]
    }
}

results_file = f"{CHECKPOINT_DIR}/wlasl_100_optimized_results.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"💾 Results saved to: {results_file}")
print()

# ============================================================================
# GENERATE TRAINING PLOTS
# ============================================================================
print("📊 Generating training plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('WLASL 100-Word Training History (Optimized)', fontsize=16, fontweight='bold')

# Plot 1: Accuracy
ax = axes[0, 0]
ax.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
ax.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('Model Accuracy', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# Plot 2: Loss
ax = axes[0, 1]
ax.plot(history.history['loss'], label='Train Loss', linewidth=2)
ax.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Loss', fontsize=11)
ax.set_title('Model Loss', fontsize=12, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Plot 3: Top-5 Accuracy
ax = axes[1, 0]
ax.plot(history.history['top_5_accuracy'], label='Train Top-5', linewidth=2)
ax.plot(history.history['val_top_5_accuracy'], label='Val Top-5', linewidth=2)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Top-5 Accuracy', fontsize=11)
ax.set_title('Top-5 Accuracy', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# Plot 4: Learning Rate
ax = axes[1, 1]
if 'lr' in history.history:
    ax.plot(history.history['lr'], linewidth=2, color='purple')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Learning Rate', fontsize=11)
    ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_file = f"{LOG_DIR}/training_optimized_history.png"
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"   ✅ Saved plot: {plot_file}")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("=" * 70)
print("✅ TRAINING PIPELINE COMPLETE!")
print("=" * 70)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("📁 Outputs:")
print(f"   Model: {CHECKPOINT_DIR}/wlasl_100_optimized.keras")
print(f"   Results: {results_file}")
print(f"   Logs: {LOG_DIR}/")
print()
print("🎯 Key Metrics:")
print(f"   Overall Test Accuracy: {test_acc*100:.2f}%")
print(f"   Top-5 Accuracy: {test_top5*100:.2f}%")
print(f"   Average Per-Class Accuracy: {np.mean(list(per_class_acc.values()))*100:.2f}%")
print()
print("🚀 Next step: Update webapp_simple.py to use wlasl_100_optimized.keras")
print("=" * 70)
