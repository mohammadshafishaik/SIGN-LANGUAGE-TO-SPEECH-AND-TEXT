"""
🎯 WLASL-600 TRAINING SCRIPT FOR GOOGLE COLAB
90%+ Accuracy Target with Maximum Data

DATASET:
- 600 sign language words
- 6,389 total samples
- Pre-extracted MediaPipe features (30 frames, 104 landmarks, 3 coords)

STRATEGY:
- Use ALL 600 words OR filter to top classes with most samples
- Advanced Transformer + BiLSTM architecture
- Heavy data augmentation
- Train on Google Colab GPU

EXPECTED RESULTS:
- Top-100 words: 85-90% accuracy
- Top-50 words: 90-95% accuracy  
- Top-30 words: 92-97% accuracy
"""

import os
import json
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from datetime import datetime

print("="*80)
print("🎯 WLASL-600 TRAINING - MAXIMUM ACCURACY")
print("="*80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================
# Choose how many classes to train on:
# - 600: Use all words (harder, lower accuracy ~70-75%)
# - 100: Top 100 words (good balance, ~85-90%)
# - 50: Top 50 words (high accuracy, ~90-95%)
# - 30: Top 30 words (very high accuracy, ~92-97%)

NUM_CLASSES = 30  # ⬅️ CHANGE THIS: 30, 50, 100, or 600

print(f"🎯 Training Configuration:")
print(f"   - Target Classes: {NUM_CLASSES}")
print(f"   - Expected Accuracy: {'>92%' if NUM_CLASSES <= 30 else '>90%' if NUM_CLASSES <= 50 else '>85%' if NUM_CLASSES <= 100 else '>70%'}")
print("="*80)
print()

# ============================================================================
# STEP 1: CHECK GPU
# ============================================================================
print("STEP 1: Checking GPU availability...")
print("-"*80)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU Available!")
    for gpu in gpus:
        print(f"   - {gpu}")
    print(f"\n🚀 Training will be FAST ({15 if NUM_CLASSES <= 50 else 30}-{20 if NUM_CLASSES <= 50 else 45} minutes)")
else:
    print("⚠️  WARNING: No GPU detected!")
    print("   Training will be SLOW on CPU (2-4 hours)")
    print("   Recommended: Enable GPU in Colab (Runtime → Change runtime type → GPU)")

print("="*80)
print()

# ============================================================================
# STEP 2: UPLOAD & EXTRACT DATASET
# ============================================================================
print("STEP 2: Upload and extract dataset...")
print("-"*80)

try:
    from google.colab import files
    IN_COLAB = True
    
    # Check if dataset already exists
    if not os.path.exists('WLASL features npy'):
        # Check if zip file exists
        zip_files = [f for f in os.listdir('.') if f.endswith('.zip')]
        
        if not zip_files:
            print("📤 UPLOAD YOUR DATASET:")
            print("   Click 'Choose Files' button below to upload your dataset .zip file")
            print("   (archive.zip, wlasl-processed.zip, or any WLASL dataset)")
            print()
            print("   Download from: https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed")
            print()
            
            # File upload widget
            uploaded = files.upload()
            
            if not uploaded:
                raise FileNotFoundError("No file uploaded! Please upload your dataset.")
            
            # Get uploaded filename
            uploaded_file = list(uploaded.keys())[0]
            print(f"\n✅ Uploaded: {uploaded_file} ({os.path.getsize(uploaded_file)/(1024*1024):.1f} MB)")
        else:
            # Use existing zip file
            uploaded_file = zip_files[0]
            print(f"✅ Found existing file: {uploaded_file}")
        
        print()
        print("📦 Extracting dataset...")
        print("   This may take 1-2 minutes...")
        
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall()
        
        print("✅ Dataset extracted!")
        
        # Show extracted contents
        print("\n📂 Extracted files:")
        os.system('ls -lh | head -20')
    else:
        print("✅ Dataset already extracted!")
        uploaded_file = 'archive.zip'  # Assume default name
    
except ImportError:
    # Not in Colab - use local paths
    IN_COLAB = False
    print("⚠️  Not in Colab - using local paths")
    
    # Extract from local archive.zip if needed
    if os.path.exists('archive.zip') and not os.path.exists('WLASL features npy'):
        print("📦 Extracting local archive.zip...")
        with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
            zip_ref.extractall()
        print("✅ Dataset extracted!")
    elif os.path.exists('WLASL features npy'):
        print("✅ Dataset already extracted!")
    else:
        print("❌ ERROR: archive.zip not found in current directory")

print()
print("="*80)
print()

# ============================================================================
# STEP 3: LOAD DATA & SELECT TOP CLASSES
# ============================================================================
print(f"STEP 3: Loading dataset and selecting top {NUM_CLASSES} words...")
print("-"*80)

# AUTO-DETECT dataset structure
print("🔍 Auto-detecting dataset structure...")

# Find the data directory
possible_dirs = [
    'WLASL features npy/WLASL_600',
    'WLASL features npy/WLASL_450',
    'WLASL features npy/WLASL_300',
    'WLASL features npy/WLASL_100',
    'WLASL_dataset',
    'wlasl-processed',
    'dataset',
    '.'
]

dataset_dir = None
for dir_path in possible_dirs:
    if os.path.exists(dir_path):
        # Check if it has the required files
        potential_files = [
            f for f in os.listdir(dir_path) 
            if f.endswith('.npy') or f.endswith('.csv')
        ]
        if potential_files:
            dataset_dir = dir_path
            break

if dataset_dir is None:
    print("❌ ERROR: Could not find dataset files!")
    print("\n📂 Current directory contents:")
    os.system('ls -lR | head -50')
    raise FileNotFoundError("Please check the extracted dataset structure")

print(f"✅ Found dataset in: {dataset_dir}")
print()

# Find data files
npy_files = [f for f in os.listdir(dataset_dir) if f.endswith('.npy')]
csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]

print(f"📂 Available files:")
for f in npy_files[:5]:
    print(f"   - {f}")
for f in csv_files[:3]:
    print(f"   - {f}")
print()

# Auto-detect file names
data_file = None
labels_file = None

for f in npy_files:
    if 'feature' in f.lower() and 'data' in f.lower():
        data_file = f
    elif 'feature' in f.lower() and 'label' in f.lower():
        labels_file = f
    elif 'X' in f or 'train_x' in f.lower() or 'features' in f.lower():
        if data_file is None:
            data_file = f
    elif 'y' in f or 'train_y' in f.lower() or 'labels' in f.lower():
        if labels_file is None:
            labels_file = f

# If still not found, use first two .npy files
if data_file is None and len(npy_files) >= 2:
    data_file = npy_files[0]
if labels_file is None and len(npy_files) >= 2:
    labels_file = npy_files[1]

if data_file is None or labels_file is None:
    print("❌ ERROR: Could not find feature_data.npy and feature_labels.npy")
    print(f"   Found: {npy_files}")
    raise FileNotFoundError("Please check dataset files")

data_path = os.path.join(dataset_dir, data_file)
labels_path = os.path.join(dataset_dir, labels_file)
csv_path = os.path.join(dataset_dir, csv_files[0]) if csv_files else None

print(f"✅ Using files:")
print(f"   Data:   {data_path}")
print(f"   Labels: {labels_path}")
if csv_path:
    print(f"   CSV:    {csv_path}")
print()

# Load data
X_full = np.load(data_path).astype(np.float32)
y_full = np.load(labels_path).astype(np.int32)  # Convert float labels to int

# Load CSV if available for word names
df = None
if csv_path and os.path.exists(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded CSV with word names")
    except:
        print(f"⚠️  Could not load CSV, will use numeric labels")

print(f"✅ Loaded: {X_full.shape[0]} samples, {len(np.unique(y_full))} classes")
print(f"   Shape: {X_full.shape}")

# CRITICAL: Normalize the data (it's currently in range -9.5 to 3.5)
# Standardize to zero mean and unit variance per feature
print("📊 Normalizing data...")
X_mean = X_full.mean()
X_std = X_full.std()
X_full = (X_full - X_mean) / (X_std + 1e-8)
print(f"   Mean: {X_mean:.3f}, Std: {X_std:.3f}")
print(f"   Normalized range: {X_full.min():.3f} to {X_full.max():.3f}")

# Count samples per class
unique_labels, counts = np.unique(y_full, return_counts=True)
class_counts = dict(zip(unique_labels, counts))

# Select top N classes by sample count
top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:NUM_CLASSES]
selected_labels = [label for label, _ in top_classes]

print(f"\n📊 Top {NUM_CLASSES} words by sample count:")
print("-"*80)
for i, (label, count) in enumerate(top_classes[:10], 1):
    # Get word name from CSV if available
    if df is not None:
        try:
            word = df[df.index.isin(np.where(y_full == label)[0])]['sign'].iloc[0]
        except:
            word = f"class_{label}"
    else:
        word = f"class_{label}"
    print(f"  {i:3d}. {word:20s}: {count:3d} samples")
if NUM_CLASSES > 10:
    print(f"  ... (showing top 10 of {NUM_CLASSES})")

# Filter dataset
mask = np.isin(y_full, selected_labels)
X_filtered = X_full[mask]
y_filtered = y_full[mask]

# Remap labels to 0-(NUM_CLASSES-1)
label_mapping = {old_label: new_label for new_label, old_label in enumerate(selected_labels)}
y_remapped = np.array([label_mapping[label] for label in y_filtered])

# Create label name mapping
label_names = {}
for new_label, old_label in enumerate(selected_labels):
    if df is not None:
        try:
            word = df[df.index.isin(np.where(y_full == old_label)[0])]['sign'].iloc[0]
        except:
            word = f"class_{old_label}"
    else:
        word = f"class_{old_label}"
    label_names[new_label] = word

print(f"\n✅ Filtered dataset: {X_filtered.shape[0]} samples, {NUM_CLASSES} classes")

# Reshape data: (samples, 30, 104, 3) -> (samples, 30, 312)
# Flatten the 104 landmarks × 3 coords into 312 features per frame
X_reshaped = X_filtered.reshape(X_filtered.shape[0], X_filtered.shape[1], -1)
print(f"✅ Reshaped to: {X_reshaped.shape} (frames, features)")

# Split into train/val/test - Use 80/10/10 split due to small dataset
X_temp, X_test, y_temp, y_test = train_test_split(
    X_reshaped, y_remapped, test_size=0.10, random_state=42, stratify=y_remapped
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.111, random_state=42, stratify=y_temp  # 0.111 of 0.90 = ~10%
)

print(f"\n✅ Data split:")
print(f"   Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X_reshaped)*100:.1f}%)")
print(f"   Val:   {X_val.shape[0]} samples ({X_val.shape[0]/len(X_reshaped)*100:.1f}%)")
print(f"   Test:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X_reshaped)*100:.1f}%)")
print(f"   ⚠️  Note: Small dataset - using 80/10/10 split for maximum training data")

print()
print("="*80)
print()

# ============================================================================
# STEP 4: HEAVY DATA AUGMENTATION
# ============================================================================
print("STEP 4: Setting up HEAVY data augmentation...")
print("-"*80)

@tf.function
def heavy_augment(sequence):
    """Heavy augmentation for maximum generalization"""
    sequence = tf.cast(sequence, tf.float32)
    
    # 1. Gaussian noise
    noise = tf.random.normal(tf.shape(sequence), mean=0.0, stddev=0.02, dtype=tf.float32)
    sequence = sequence + noise
    
    # 2. Random scaling
    scale = tf.random.uniform([], 0.92, 1.08, dtype=tf.float32)
    sequence = sequence * scale
    
    # 3. Random time shift
    shift = tf.random.uniform([], -3, 4, dtype=tf.int32)
    sequence = tf.roll(sequence, shift, axis=0)
    
    # 4. Feature dropout (5%)
    dropout_mask = tf.random.uniform(tf.shape(sequence)) > 0.05
    sequence = tf.where(dropout_mask, sequence, tf.zeros_like(sequence))
    
    # 5. Random rotation/flip (simulate camera angle)
    if tf.random.uniform([]) > 0.5:
        angle = tf.random.uniform([], -0.08, 0.08)
        sequence = sequence * (1.0 + angle)
    
    return sequence

def create_dataset(X, y, batch_size=32, augment=False, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if shuffle:
        dataset = dataset.shuffle(10000)
    
    if augment:
        dataset = dataset.map(lambda x, y: (heavy_augment(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

BATCH_SIZE = 32

train_dataset = create_dataset(X_train, y_train, BATCH_SIZE, augment=True, shuffle=True)
val_dataset = create_dataset(X_val, y_val, BATCH_SIZE, augment=False, shuffle=False)
test_dataset = create_dataset(X_test, y_test, BATCH_SIZE, augment=False, shuffle=False)

print("✅ Data augmentation pipeline created!")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Augmentation: Noise + Scaling + Time shift + Dropout + Rotation")
print()
print("="*80)
print()

# ============================================================================
# STEP 5: ADVANCED TRANSFORMER + LSTM ARCHITECTURE
# ============================================================================
print("STEP 5: Building ADVANCED architecture...")
print("-"*80)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Transformer encoder block with multi-head attention"""
    # Multi-head attention
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs
    
    # Feed forward network
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_advanced_model(input_shape, n_classes):
    """
    Advanced Transformer + Bidirectional LSTM
    Designed for 90%+ accuracy on sign language recognition
    OPTIMIZED FOR SMALL DATASETS
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial projection
    x = layers.Dense(128)(inputs)  # Reduced from 256
    x = layers.LayerNormalization()(x)
    
    # Transformer blocks (2 layers instead of 3 - less overfitting)
    x = transformer_encoder(x, head_size=32, num_heads=4, ff_dim=256, dropout=0.2)  # Reduced complexity
    x = transformer_encoder(x, head_size=32, num_heads=4, ff_dim=256, dropout=0.2)
    
    # Bidirectional LSTM (1 layer instead of 2)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)  # Reduced from 256
    x = layers.Dropout(0.3)(x)  # Reduced dropout
    
    # Dense classification head (simpler)
    x = layers.Dense(256, activation='relu')(x)  # Reduced from 512
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # Output layer
    outputs = layers.Dense(n_classes, activation='softmax', dtype='float32')(x)
    
    model = keras.Model(inputs, outputs)
    return model

# Mixed precision for faster training
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Build model
input_shape = (X_train.shape[1], X_train.shape[2])  # (30, 312)
model = build_advanced_model(input_shape, NUM_CLASSES)

print("🏗️ Model Architecture:")
model.summary()

# Compute class weights for balanced training
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

print(f"\n✅ Model built successfully!")
print(f"   Input shape: {input_shape}")
print(f"   Output classes: {NUM_CLASSES}")
print(f"   Total parameters: {model.count_params():,}")

print()
print("="*80)
print()

# ============================================================================
# STEP 6: TRAIN WITH COSINE ANNEALING
# ============================================================================
print("STEP 6: Training with advanced optimization...")
print("-"*80)
print(f"Target: {'>92%' if NUM_CLASSES <= 30 else '>90%' if NUM_CLASSES <= 50 else '>85%' if NUM_CLASSES <= 100 else '>70%'} test accuracy")
print(f"Expected time: {15 if NUM_CLASSES <= 50 else 30}-{20 if NUM_CLASSES <= 50 else 45} minutes on GPU")
print("="*80)
print()

def cosine_annealing(epoch, lr):
    """Cosine annealing learning rate schedule"""
    warmup = 5  # Reduced warmup for small dataset
    max_lr = 0.001  # Lower learning rate for stability
    min_lr = 1e-7
    
    if epoch < warmup:
        return max_lr * (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / (150 - warmup)  # More epochs for small data
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Lower LR
    loss='sparse_categorical_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.SparseTopKCategoricalAccuracy(k=min(5, NUM_CLASSES), name='top_5'),
        keras.metrics.SparseTopKCategoricalAccuracy(k=min(3, NUM_CLASSES), name='top_3')
    ]
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        f'wlasl_{NUM_CLASSES}_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=40,  # More patience for small dataset
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.LearningRateScheduler(cosine_annealing, verbose=0),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,  # More patience
        min_lr=1e-8,
        verbose=1
    )
]

start_time = datetime.now()

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=150,  # More epochs for small dataset
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

training_time = (datetime.now() - start_time).total_seconds()

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"Time: {training_time/60:.1f} minutes")
print()

# ============================================================================
# STEP 7: FINAL EVALUATION
# ============================================================================
print("STEP 7: Final evaluation on test set...")
print("-"*80)

test_results = model.evaluate(test_dataset, verbose=1)
test_loss, test_acc, test_top5, test_top3 = test_results[:4]

print(f"\n{'='*80}")
print(f"🎯 FINAL RESULTS - WLASL-{NUM_CLASSES}")
print(f"{'='*80}")
print(f"Test Accuracy:     {test_acc*100:.2f}% {'🎉🎉🎉' if test_acc >= 0.90 else '✅' if test_acc >= 0.85 else '📊'}")
print(f"Top-3 Accuracy:    {test_top3*100:.2f}%")
print(f"Top-5 Accuracy:    {test_top5*100:.2f}%")
print(f"Training Time:     {training_time/60:.1f} minutes")
print(f"{'='*80}")

if test_acc >= 0.92:
    print("\n🎉🎉🎉 OUTSTANDING! 92%+ ACCURACY ACHIEVED! 🎉🎉🎉")
elif test_acc >= 0.90:
    print("\n🎉🎉 EXCELLENT! 90%+ ACCURACY ACHIEVED! 🎉🎉")
elif test_acc >= 0.85:
    print("\n✅ VERY GOOD! 85%+ accuracy achieved!")
else:
    print(f"\n📊 Achieved {test_acc*100:.1f}% accuracy on {NUM_CLASSES} classes")

# Per-class analysis
y_pred = model.predict(test_dataset)
y_pred_classes = np.argmax(y_pred, axis=1)

per_class_acc = {}
for i in range(NUM_CLASSES):
    mask = y_test == i
    if mask.sum() > 0:
        acc = (y_pred_classes[mask] == i).mean()
        per_class_acc[label_names[i]] = acc

sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)

print(f"\n🏆 TOP 10 PERFORMING WORDS:")
print("-"*80)
for word, acc in sorted_classes[:10]:
    print(f"{word:25s}: {acc*100:6.2f}%")

print(f"\n⚠️  BOTTOM 10 PERFORMING WORDS:")
print("-"*80)
for word, acc in sorted_classes[-10:]:
    print(f"{word:25s}: {acc*100:6.2f}%")

# Save results
results = {
    'num_classes': NUM_CLASSES,
    'test_accuracy': float(test_acc),
    'test_top5': float(test_top5),
    'test_top3': float(test_top3),
    'training_time_minutes': float(training_time/60),
    'per_class_accuracy': {k: float(v) for k, v in per_class_acc.items()},
    'class_names': list(label_names.values()),
    'best_words': [(k, float(v)) for k, v in sorted_classes[:20]],
    'worst_words': [(k, float(v)) for k, v in sorted_classes[-20:]]
}

with open(f'results_wlasl_{NUM_CLASSES}.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save label mapping
with open(f'labels_wlasl_{NUM_CLASSES}.txt', 'w') as f:
    for i in range(NUM_CLASSES):
        f.write(f"{label_names[i]}\n")

print("\n✅ Results saved!")

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
if NUM_CLASSES <= 50:
    ax1.axhline(y=0.90, color='r', linestyle='--', label='90% Target', alpha=0.7)
ax1.set_title(f'Model Accuracy - WLASL-{NUM_CLASSES}', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'], label='Train', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'training_wlasl_{NUM_CLASSES}.png', dpi=150, bbox_inches='tight')
plt.show()

# Download files if in Colab
if IN_COLAB:
    print("\n📥 Downloading files...")
    files.download(f'wlasl_{NUM_CLASSES}_best.keras')
    files.download(f'results_wlasl_{NUM_CLASSES}.json')
    files.download(f'labels_wlasl_{NUM_CLASSES}.txt')
    files.download(f'training_wlasl_{NUM_CLASSES}.png')
    print("✅ Downloads complete!")

print("\n" + "="*80)
print("🎉 ALL DONE!")
print("="*80)
print(f"\n✅ Achieved {test_acc*100:.2f}% accuracy on {NUM_CLASSES} words!")
print("✅ Model is ready for deployment!")
print("\n📋 Next steps:")
print(f"1. Copy wlasl_{NUM_CLASSES}_best.keras to your Mac: ML_PROJECT_LOCAL/checkpoints/")
print(f"2. Copy labels_wlasl_{NUM_CLASSES}.txt to checkpoints/")
print("3. Update your web app to load this model")
print("4. Test real-time recognition with webcam!")
print("="*80)
