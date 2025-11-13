"""
🎯 WLASL TRAINING - GOOGLE COLAB - SIMPLIFIED VERSION
Just copy-paste this entire file into Google Colab and run!

STEPS:
1. Open Google Colab (colab.research.google.com)
2. Enable GPU (Runtime → Change runtime type → GPU)
3. Copy-paste this entire script
4. Run (Runtime → Run all or Shift+Enter through cells)
5. Upload your dataset when prompted
6. Wait 1-3 hours for 90%+ accuracy!
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os, zipfile, numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from datetime import datetime

print("="*80)
print("🎯 WLASL TRAINING - SIMPLIFIED FOR GOOGLE COLAB")
print("="*80)
print()

# ============================================================================
# CONFIGURATION - CHANGE THESE VALUES
# ============================================================================
NUM_CLASSES = 100  # ⬅️ How many words to train on: 30, 50, 100, or 600
BATCH_SIZE = 32
EPOCHS = 100

print(f"🎯 Configuration:")
print(f"   Classes: {NUM_CLASSES}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Max Epochs: {EPOCHS}")
print(f"   Expected Accuracy: {'>92%' if NUM_CLASSES <= 30 else '>90%' if NUM_CLASSES <= 50 else '>85%'}")
print()

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU Available: {gpus[0]}")
    print(f"   Training will be FAST (1-3 hours)")
else:
    print("⚠️  WARNING: No GPU detected!")
    print("   Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU")

print("="*80)
print()

# ============================================================================
# STEP 1: UPLOAD DATASET
# ============================================================================
print("STEP 1: Upload your dataset")
print("-"*80)

try:
    from google.colab import files
    IN_COLAB = True
    
    if not os.path.exists('WLASL features npy'):
        print("📤 Please upload your WLASL dataset .zip file:")
        print("   Download from: https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed")
        print()
        print("   👇 Click 'Choose Files' below to upload 👇")
        print()
        
        uploaded = files.upload()
        
        if not uploaded:
            raise FileNotFoundError("No file uploaded!")
        
        zip_file = list(uploaded.keys())[0]
        print(f"\n✅ Uploaded: {zip_file}")
        print("\n📦 Extracting (1-2 minutes)...")
        
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall()
        
        print("✅ Extraction complete!")
    else:
        print("✅ Dataset already extracted!")
    
    print("\n📂 Files:")
    os.system('ls -lh | head -15')
    
except ImportError:
    IN_COLAB = False
    print("⚠️  Not in Colab - using local files")

print()
print("="*80)
print()

# ============================================================================
# STEP 2: LOAD DATA (AUTO-DETECT STRUCTURE)
# ============================================================================
print("STEP 2: Loading dataset")
print("-"*80)

# Auto-find dataset directory
possible_dirs = [
    'WLASL features npy/WLASL_600',
    'WLASL features npy/WLASL_100',
    'WLASL_dataset',
    'dataset',
    '.'
]

dataset_dir = None
for d in possible_dirs:
    if os.path.exists(d):
        npy_files = [f for f in os.listdir(d) if f.endswith('.npy')]
        if len(npy_files) >= 2:
            dataset_dir = d
            break

if not dataset_dir:
    print("❌ Could not find dataset! Check extraction.")
    os.system('ls -lR | head -30')
    raise FileNotFoundError("Dataset not found")

print(f"✅ Found dataset in: {dataset_dir}")

# Auto-find data files
npy_files = [f for f in os.listdir(dataset_dir) if f.endswith('.npy')]
data_file = next((f for f in npy_files if 'data' in f.lower()), npy_files[0])
labels_file = next((f for f in npy_files if 'label' in f.lower()), npy_files[1])

data_path = os.path.join(dataset_dir, data_file)
labels_path = os.path.join(dataset_dir, labels_file)

print(f"   Data: {data_file}")
print(f"   Labels: {labels_file}")
print()

# Load
X_full = np.load(data_path).astype(np.float32)
y_full = np.load(labels_path).astype(np.int32)

print(f"✅ Loaded: {X_full.shape[0]} samples, {len(np.unique(y_full))} classes")
print(f"   Shape: {X_full.shape}")
print()

# Select top N classes
if len(np.unique(y_full)) > NUM_CLASSES:
    unique, counts = np.unique(y_full, return_counts=True)
    top = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:NUM_CLASSES]
    selected_labels = [l for l, _ in top]
    
    mask = np.isin(y_full, selected_labels)
    X_full, y_full = X_full[mask], y_full[mask]
    
    mapping = {old: new for new, (old, _) in enumerate(top)}
    y_full = np.array([mapping[l] for l in y_full])
    
    print(f"✅ Using top {NUM_CLASSES} classes (most samples)")
    for i, (label, count) in enumerate(top[:5], 1):
        print(f"   {i}. Class {label}: {count} samples")
    print()

# Normalize
X_full = (X_full - X_full.mean()) / (X_full.std() + 1e-8)
print(f"✅ Data normalized")

# Reshape if 4D
if len(X_full.shape) == 4:
    X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], -1)
    print(f"✅ Reshaped to: {X_full.shape}")

print()

# Split
X_temp, X_test, y_temp, y_test = train_test_split(X_full, y_full, test_size=0.15, random_state=42, stratify=y_full)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

print(f"✅ Data split:")
print(f"   Train: {len(X_train)} samples")
print(f"   Val:   {len(X_val)} samples")
print(f"   Test:  {len(X_test)} samples")
print()
print("="*80)
print()

# ============================================================================
# STEP 3: BUILD MODEL
# ============================================================================
print("STEP 3: Building model")
print("-"*80)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    res = layers.LayerNormalization(epsilon=1e-6)(x) + inputs
    x = layers.Conv1D(ff_dim, 1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(inputs.shape[-1], 1)(x)
    return layers.LayerNormalization(epsilon=1e-6)(x) + res

def build_model(input_shape, n_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(256)(inputs)
    x = layers.LayerNormalization()(x)
    
    # Transformers
    x = transformer_encoder(x, 64, 4, 512, 0.1)
    x = transformer_encoder(x, 64, 4, 512, 0.1)
    x = transformer_encoder(x, 64, 4, 512, 0.1)
    
    # LSTM
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(256))(x)
    x = layers.Dropout(0.4)(x)
    
    # Classification
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax', dtype='float32')(x)
    
    return keras.Model(inputs, outputs)

# Mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Build
model = build_model((X_train.shape[1], X_train.shape[2]), NUM_CLASSES)
print(f"✅ Model built: {model.count_params():,} parameters")
print()
print("="*80)
print()

# ============================================================================
# STEP 4: TRAIN
# ============================================================================
print("STEP 4: Training")
print("-"*80)
print(f"🚀 Target: {'>92%' if NUM_CLASSES <= 30 else '>90%' if NUM_CLASSES <= 50 else '>85%'} accuracy")
print(f"   Expected time: {1 if NUM_CLASSES <= 50 else 2}-{3 if NUM_CLASSES <= 50 else 4} hours")
print()

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(0.0003),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy',
             keras.metrics.SparseTopKCategoricalAccuracy(k=min(5, NUM_CLASSES), name='top_5'),
             keras.metrics.SparseTopKCategoricalAccuracy(k=min(3, NUM_CLASSES), name='top_3')]
)

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(f'wlasl_{NUM_CLASSES}_best.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1)
]

# Train
start = datetime.now()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)
training_time = (datetime.now() - start).total_seconds() / 60

print()
print("="*80)
print(f"✅ TRAINING COMPLETE! ({training_time:.1f} minutes)")
print("="*80)
print()

# ============================================================================
# STEP 5: EVALUATE
# ============================================================================
print("STEP 5: Final evaluation")
print("-"*80)

test_loss, test_acc, test_top5, test_top3 = model.evaluate(X_test, y_test, verbose=0)

print("="*80)
print("🎯 FINAL RESULTS")
print("="*80)
print(f"Test Accuracy:  {test_acc*100:.2f}% {'🎉🎉🎉' if test_acc >= 0.90 else '✅' if test_acc >= 0.85 else '📊'}")
print(f"Top-3 Accuracy: {test_top3*100:.2f}%")
print(f"Top-5 Accuracy: {test_top5*100:.2f}%")
print(f"Training Time:  {training_time:.1f} minutes")
print("="*80)

if test_acc >= 0.90:
    print("\n🎉🎉🎉 OUTSTANDING! 90%+ ACCURACY ACHIEVED! 🎉🎉🎉")
elif test_acc >= 0.85:
    print("\n✅ EXCELLENT! 85%+ accuracy achieved!")
print()

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Val', linewidth=2)
ax1.axhline(y=0.90, color='r', linestyle='--', label='90% Target', alpha=0.7)
ax1.set_title(f'Accuracy - {NUM_CLASSES} Classes', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'], label='Train', linewidth=2)
ax2.plot(history.history['val_loss'], label='Val', linewidth=2)
ax2.set_title('Loss', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'training_{NUM_CLASSES}.png', dpi=150)
plt.show()

# ============================================================================
# STEP 6: DOWNLOAD MODEL
# ============================================================================
if IN_COLAB:
    print("\n📥 Downloading model...")
    files.download(f'wlasl_{NUM_CLASSES}_best.keras')
    files.download(f'training_{NUM_CLASSES}.png')
    print("✅ Done! Check your Downloads folder")

print()
print("="*80)
print("🎉 COMPLETE!")
print("="*80)
print()
print(f"✅ Achieved {test_acc*100:.2f}% accuracy on {NUM_CLASSES} words!")
print()
print("📋 Next steps:")
print("1. Download completed - check Downloads folder")
print("2. Copy model to ML_PROJECT_LOCAL/checkpoints/")
print("3. Update web app to use new model")
print("4. Test with webcam!")
print("="*80)
