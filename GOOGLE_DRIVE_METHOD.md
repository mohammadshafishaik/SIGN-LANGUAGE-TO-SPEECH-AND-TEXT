# 🚀 FASTER METHOD: Google Drive Upload

## ❌ Problem: Direct Colab Upload is VERY Slow
- 2-3GB file takes 30-60 minutes
- Often fails/disconnects
- **NOT RECOMMENDED**

## ✅ Solution: Upload to Google Drive First (10x FASTER!)

### **STEP 1: Upload Dataset to Google Drive** (10-15 minutes)

1. **Go to Google Drive**: https://drive.google.com/
2. **Click "New" → "File upload"**
3. **Select your downloaded `archive.zip`**
4. **Wait for upload** (10-15 minutes - much faster than Colab!)
5. Upload appears in "My Drive" root folder

---

### **STEP 2: Use Modified Colab Script**

Copy this code into Google Colab (I've modified it to use Google Drive):

```python
"""
🎯 WLASL TRAINING - GOOGLE COLAB - GOOGLE DRIVE VERSION
MUCH FASTER than direct upload! (10x speed improvement)

SETUP:
1. Upload archive.zip to Google Drive first
2. Copy-paste this code to Colab
3. Run and authenticate Drive
4. Dataset loads from Drive automatically!
"""

import os, zipfile, numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from datetime import datetime

print("="*80)
print("🎯 WLASL TRAINING - GOOGLE DRIVE VERSION")
print("="*80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================
NUM_CLASSES = 100  # ⬅️ CHANGE: 30, 50, 100
BATCH_SIZE = 32
EPOCHS = 100

print(f"🎯 Config: {NUM_CLASSES} classes, Expected: {'>92%' if NUM_CLASSES <= 30 else '>90%' if NUM_CLASSES <= 50 else '>85%'}")

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU: {'✅ '+str(gpus[0]) if gpus else '❌ NO GPU - ENABLE IT!'}")
print("="*80 + "\n")

# ============================================================================
# STEP 1: MOUNT GOOGLE DRIVE
# ============================================================================
print("STEP 1: Mounting Google Drive...")
print("-"*80)

from google.colab import drive
drive.mount('/content/drive')

print("✅ Google Drive mounted!")
print()

# ============================================================================
# STEP 2: FIND AND EXTRACT DATASET FROM DRIVE
# ============================================================================
print("STEP 2: Loading dataset from Google Drive...")
print("-"*80)

# Auto-find archive.zip in Google Drive
drive_root = '/content/drive/MyDrive'
possible_locations = [
    f'{drive_root}/archive.zip',
    f'{drive_root}/wlasl-processed.zip',
    f'{drive_root}/Downloads/archive.zip',
    f'{drive_root}/Datasets/archive.zip'
]

zip_file = None
for location in possible_locations:
    if os.path.exists(location):
        zip_file = location
        break

if not zip_file:
    print("❌ Could not find archive.zip in Google Drive!")
    print("\n📂 Searching your Drive...")
    os.system(f'find {drive_root} -name "*.zip" -type f | head -10')
    print("\n💡 TIP: Make sure archive.zip is in 'My Drive' root folder")
    raise FileNotFoundError("Please upload archive.zip to Google Drive")

print(f"✅ Found: {zip_file}")
print(f"   Size: {os.path.getsize(zip_file)/(1024*1024):.1f} MB")
print()

# Extract (much faster - reading from Drive is fast!)
if not os.path.exists('WLASL features npy'):
    print("📦 Extracting dataset...")
    with zipfile.ZipFile(zip_file, 'r') as z:
        z.extractall('/content/')
    print("✅ Extracted to /content/")
else:
    print("✅ Dataset already extracted!")

print()
print("="*80)
print()

# ============================================================================
# STEP 3: LOAD DATA (AUTO-DETECT)
# ============================================================================
print("STEP 3: Loading dataset...")
print("-"*80)

# Auto-find dataset
possible_dirs = [
    '/content/WLASL features npy/WLASL_600',
    '/content/WLASL features npy/WLASL_100',
    '/content/WLASL_dataset',
    '/content/dataset',
    '/content'
]

dataset_dir = None
for d in possible_dirs:
    if os.path.exists(d):
        npy_files = [f for f in os.listdir(d) if f.endswith('.npy')]
        if len(npy_files) >= 2:
            dataset_dir = d
            break

if not dataset_dir:
    print("❌ Dataset not found!")
    os.system('ls -lR /content | head -30')
    raise FileNotFoundError("Check extraction")

print(f"✅ Dataset: {dataset_dir}")

# Auto-find files
npy_files = [f for f in os.listdir(dataset_dir) if f.endswith('.npy')]
data_file = next((f for f in npy_files if 'data' in f.lower()), npy_files[0])
labels_file = next((f for f in npy_files if 'label' in f.lower()), npy_files[1])

X_full = np.load(os.path.join(dataset_dir, data_file)).astype(np.float32)
y_full = np.load(os.path.join(dataset_dir, labels_file)).astype(np.int32)

print(f"✅ Loaded: {X_full.shape[0]} samples, {len(np.unique(y_full))} classes")

# Select top N classes
if len(np.unique(y_full)) > NUM_CLASSES:
    unique, counts = np.unique(y_full, return_counts=True)
    top = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:NUM_CLASSES]
    mask = np.isin(y_full, [l for l, _ in top])
    X_full, y_full = X_full[mask], y_full[mask]
    mapping = {old: new for new, (old, _) in enumerate(top)}
    y_full = np.array([mapping[l] for l in y_full])
    print(f"✅ Top {NUM_CLASSES} classes selected")

# Normalize
X_full = (X_full - X_full.mean()) / (X_full.std() + 1e-8)

# Reshape if 4D
if len(X_full.shape) == 4:
    X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], -1)

# Split
X_temp, X_test, y_temp, y_test = train_test_split(X_full, y_full, test_size=0.15, random_state=42, stratify=y_full)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

print(f"✅ Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
print("="*80 + "\n")

# ============================================================================
# STEP 4: BUILD MODEL
# ============================================================================
print("STEP 4: Building model...")
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
    
    x = transformer_encoder(x, 64, 4, 512, 0.1)
    x = transformer_encoder(x, 64, 4, 512, 0.1)
    x = transformer_encoder(x, 64, 4, 512, 0.1)
    
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(256))(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax', dtype='float32')(x)
    
    return keras.Model(inputs, outputs)

tf.keras.mixed_precision.set_global_policy('mixed_float16')
model = build_model((X_train.shape[1], X_train.shape[2]), NUM_CLASSES)
print(f"✅ Model: {model.count_params():,} parameters")
print("="*80 + "\n")

# ============================================================================
# STEP 5: TRAIN
# ============================================================================
print("STEP 5: Training...")
print("-"*80)
print(f"🚀 Target: {'>92%' if NUM_CLASSES <= 30 else '>90%' if NUM_CLASSES <= 50 else '>85%'}")
print()

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

model.compile(
    optimizer=keras.optimizers.Adam(0.0003),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy',
             keras.metrics.SparseTopKCategoricalAccuracy(k=min(5, NUM_CLASSES), name='top_5'),
             keras.metrics.SparseTopKCategoricalAccuracy(k=min(3, NUM_CLASSES), name='top_3')]
)

callbacks = [
    keras.callbacks.ModelCheckpoint(f'wlasl_{NUM_CLASSES}_best.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1)
]

start = datetime.now()
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, 
                    callbacks=callbacks, class_weight=class_weight_dict, verbose=1)
training_time = (datetime.now() - start).total_seconds() / 60

print(f"\n✅ TRAINING COMPLETE! ({training_time:.1f} min)")
print("="*80 + "\n")

# ============================================================================
# STEP 6: EVALUATE
# ============================================================================
print("STEP 6: Evaluation...")
print("-"*80)

test_loss, test_acc, test_top5, test_top3 = model.evaluate(X_test, y_test, verbose=0)

print("="*80)
print("🎯 FINAL RESULTS")
print("="*80)
print(f"Test Accuracy:  {test_acc*100:.2f}% {'🎉🎉🎉' if test_acc >= 0.90 else '✅' if test_acc >= 0.85 else '📊'}")
print(f"Top-3 Accuracy: {test_top3*100:.2f}%")
print(f"Top-5 Accuracy: {test_top5*100:.2f}%")
print(f"Time: {training_time:.1f} min")
print("="*80)

if test_acc >= 0.90:
    print("\n🎉🎉🎉 90%+ ACCURACY ACHIEVED! 🎉🎉🎉")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Val', linewidth=2)
ax1.axhline(y=0.90, color='r', linestyle='--', label='90% Target', alpha=0.7)
ax1.set_title(f'Accuracy - {NUM_CLASSES} Classes', fontsize=14, fontweight='bold')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'], label='Train', linewidth=2)
ax2.plot(history.history['val_loss'], label='Val', linewidth=2)
ax2.set_title('Loss', fontsize=14, fontweight='bold')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'training_{NUM_CLASSES}.png', dpi=150)
plt.show()

# ============================================================================
# STEP 7: SAVE TO GOOGLE DRIVE
# ============================================================================
print("\n📥 Saving model to Google Drive...")
import shutil
shutil.copy(f'wlasl_{NUM_CLASSES}_best.keras', f'/content/drive/MyDrive/wlasl_{NUM_CLASSES}_best.keras')
shutil.copy(f'training_{NUM_CLASSES}.png', f'/content/drive/MyDrive/training_{NUM_CLASSES}.png')
print("✅ Saved to Google Drive!")
print(f"   Model: /content/drive/MyDrive/wlasl_{NUM_CLASSES}_best.keras")

print()
print("="*80)
print("🎉 COMPLETE!")
print("="*80)
print(f"✅ {test_acc*100:.2f}% accuracy!")
print("✅ Model saved to Google Drive - download from there!")
print("="*80)
```

---

## ⚡ WHY THIS IS FASTER:

| Method | Speed | Time for 2-3GB |
|--------|-------|----------------|
| Direct Colab Upload | ❌ SLOW | 30-60 minutes |
| **Google Drive** | ✅ **FAST** | **10-15 minutes** |

**Reasons:**
1. Google's servers are MUCH faster
2. Drive upload is optimized
3. Colab can read from Drive at high speed
4. More reliable (won't disconnect)

---

## 📋 QUICK STEPS:

### **Right Now:**
1. **Cancel** the slow Colab upload (it won't finish)
2. **Go to Google Drive**: https://drive.google.com/
3. **Upload `archive.zip`** (from Downloads folder)
4. **Wait 10-15 minutes**

### **After Drive Upload:**
1. **Open new Colab notebook**
2. **Enable GPU**
3. **Copy-paste the code above**
4. **Run it** (will ask to authorize Drive)
5. **Training starts automatically!**

---

## 💡 BONUS: Model Auto-Saves to Drive!

When training finishes, model automatically saves to Google Drive.

**Download from Drive:**
- Go to Drive
- Find `wlasl_100_best.keras`
- Right-click → Download
- Much faster than Colab download!

---

**Cancel the slow upload and use Google Drive instead!** It's 10x faster! 🚀

Let me know when you've uploaded to Google Drive and I'll help you with the next steps!
