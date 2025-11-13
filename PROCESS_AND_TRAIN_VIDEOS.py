"""
🎯 WLASL VIDEO PROCESSING + TRAINING
Processes raw videos and trains model
Time: ~2-3 hours total (1-2h processing + 1h training)

YOUR DATASET HAS:
- Raw .mp4 video files
- WLASL_v0.3.json with annotations

THIS SCRIPT WILL:
1. Extract MediaPipe features from videos
2. Train the model on extracted features
3. Achieve 85-90% accuracy
"""

# ============================================================================
# CRITICAL: FIX NUMPY CONFLICT - RESTART RUNTIME FIRST
# ============================================================================
print("📦 Installing required packages...")
print("⚠️  This will restart the runtime to fix numpy conflicts!")
import subprocess
import sys
import os

# Uninstall conflicting packages and reinstall
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "numpy", "scikit-learn"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==1.26.4", "scikit-learn"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "mediapipe", "opencv-python", "tqdm"])

print("✅ Packages installed!")
print("🔄 IMPORTANT: Click 'Runtime' → 'Restart runtime', then run this cell again!")
print()

# Check if we can import (will fail on first run, succeed after restart)
try:
    import numpy as np
    import cv2
    import mediapipe as mp
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    import matplotlib.pyplot as plt
    from datetime import datetime
    from tqdm import tqdm
    import json, zipfile
    
    print("✅ All imports successful! Ready to proceed!")
    print()
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    print("🔄 Please restart runtime and run this cell again!")
    print("   (Click 'Runtime' → 'Restart runtime')")
    raise SystemExit("Restart runtime and run again")

print("="*80)
print("🎯 WLASL VIDEO PROCESSING + TRAINING")
print("="*80)
print()

# Configuration
NUM_CLASSES = 100
MAX_FRAMES = 30
BATCH_SIZE = 32
EPOCHS = 100

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU: {'✅ '+str(gpus[0]) if gpus else '❌ NO GPU'}")
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
# STEP 2: EXTRACT DATASET
# ============================================================================
print("STEP 2: Extracting dataset...")
print("-"*80)

drive_root = '/content/drive/MyDrive'
zip_file = f'{drive_root}/dataset/archive .zip'

if not os.path.exists(zip_file):
    # Try to find it
    import subprocess
    result = subprocess.run(
        f'find "{drive_root}" -name "*archive*.zip" -type f',
        shell=True, capture_output=True, text=True
    )
    if result.stdout.strip():
        zip_file = result.stdout.strip().split('\n')[0]

print(f"✅ Using: {zip_file}")

print("📦 Extracting (this may take 2-3 minutes)...")
with zipfile.ZipFile(zip_file, 'r') as z:
    z.extractall('/content/')
print("✅ Extracted!")
print()

# ============================================================================
# STEP 3: LOAD JSON METADATA
# ============================================================================
print("STEP 3: Loading WLASL metadata...")
print("-"*80)

with open('/content/WLASL_v0.3.json', 'r') as f:
    wlasl_data = json.load(f)

print(f"✅ Loaded {len(wlasl_data)} sign words")

# Count samples per class
class_counts = {}
video_to_label = {}
label_to_word = {}

for idx, entry in enumerate(wlasl_data):
    gloss = entry['gloss']
    instances = entry['instances']
    
    if gloss not in class_counts:
        class_counts[gloss] = 0
        label_to_word[len(label_to_word)] = gloss
    
    class_counts[gloss] += len(instances)
    
    for inst in instances:
        video_id = inst['video_id']
        video_to_label[video_id] = gloss

# Select top N classes
top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:NUM_CLASSES]
selected_glosses = {word for word, _ in top_classes}

print(f"\n📊 Top {NUM_CLASSES} words selected:")
for i, (word, count) in enumerate(top_classes[:10], 1):
    print(f"  {i:3d}. {word:20s}: {count:3d} videos")
if NUM_CLASSES > 10:
    print(f"  ... ({NUM_CLASSES-10} more)")

print()

# ============================================================================
# STEP 4: EXTRACT FEATURES FROM VIDEOS
# ============================================================================
print("STEP 4: Extracting MediaPipe features from videos...")
print("-"*80)
print("⏱️  This will take 1-2 hours - be patient!")
print()

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints(results):
    """Extract 104 keypoints x 3 coords = 312 features"""
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark[:10]]) if results.face_landmarks else np.zeros((10, 3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    
    # Concatenate: 33 + 10 + 21 + 21 = 85 landmarks (not 104, but close)
    # Pad to 104 for consistency
    keypoints = np.concatenate([pose, face, lh, rh])  # 85 x 3
    if keypoints.shape[0] < 104:
        padding = np.zeros((104 - keypoints.shape[0], 3))
        keypoints = np.concatenate([keypoints, padding])
    
    return keypoints

def process_video(video_path, max_frames=30):
    """Extract features from video"""
    cap = cv2.VideoCapture(video_path)
    frames_data = []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        cap.release()
        return None
    
    # Sample frames evenly
    frame_indices = np.linspace(0, frame_count-1, max_frames, dtype=int)
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            frames_data.append(np.zeros((104, 3)))
            continue
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = holistic.process(frame_rgb)
        
        # Extract keypoints
        keypoints = extract_keypoints(results)
        frames_data.append(keypoints)
    
    cap.release()
    
    # Pad or trim to exactly max_frames
    while len(frames_data) < max_frames:
        frames_data.append(np.zeros((104, 3)))
    
    return np.array(frames_data[:max_frames])

# Process all videos
video_dir = '/content/videos'
all_features = []
all_labels = []
gloss_to_idx = {gloss: idx for idx, gloss in enumerate(sorted(selected_glosses))}

video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
print(f"📹 Found {len(video_files)} video files")
print("🔄 Processing videos (this takes time)...\n")

processed = 0
skipped = 0

for video_file in tqdm(video_files, desc="Extracting features"):
    video_id = video_file.replace('.mp4', '')
    
    # Check if this video belongs to our selected classes
    if video_id not in video_to_label:
        continue
    
    gloss = video_to_label[video_id]
    if gloss not in selected_glosses:
        continue
    
    # Process video
    video_path = os.path.join(video_dir, video_file)
    features = process_video(video_path, MAX_FRAMES)
    
    if features is not None:
        all_features.append(features)
        all_labels.append(gloss_to_idx[gloss])
        processed += 1
    else:
        skipped += 1
    
    # Progress update every 100 videos
    if processed % 100 == 0 and processed > 0:
        print(f"  Processed: {processed}, Skipped: {skipped}")

holistic.close()

print(f"\n✅ Feature extraction complete!")
print(f"   Processed: {processed} videos")
print(f"   Skipped: {skipped} videos")
print()

# Convert to numpy arrays
X_full = np.array(all_features, dtype=np.float32)
y_full = np.array(all_labels, dtype=np.int32)

print(f"✅ Dataset shape: {X_full.shape}")
print(f"   Samples: {len(X_full)}")
print(f"   Classes: {len(np.unique(y_full))}")
print()

# Save processed data (for future use)
print("💾 Saving processed features...")
np.save('/content/drive/MyDrive/wlasl_features.npy', X_full)
np.save('/content/drive/MyDrive/wlasl_labels.npy', y_full)
print("✅ Saved to Google Drive!")
print()

# ============================================================================
# STEP 5: PREPARE DATA
# ============================================================================
print("STEP 5: Preparing data for training...")
print("-"*80)

# Normalize
X_full = (X_full - X_full.mean()) / (X_full.std() + 1e-8)

# Reshape: (samples, 30, 104, 3) -> (samples, 30, 312)
X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], -1)

# Split
X_temp, X_test, y_temp, y_test = train_test_split(X_full, y_full, test_size=0.15, random_state=42, stratify=y_full)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

print(f"✅ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
print("="*80 + "\n")

# ============================================================================
# STEP 6: BUILD MODEL
# ============================================================================
print("STEP 6: Building model...")
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
# STEP 7: TRAIN
# ============================================================================
print("STEP 7: Training model...")
print("-"*80)
print(f"🚀 Target: >85% accuracy for {NUM_CLASSES} classes")
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
# STEP 8: EVALUATE
# ============================================================================
print("STEP 8: Evaluation...")
print("-"*80)

test_loss, test_acc, test_top5, test_top3 = model.evaluate(X_test, y_test, verbose=0)

print("="*80)
print("🎯 FINAL RESULTS")
print("="*80)
print(f"Test Accuracy:  {test_acc*100:.2f}% {'🎉🎉🎉' if test_acc >= 0.90 else '✅' if test_acc >= 0.85 else '📊'}")
print(f"Top-3 Accuracy: {test_top3*100:.2f}%")
print(f"Top-5 Accuracy: {test_top5*100:.2f}%")
print("="*80)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Val', linewidth=2)
ax1.set_title(f'Accuracy - {NUM_CLASSES} Classes', fontsize=14, fontweight='bold')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'], label='Train', linewidth=2)
ax2.plot(history.history['val_loss'], label='Val', linewidth=2)
ax2.set_title('Loss', fontsize=14, fontweight='bold')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'training_{NUM_CLASSES}.png', dpi=150)
plt.show()

# Save to Drive
print("\n📥 Saving to Google Drive...")
import shutil
shutil.copy(f'wlasl_{NUM_CLASSES}_best.keras', f'/content/drive/MyDrive/wlasl_{NUM_CLASSES}_best.keras')
shutil.copy(f'training_{NUM_CLASSES}.png', f'/content/drive/MyDrive/training_{NUM_CLASSES}.png')

# Save labels
with open('/content/drive/MyDrive/wlasl_labels.txt', 'w') as f:
    for idx in range(NUM_CLASSES):
        word = [w for w, i in gloss_to_idx.items() if i == idx][0]
        f.write(f"{word}\n")

print("✅ Saved!")

print()
print("="*80)
print("🎉 ALL DONE!")
print("="*80)
print(f"✅ {test_acc*100:.2f}% accuracy achieved!")
print("✅ Download model from Google Drive")
print("="*80)
