"""
🎯 WLASL VIDEO PROCESSING + TRAINING - FIXED VERSION
Addresses low accuracy issues by ensuring quality data

FIXES:
- Reduced to 50 classes (more samples per class)
- Minimum 50 videos per class requirement
- Better error handling and logging
- Slower, more accurate feature extraction

TIME: ~2-3 hours total
"""

# ============================================================================
# STEP 0: INSTALL PACKAGES
# ============================================================================
print("📦 Installing required packages...")
import subprocess, sys

print("Fixing numpy...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==1.26.4"])
print("Installing mediapipe and opencv...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "mediapipe==0.10.21", "opencv-python==4.10.0.84"])
print("✅ Packages installed!\n")

import os, json, zipfile
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

print("="*80)
print("🎯 WLASL TRAINING - FIXED VERSION (50 CLASSES)")
print("="*80)
print()

# Configuration - REDUCED FOR BETTER ACCURACY
NUM_CLASSES = 50  # Reduced from 100
MIN_VIDEOS_PER_CLASS = 50  # Minimum samples
MAX_FRAMES = 30
BATCH_SIZE = 16  # Reduced for better learning
EPOCHS = 100

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU: {'✅ '+str(gpus[0]) if gpus else '❌ NO GPU'}")
print("="*80 + "\n")

# ============================================================================
# STEP 1: FIND DATASET
# ============================================================================
print("STEP 1: Finding dataset...")
print("-"*80)

input_dir = '/kaggle/input'
possible_dirs = [
    f'{input_dir}/wlsal-videos',
    f'{input_dir}/wlasl-videos',
    f'{input_dir}/wlasl',
]

dataset_dir = None
for path in possible_dirs:
    if os.path.exists(path):
        dataset_dir = path
        break

if not dataset_dir:
    for root, dirs, files in os.walk(input_dir):
        if 'videos' in dirs and 'WLASL_v0.3.json' in files:
            dataset_dir = root
            break

if not dataset_dir:
    raise FileNotFoundError("WLASL dataset not found!")

video_dir = f'{dataset_dir}/videos'
json_path = f'{dataset_dir}/WLASL_v0.3.json'

print(f"✅ Dataset: {dataset_dir}")
print(f"✅ Videos: {len([f for f in os.listdir(video_dir) if f.endswith('.mp4')]):,}")
print()

# ============================================================================
# STEP 2: SELECT CLASSES WITH ENOUGH DATA
# ============================================================================
print("STEP 2: Selecting classes with sufficient data...")
print("-"*80)

with open(json_path, 'r') as f:
    wlasl_data = json.load(f)

# Count available videos per class
class_video_counts = {}
video_to_label = {}

for entry in wlasl_data:
    gloss = entry['gloss']
    instances = entry['instances']
    
    # Count only videos that actually exist
    available_videos = []
    for inst in instances:
        video_id = inst['video_id']
        video_file = f"{video_id}.mp4"
        if os.path.exists(os.path.join(video_dir, video_file)):
            available_videos.append(video_id)
            video_to_label[video_id] = gloss
    
    if available_videos:
        class_video_counts[gloss] = len(available_videos)

# Filter classes with enough samples
valid_classes = {word: count for word, count in class_video_counts.items() 
                 if count >= MIN_VIDEOS_PER_CLASS}

# Select top N
top_classes = sorted(valid_classes.items(), key=lambda x: x[1], reverse=True)[:NUM_CLASSES]
selected_glosses = {word for word, _ in top_classes}

print(f"✅ Total classes in dataset: {len(class_video_counts)}")
print(f"✅ Classes with ≥{MIN_VIDEOS_PER_CLASS} videos: {len(valid_classes)}")
print(f"✅ Selected top {NUM_CLASSES} classes\n")

print("📊 Selected classes:")
for i, (word, count) in enumerate(top_classes[:15], 1):
    print(f"  {i:2d}. {word:20s}: {count:3d} videos")
if NUM_CLASSES > 15:
    print(f"  ... ({NUM_CLASSES-15} more)")
print()

total_expected = sum(count for _, count in top_classes)
print(f"📈 Expected total videos: {total_expected:,}")
print()

# ============================================================================
# STEP 3: EXTRACT FEATURES WITH ERROR HANDLING
# ============================================================================
print("STEP 3: Extracting MediaPipe features...")
print("-"*80)
print("⏱️  This will take 1-2 hours\n")

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.3,  # Lower threshold for more detections
    min_tracking_confidence=0.3
)

def extract_keypoints(results):
    """Extract keypoints"""
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark[:10]]) if results.face_landmarks else np.zeros((10, 3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    
    keypoints = np.concatenate([pose, face, lh, rh])
    if keypoints.shape[0] < 104:
        padding = np.zeros((104 - keypoints.shape[0], 3))
        keypoints = np.concatenate([keypoints, padding])
    
    return keypoints

def process_video(video_path, max_frames=30):
    """Extract features with better error handling"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        frames_data = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count == 0 or frame_count < 5:  # Skip very short videos
            cap.release()
            return None
        
        # Sample frames evenly
        frame_indices = np.linspace(0, frame_count-1, min(max_frames, frame_count), dtype=int)
        
        successful_frames = 0
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                frames_data.append(np.zeros((104, 3)))
                continue
            
            # Resize if too large (faster processing)
            if frame.shape[0] > 480:
                scale = 480 / frame.shape[0]
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            keypoints = extract_keypoints(results)
            frames_data.append(keypoints)
            successful_frames += 1
        
        cap.release()
        
        # Reject video if too few successful frames
        if successful_frames < max_frames * 0.5:  # At least 50% success
            return None
        
        # Pad to max_frames
        while len(frames_data) < max_frames:
            frames_data.append(np.zeros((104, 3)))
        
        return np.array(frames_data[:max_frames])
    
    except Exception as e:
        return None

# Process videos
all_features = []
all_labels = []
gloss_to_idx = {gloss: idx for idx, gloss in enumerate(sorted(selected_glosses))}

processed = 0
skipped = 0
class_sample_counts = {gloss: 0 for gloss in selected_glosses}

video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
print(f"📹 Scanning {len(video_files):,} videos")
print("🔄 Processing...\n")

for video_file in tqdm(video_files, desc="Extracting features"):
    video_id = video_file.replace('.mp4', '')
    
    if video_id not in video_to_label:
        continue
    
    gloss = video_to_label[video_id]
    if gloss not in selected_glosses:
        continue
    
    video_path = os.path.join(video_dir, video_file)
    features = process_video(video_path, MAX_FRAMES)
    
    if features is not None:
        all_features.append(features)
        all_labels.append(gloss_to_idx[gloss])
        class_sample_counts[gloss] += 1
        processed += 1
    else:
        skipped += 1
    
    # Progress update
    if processed % 200 == 0 and processed > 0:
        print(f"\n  ✅ Processed: {processed}, ❌ Skipped: {skipped}")

holistic.close()

print(f"\n{'='*80}")
print("✅ FEATURE EXTRACTION COMPLETE!")
print(f"{'='*80}")
print(f"✅ Processed: {processed} videos")
print(f"❌ Skipped: {skipped} videos")
print(f"📊 Success rate: {processed/(processed+skipped)*100:.1f}%\n")

# Show per-class distribution
print("📊 Samples per class:")
sorted_counts = sorted(class_sample_counts.items(), key=lambda x: x[1], reverse=True)
for i, (word, count) in enumerate(sorted_counts[:10], 1):
    print(f"  {i:2d}. {word:20s}: {count:3d} samples")
if len(sorted_counts) > 10:
    print(f"  ... ({len(sorted_counts)-10} more)")
print()

# Check if we have enough data
if processed < NUM_CLASSES * 30:  # At least 30 per class
    print(f"⚠️  WARNING: Only {processed} samples for {NUM_CLASSES} classes!")
    print(f"   That's only {processed/NUM_CLASSES:.1f} per class on average.")
    print(f"   Consider reducing NUM_CLASSES or checking video files.")

# Convert to arrays
X_full = np.array(all_features, dtype=np.float32)
y_full = np.array(all_labels, dtype=np.int32)

print(f"✅ Final dataset shape: {X_full.shape}")
print(f"   Samples: {len(X_full):,}")
print(f"   Classes: {len(np.unique(y_full))}")
print()

# ============================================================================
# STEP 4: PREPARE DATA
# ============================================================================
print("STEP 4: Preparing data...")
print("-"*80)

# Normalize
X_full = (X_full - X_full.mean()) / (X_full.std() + 1e-8)
X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], -1)

# Split with stratification
X_temp, X_test, y_temp, y_test = train_test_split(X_full, y_full, test_size=0.15, random_state=42, stratify=y_full)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

print(f"✅ Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")
print(f"   Average per class: {len(X_train)/NUM_CLASSES:.1f} train, {len(X_val)/NUM_CLASSES:.1f} val")
print("="*80 + "\n")

# ============================================================================
# STEP 5: BUILD MODEL
# ============================================================================
print("STEP 5: Building model...")
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
    
    # 3 Transformer blocks
    x = transformer_encoder(x, 64, 4, 512, 0.15)
    x = transformer_encoder(x, 64, 4, 512, 0.15)
    x = transformer_encoder(x, 64, 4, 512, 0.15)
    
    # BiLSTM layers
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(256))(x)
    x = layers.Dropout(0.4)(x)
    
    # Dense layers
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
# STEP 6: TRAIN
# ============================================================================
print("STEP 6: Training...")
print("-"*80)
print(f"🚀 Target: >70% accuracy for {NUM_CLASSES} classes")
print(f"   Random baseline: {100/NUM_CLASSES:.1f}%\n")

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

model.compile(
    optimizer=keras.optimizers.Adam(0.0001),  # Lower learning rate
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy',
             keras.metrics.SparseTopKCategoricalAccuracy(k=min(5, NUM_CLASSES), name='top_5'),
             keras.metrics.SparseTopKCategoricalAccuracy(k=min(3, NUM_CLASSES), name='top_3')]
)

callbacks = [
    keras.callbacks.ModelCheckpoint(f'wlasl_{NUM_CLASSES}_best.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1)
]

start = datetime.now()
history = model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val), 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    callbacks=callbacks, 
    verbose=1
)
training_time = (datetime.now() - start).total_seconds() / 60

print(f"\n✅ TRAINING COMPLETE! ({training_time:.1f} min)")
print("="*80 + "\n")

# ============================================================================
# STEP 7: EVALUATE
# ============================================================================
print("STEP 7: Final Evaluation...")
print("-"*80)

test_loss, test_acc, test_top5, test_top3 = model.evaluate(X_test, y_test, verbose=0)

print("="*80)
print("🎯 FINAL RESULTS")
print("="*80)
print(f"Classes: {NUM_CLASSES}")
print(f"Random baseline: {100/NUM_CLASSES:.2f}%")
print(f"")
print(f"Test Accuracy:  {test_acc*100:.2f}% {'🎉🎉🎉' if test_acc >= 0.70 else '✅' if test_acc >= 0.50 else '⚠️' if test_acc >= 0.30 else '❌'}")
print(f"Top-3 Accuracy: {test_top3*100:.2f}%")
print(f"Top-5 Accuracy: {test_top5*100:.2f}%")
print("="*80)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Val', linewidth=2)
ax1.axhline(y=100/NUM_CLASSES/100, color='gray', linestyle=':', alpha=0.5, label='Random')
ax1.set_title(f'Accuracy - {NUM_CLASSES} Classes', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'], label='Train', linewidth=2)
ax2.plot(history.history['val_loss'], label='Val', linewidth=2)
ax2.set_title('Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'training_{NUM_CLASSES}.png', dpi=150, bbox_inches='tight')
plt.show()

# Save labels
with open('wlasl_labels.txt', 'w') as f:
    idx_to_word = {idx: word for word, idx in gloss_to_idx.items()}
    for idx in range(NUM_CLASSES):
        f.write(f"{idx_to_word[idx]}\n")

print()
print("="*80)
print("🎉 TRAINING COMPLETE!")
print("="*80)
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
print(f"✅ Files saved:")
print(f"   - wlasl_{NUM_CLASSES}_best.keras (model)")
print(f"   - wlasl_labels.txt (class names)")
print(f"   - training_{NUM_CLASSES}.png (graphs)")
print("="*80)
