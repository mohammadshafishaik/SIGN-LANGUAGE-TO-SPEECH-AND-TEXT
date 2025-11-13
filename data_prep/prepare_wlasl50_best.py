#!/usr/bin/env python3
"""
Prepare WLASL Top 50 dataset - only classes with 150+ samples
This ensures better quality training data for 80%+ accuracy
"""

import os
import json
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

print("=" * 70)
print("📦 PREPARING WLASL TOP 50 DATASET (150+ samples each)")
print("=" * 70)

# Paths
DATA_DIR = "datasets_local.nosync"
OUTPUT_DIR = "dataset/wlasl_50_best"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print("\n📂 Loading data...")
with open(f"{DATA_DIR}/train_100.json", 'r') as f:
    class_mapping = json.load(f)

with open(f"{DATA_DIR}/wasl100_landmarks_train.json", 'r') as f:
    train_landmarks = json.load(f)

with open(f"{DATA_DIR}/wasl100_landmarks_val.json", 'r') as f:
    val_landmarks = json.load(f)

with open(f"{DATA_DIR}/wasl100_landmarks_test.json", 'r') as f:
    test_landmarks = json.load(f)

print(f"✅ Loaded {len(class_mapping)} class mappings")
print(f"✅ Loaded {len(train_landmarks)} train videos")
print(f"✅ Loaded {len(val_landmarks)} val videos")
print(f"✅ Loaded {len(test_landmarks)} test videos")

# Count samples per class
print("\n📊 Counting samples per class...")
class_counts = defaultdict(int)

for video_id, frames in train_landmarks.items():
    if len(frames) > 0:
        class_id = frames[0].get('class_id')
        if class_id is not None:
            class_counts[class_id] += 1

# Sort by count and get top 50 with 150+ samples
sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
top_50_classes = [(cls_id, count) for cls_id, count in sorted_classes if count >= 150][:50]

print(f"\n✅ Found {len(top_50_classes)} classes with 150+ samples")
print(f"   Range: {top_50_classes[-1][1]} - {top_50_classes[0][1]} samples")

# Create new class mapping (0-49)
old_to_new = {old_id: new_id for new_id, (old_id, _) in enumerate(top_50_classes)}
selected_class_ids = set(old_to_new.keys())

print(f"\n🎯 Selected classes:")
for old_id, count in top_50_classes[:10]:
    word = None
    for word_label, idx in class_mapping.items():
        if idx == old_id:
            word = word_label.split('\t')[1] if '\t' in word_label else word_label
            break
    print(f"   {word:15s}: {count:3d} samples")
print(f"   ... and {len(top_50_classes) - 10} more")

# Process function
def process_video_landmarks(frames, n_timesteps=60, n_features=171):
    """Extract and normalize landmarks from video frames"""
    if not frames or len(frames) == 0:
        return None, None
    
    class_id = frames[0].get('class_id')
    if class_id not in selected_class_ids:
        return None, None
    
    # Extract landmarks
    landmarks_sequence = []
    for frame_data in frames:
        landmarks = frame_data.get('landmarks', [])
        if len(landmarks) != n_features:
            return None, None
        landmarks_sequence.append(landmarks)
    
    if len(landmarks_sequence) == 0:
        return None, None
    
    # Pad or truncate to n_timesteps
    landmarks_array = np.array(landmarks_sequence)
    
    if len(landmarks_array) < n_timesteps:
        # Pad with last frame
        padding = np.repeat([landmarks_array[-1]], n_timesteps - len(landmarks_array), axis=0)
        landmarks_array = np.vstack([landmarks_array, padding])
    elif len(landmarks_array) > n_timesteps:
        # Sample evenly
        indices = np.linspace(0, len(landmarks_array) - 1, n_timesteps, dtype=int)
        landmarks_array = landmarks_array[indices]
    
    return landmarks_array, old_to_new[class_id]

# Process all datasets
print("\n🔄 Processing datasets...")
n_timesteps = 60
n_features = 171

def process_dataset(landmarks_dict, name):
    X_list, y_list = [], []
    valid, invalid = 0, 0
    
    for video_id, frames in landmarks_dict.items():
        X, y = process_video_landmarks(frames, n_timesteps, n_features)
        if X is not None and y is not None:
            X_list.append(X)
            y_list.append(y)
            valid += 1
        else:
            invalid += 1
    
    print(f"   {name:10s}: {valid:5d} valid, {invalid:5d} invalid")
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)

X_train, y_train = process_dataset(train_landmarks, "Train")
X_val, y_val = process_dataset(val_landmarks, "Val")
X_test, y_test = process_dataset(test_landmarks, "Test")

print(f"\n✅ Final shapes:")
print(f"   Train: {X_train.shape}, {y_train.shape}")
print(f"   Val:   {X_val.shape}, {y_val.shape}")
print(f"   Test:  {X_test.shape}, {y_test.shape}")

# Save data
print("\n💾 Saving processed data...")
np.savez_compressed(f"{OUTPUT_DIR}/train_data.npz", X=X_train, y=y_train)
np.savez_compressed(f"{OUTPUT_DIR}/val_data.npz", X=X_val, y=y_val)
np.savez_compressed(f"{OUTPUT_DIR}/test_data.npz", X=X_test, y=y_test)

# Create metadata with label_to_idx
label_to_idx = {}
for old_id, new_id in old_to_new.items():
    # Find word for this class
    for word_label, idx in class_mapping.items():
        if idx == old_id:
            label_to_idx[word_label] = new_id
            break

metadata = {
    "n_classes": 50,
    "n_features": n_features,
    "n_frames": n_timesteps,
    "label_to_idx": label_to_idx
}

with open(f"{OUTPUT_DIR}/metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Saved to {OUTPUT_DIR}/")
print(f"   - train_data.npz ({X_train.nbytes / 1024 / 1024:.1f} MB)")
print(f"   - val_data.npz ({X_val.nbytes / 1024 / 1024:.1f} MB)")
print(f"   - test_data.npz ({X_test.nbytes / 1024 / 1024:.1f} MB)")
print(f"   - metadata.json")

print("\n" + "=" * 70)
print("✅ DATASET PREPARATION COMPLETE!")
print("=" * 70)
