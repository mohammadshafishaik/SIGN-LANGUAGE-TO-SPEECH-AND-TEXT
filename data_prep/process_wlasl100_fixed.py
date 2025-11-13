#!/usr/bin/env python3
"""
Process WLASL-100 Landmarks Dataset (FIXED VERSION)
Correctly handles the actual JSON structure from Kaggle dataset
"""

import numpy as np
import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm

def load_class_names(classes_file):
    """Load the 100 class names from top_100_classes.txt"""
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
    print(f"📋 Loaded {len(classes)} class names")
    return classes

def load_json_landmarks(landmarks_file, metadata_file, class_names):
    """
    Load landmarks from JSON file and match with metadata labels
    
    Structure:
    - landmarks_file: dict[class_id] -> list of videos
    - metadata_file: dict[class_id] -> list of video paths
    - Each video: {'keyframes': int, 'landmarks': dict[frame_id] -> {pose, right, left}}
    
    Args:
        landmarks_file: Path to wasl100_landmarks_*.json (contains keypoints)
        metadata_file: Path to *_100.json (maps class_id to video paths)
        class_names: List of 100 class names in order
    
    Returns:
        X: List of landmark arrays
        y: List of class labels (strings)
    """
    print(f"📂 Loading landmarks: {landmarks_file.name}")
    print(f"📂 Loading metadata: {metadata_file.name}")
    
    # Load landmark data: dict[class_id] -> list[video]
    with open(landmarks_file, 'r') as f:
        landmark_data = json.load(f)
    
    # Load metadata: dict[class_id] -> list[video_path]
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Create mapping: class_id -> class_name
    # The metadata keys are sorted class IDs like '1', '2', '4'...
    sorted_class_ids = sorted(metadata.keys(), key=int)
    class_id_to_name = {}
    for idx, class_id_str in enumerate(sorted_class_ids):
        class_id_to_name[class_id_str] = class_names[idx]
    
    # Count total videos
    total_videos = sum(len(videos) for videos in landmark_data.values())
    print(f"📋 Processing {total_videos} videos across {len(metadata)} classes...")
    
    X = []
    y = []
    class_counts = Counter()
    
    # Process each class
    for class_id_str in tqdm(sorted_class_ids, desc="Processing classes"):
        class_name = class_id_to_name[class_id_str]
        
        # Get videos for this class from landmarks file
        if class_id_str not in landmark_data:
            print(f"⚠️  Warning: Class ID {class_id_str} not found in landmarks data")
            continue
        
        videos = landmark_data[class_id_str]
        
        # Process each video for this class
        for video in videos:
            try:
                # Extract landmarks from all frames
                landmarks_dict = video['landmarks']
                num_frames = video['keyframes']
                
                # Collect all frame data
                frame_data = []
                for frame_id in sorted(landmarks_dict.keys(), key=int):
                    frame = landmarks_dict[frame_id]
                    
                    # Concatenate pose (15), right hand (21), left hand (21) landmarks
                    pose = np.array(frame['pose'])      # (15, 3)
                    right = np.array(frame['right'])     # (21, 3)
                    left = np.array(frame['left'])       # (21, 3)
                    
                    # Flatten and concatenate: total = (15 + 21 + 21) * 3 = 171 features
                    frame_features = np.concatenate([
                        pose.flatten(),
                        right.flatten(),
                        left.flatten()
                    ])
                    
                    frame_data.append(frame_features)
                
                if len(frame_data) > 0:
                    X.append(np.array(frame_data))  # Shape: (num_frames, 171)
                    y.append(class_name)
                    class_counts[class_name] += 1
                
            except Exception as e:
                print(f"⚠️  Error processing video for class {class_name}: {e}")
                continue
    
    print(f"   ✅ Loaded {len(X)} samples across {len(set(y))} classes")
    if len(class_counts) > 0:
        print(f"   📊 Samples per class: min={min(class_counts.values())}, max={max(class_counts.values())}, avg={np.mean(list(class_counts.values())):.1f}")
    
    return X, y

def normalize_landmarks(X, target_frames=60):
    """Normalize landmark sequences to fixed length - optimized version"""
    print(f"🔧 Normalizing {len(X)} sequences to {target_frames} frames...")
    
    X_normalized = []
    
    for i, landmarks in enumerate(X):
        if i % 1000 == 0:
            print(f"   Progress: {i}/{len(X)}")
        
        frames = len(landmarks)
        
        if frames == target_frames:
            normalized = landmarks
        elif frames > target_frames:
            # Downsample - simple indexing
            indices = np.linspace(0, frames - 1, target_frames, dtype=int)
            normalized = landmarks[indices]
        else:
            # Upsample - repeat frames
            normalized = np.repeat(landmarks, int(np.ceil(target_frames / frames)), axis=0)[:target_frames]
        
        X_normalized.append(normalized)
    
    print(f"   ✅ Normalized {len(X_normalized)} sequences")
    return np.array(X_normalized)

def create_label_mapping(y):
    """Create label to index mapping"""
    unique_labels = sorted(set(y))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    y_encoded = np.array([label_to_idx[label] for label in y])
    
    return y_encoded, label_to_idx, idx_to_label

def process_dataset():
    """Main processing function"""
    print("=" * 60)
    print("WLASL-100 Dataset Processing (FIXED)")
    print("=" * 60)
    
    # Paths
    data_dir = Path("/Users/shaikshafi/Documents/ML PROJECT/datasets_local.nosync")
    output_dir = Path("/Users/shaikshafi/Documents/ML PROJECT/dataset/wlasl_100_processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load class names
    class_names = load_class_names(data_dir / "top_100_classes.txt")
    
    # Process training data
    print("\n" + "=" * 60)
    print("TRAINING DATA")
    print("=" * 60)
    train_X, train_y = load_json_landmarks(
        data_dir / "wasl100_landmarks_train.json",
        data_dir / "train_100.json",
        class_names
    )
    
    # Process validation data
    print("\n" + "=" * 60)
    print("VALIDATION DATA")
    print("=" * 60)
    val_X, val_y = load_json_landmarks(
        data_dir / "wasl100_landmarks_val.json",
        data_dir / "val_100.json",
        class_names
    )
    
    # Process test data
    print("\n" + "=" * 60)
    print("TEST DATA")
    print("=" * 60)
    test_X, test_y = load_json_landmarks(
        data_dir / "wasl100_landmarks_test.json",
        data_dir / "test_100.json",
        class_names
    )
    
    # Normalize to 60 frames
    print("\n" + "=" * 60)
    print("NORMALIZATION")
    print("=" * 60)
    train_X = normalize_landmarks(train_X, target_frames=60)
    val_X = normalize_landmarks(val_X, target_frames=60)
    test_X = normalize_landmarks(test_X, target_frames=60)
    
    print(f"   ✅ Training set: {train_X.shape}")
    print(f"   ✅ Validation set: {val_X.shape}")
    print(f"   ✅ Test set: {test_X.shape}")
    
    # Create label mappings
    print("\n" + "=" * 60)
    print("LABEL ENCODING")
    print("=" * 60)
    
    # Combine all labels to create consistent mapping
    all_labels = train_y + val_y + test_y
    unique_labels = sorted(set(all_labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    train_y_encoded = np.array([label_to_idx[label] for label in train_y])
    val_y_encoded = np.array([label_to_idx[label] for label in val_y])
    test_y_encoded = np.array([label_to_idx[label] for label in test_y])
    
    print(f"   ✅ {len(unique_labels)} unique classes encoded")
    print(f"   📋 Sample mappings:")
    for i, (label, idx) in enumerate(list(label_to_idx.items())[:5]):
        print(f"      {idx}: {label}")
    
    # Save processed data
    print("\n" + "=" * 60)
    print("SAVING")
    print("=" * 60)
    
    np.savez_compressed(
        output_dir / 'train_data.npz',
        X=train_X,
        y=train_y_encoded,
        labels=train_y
    )
    print(f"   ✅ Saved: train_data.npz ({train_X.shape[0]} samples)")
    
    np.savez_compressed(
        output_dir / 'val_data.npz',
        X=val_X,
        y=val_y_encoded,
        labels=val_y
    )
    print(f"   ✅ Saved: val_data.npz ({val_X.shape[0]} samples)")
    
    np.savez_compressed(
        output_dir / 'test_data.npz',
        X=test_X,
        y=test_y_encoded,
        labels=test_y
    )
    print(f"   ✅ Saved: test_data.npz ({test_X.shape[0]} samples)")
    
    # Save label mappings
    metadata = {
        'n_classes': len(unique_labels),
        'n_features': train_X.shape[2],
        'n_frames': train_X.shape[1],
        'n_train': len(train_X),
        'n_val': len(val_X),
        'n_test': len(test_X),
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'class_names': unique_labels
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Saved: metadata.json")
    
    print("\n" + "=" * 60)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"📊 Dataset Summary:")
    print(f"   Training:   {train_X.shape[0]:,} samples")
    print(f"   Validation: {val_X.shape[0]:,} samples")
    print(f"   Test:       {test_X.shape[0]:,} samples")
    print(f"   Classes:    {len(unique_labels)}")
    print(f"   Features:   {train_X.shape[2]} per frame")
    print(f"   Frames:     {train_X.shape[1]} per sample")
    print(f"\n📁 Output directory: {output_dir}")

if __name__ == "__main__":
    process_dataset()
