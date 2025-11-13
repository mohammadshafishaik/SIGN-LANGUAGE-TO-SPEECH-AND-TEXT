#!/usr/bin/env python3
"""
Process WLASL-100 Landmarks Dataset
Convert JSON format to numpy arrays for training
"""

import numpy as np
import json
from pathlib import Path
from collections import Counter
import pickle

def load_json_landmarks(json_file):
    """Load landmarks from JSON file"""
    print(f"📂 Loading: {json_file.name}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    samples = []
    labels = []
    
    for entry in data:
        try:
            # Extract landmarks and label
            landmarks = np.array(entry['landmarks'])  # Shape: (frames, landmarks, 3)
            label = entry.get('gloss', entry.get('label', 'unknown'))
            
            if len(landmarks) > 0:
                samples.append(landmarks)
                labels.append(label)
        except Exception as e:
            continue
    
    print(f"   ✅ Loaded {len(samples)} samples")
    return samples, labels

def normalize_landmarks(landmarks, target_frames=60):
    """Normalize landmark sequences to fixed length"""
    frames = len(landmarks)
    
    if frames > target_frames:
        # Downsample
        indices = np.linspace(0, frames - 1, target_frames, dtype=int)
        return landmarks[indices]
    elif frames < target_frames:
        # Pad with zeros
        padding = np.zeros((target_frames - frames, *landmarks.shape[1:]))
        return np.vstack([landmarks, padding])
    else:
        return landmarks

def process_dataset(data_dir, output_dir, target_frames=60):
    """Process WLASL-100 dataset"""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("🚀 PROCESSING WLASL-100 DATASET")
    print("=" * 70)
    print(f"📁 Input: {data_path}")
    print(f"📁 Output: {output_path}")
    print(f"📏 Target frames: {target_frames}")
    print("=" * 70)
    
    # Load data
    print("\n📥 Loading datasets...")
    
    train_samples, train_labels = load_json_landmarks(
        data_path / "wasl100_landmarks_train.json"
    )
    val_samples, val_labels = load_json_landmarks(
        data_path / "wasl100_landmarks_val.json"
    )
    test_samples, test_labels = load_json_landmarks(
        data_path / "wasl100_landmarks_test.json"
    )
    
    # Create label mappings
    print("\n🏷️  Creating label mappings...")
    all_labels = set(train_labels + val_labels + test_labels)
    word_to_idx = {word: idx for idx, word in enumerate(sorted(all_labels))}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    print(f"   ✅ Found {len(word_to_idx)} unique words")
    
    # Convert labels to indices
    train_y = np.array([word_to_idx[label] for label in train_labels])
    val_y = np.array([word_to_idx[label] for label in val_labels])
    test_y = np.array([word_to_idx[label] for label in test_labels])
    
    # Normalize sequences
    print(f"\n🔧 Normalizing sequences to {target_frames} frames...")
    
    train_X = np.array([normalize_landmarks(seq, target_frames) for seq in train_samples])
    val_X = np.array([normalize_landmarks(seq, target_frames) for seq in val_samples])
    test_X = np.array([normalize_landmarks(seq, target_frames) for seq in test_samples])
    
    print(f"   ✅ Training set: {train_X.shape}")
    print(f"   ✅ Validation set: {val_X.shape}")
    print(f"   ✅ Test set: {test_X.shape}")
    
    # Save processed data
    print("\n💾 Saving processed data...")
    
    np.savez_compressed(
        output_path / 'train_data.npz',
        X=train_X,
        y=train_y
    )
    print("   ✅ Saved train_data.npz")
    
    np.savez_compressed(
        output_path / 'val_data.npz',
        X=val_X,
        y=val_y
    )
    print("   ✅ Saved val_data.npz")
    
    np.savez_compressed(
        output_path / 'test_data.npz',
        X=test_X,
        y=test_y
    )
    print("   ✅ Saved test_data.npz")
    
    # Save label mappings
    with open(output_path / 'word_to_idx.json', 'w') as f:
        json.dump(word_to_idx, f, indent=2)
    print("   ✅ Saved word_to_idx.json")
    
    with open(output_path / 'idx_to_word.json', 'w') as f:
        json.dump({str(k): v for k, v in idx_to_word.items()}, f, indent=2)
    print("   ✅ Saved idx_to_word.json")
    
    # Save metadata
    metadata = {
        'n_classes': len(word_to_idx),
        'n_train': len(train_X),
        'n_val': len(val_X),
        'n_test': len(test_X),
        'sequence_length': target_frames,
        'n_features': train_X.shape[2] * train_X.shape[3] if len(train_X.shape) > 3 else train_X.shape[2],
        'words': sorted(word_to_idx.keys())
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("   ✅ Saved metadata.json")
    
    # Print summary
    print("\n" + "=" * 70)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"📊 Dataset Summary:")
    print(f"   - Classes: {metadata['n_classes']} words")
    print(f"   - Training samples: {metadata['n_train']}")
    print(f"   - Validation samples: {metadata['n_val']}")
    print(f"   - Test samples: {metadata['n_test']}")
    print(f"   - Sequence length: {metadata['sequence_length']} frames")
    print(f"   - Features per frame: {metadata['n_features']}")
    
    print(f"\n📋 Sample words:")
    for i, word in enumerate(sorted(word_to_idx.keys())[:20]):
        print(f"   {i+1}. {word}")
    print(f"   ... and {len(word_to_idx) - 20} more!")
    
    print("\n🚀 Next step: Train the model!")
    print(f"   python models/train_wlasl100.py")
    print("=" * 70)
    
    return metadata

if __name__ == "__main__":
    data_dir = "datasets_local.nosync"
    output_dir = "dataset/wlasl_100_processed"
    
    metadata = process_dataset(data_dir, output_dir, target_frames=60)
