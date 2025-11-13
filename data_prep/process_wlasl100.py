#!/usr/bin/env python3
"""
Process WLASL Dataset with MediaPipe Landmarks
Prepare data for training 100-word ASL recognition model
"""

import numpy as np
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import pickle

def load_mediapipe_data(data_dir):
    """
    Load MediaPipe landmarks from WLASL dataset
    """
    print("🔍 Loading MediaPipe landmarks from WLASL dataset...")
    
    data_path = Path(data_dir)
    
    # Find all .npy or .pkl files
    landmark_files = list(data_path.rglob("*.npy")) + list(data_path.rglob("*.pkl"))
    
    print(f"📦 Found {len(landmark_files)} landmark files")
    
    samples = []
    labels = []
    word_counts = Counter()
    
    for file_path in landmark_files:
        try:
            # Extract word/label from filename
            # Format: word_videoID.npy or similar
            filename = file_path.stem
            parts = filename.split('_')
            word = parts[0].lower()  # First part is usually the word
            
            # Load landmarks
            if file_path.suffix == '.npy':
                landmarks = np.load(file_path)
            elif file_path.suffix == '.pkl':
                with open(file_path, 'rb') as f:
                    landmarks = pickle.load(f)
            else:
                continue
            
            # Validate landmarks shape
            if landmarks is None or len(landmarks) == 0:
                continue
            
            # Ensure 3D array: (frames, landmarks, coordinates)
            if landmarks.ndim == 2:
                # If 2D, assume (landmarks, coordinates) for single frame
                landmarks = landmarks[np.newaxis, ...]
            
            samples.append(landmarks)
            labels.append(word)
            word_counts[word] += 1
            
        except Exception as e:
            # Skip problematic files
            continue
    
    print(f"\n✅ Loaded {len(samples)} samples")
    print(f"📊 Found {len(word_counts)} unique words")
    
    return samples, labels, word_counts

def select_top_100_words(samples, labels, word_counts, n_words=100, min_samples=10):
    """
    Select top 100 words with most samples
    """
    print(f"\n🎯 Selecting top {n_words} words...")
    
    # Filter words with minimum samples
    valid_words = {word: count for word, count in word_counts.items() 
                   if count >= min_samples}
    
    print(f"📊 Words with >={min_samples} samples: {len(valid_words)}")
    
    # Get top N words
    top_words = [word for word, _ in sorted(valid_words.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True)[:n_words]]
    
    # Filter samples and labels
    filtered_samples = []
    filtered_labels = []
    
    for sample, label in zip(samples, labels):
        if label in top_words:
            filtered_samples.append(sample)
            filtered_labels.append(label)
    
    # Create label mapping
    word_to_idx = {word: idx for idx, word in enumerate(sorted(top_words))}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    # Convert labels to indices
    numeric_labels = [word_to_idx[label] for label in filtered_labels]
    
    print(f"\n✅ Selected {len(top_words)} words")
    print(f"📊 Total samples: {len(filtered_samples)}")
    print(f"\nTop 20 words by sample count:")
    for word in top_words[:20]:
        count = sum(1 for l in filtered_labels if l == word)
        print(f"  {word}: {count} samples")
    
    return filtered_samples, numeric_labels, word_to_idx, idx_to_word

def normalize_sequence_length(sequences, max_length=60):
    """
    Normalize all sequences to same length
    """
    print(f"\n🔧 Normalizing sequence lengths to {max_length} frames...")
    
    normalized = []
    
    for seq in sequences:
        seq_len = len(seq)
        
        if seq_len > max_length:
            # Downsample
            indices = np.linspace(0, seq_len - 1, max_length, dtype=int)
            normalized.append(seq[indices])
        elif seq_len < max_length:
            # Pad with zeros
            padding = np.zeros((max_length - seq_len, *seq.shape[1:]))
            normalized.append(np.vstack([seq, padding]))
        else:
            normalized.append(seq)
    
    return np.array(normalized)

def create_train_val_test_splits(X, y, test_size=0.15, val_size=0.15):
    """
    Create train/val/test splits
    """
    print("\n📂 Creating train/val/test splits...")
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def save_processed_data(output_dir, train_data, val_data, test_data, 
                        word_to_idx, idx_to_word):
    """
    Save processed data to disk
    """
    print(f"\n💾 Saving processed data to {output_dir}...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    np.savez_compressed(
        output_path / 'train_data.npz',
        X=train_data[0],
        y=train_data[1]
    )
    
    np.savez_compressed(
        output_path / 'val_data.npz',
        X=val_data[0],
        y=val_data[1]
    )
    
    np.savez_compressed(
        output_path / 'test_data.npz',
        X=test_data[0],
        y=test_data[1]
    )
    
    # Save label mappings
    with open(output_path / 'word_to_idx.json', 'w') as f:
        json.dump(word_to_idx, f, indent=2)
    
    with open(output_path / 'idx_to_word.json', 'w') as f:
        json.dump({str(k): v for k, v in idx_to_word.items()}, f, indent=2)
    
    # Save metadata
    metadata = {
        'n_classes': len(word_to_idx),
        'n_train': len(train_data[0]),
        'n_val': len(val_data[0]),
        'n_test': len(test_data[0]),
        'sequence_length': train_data[0].shape[1],
        'n_features': train_data[0].shape[2],
        'words': sorted(word_to_idx.keys())
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved:")
    print(f"  - train_data.npz ({metadata['n_train']} samples)")
    print(f"  - val_data.npz ({metadata['n_val']} samples)")
    print(f"  - test_data.npz ({metadata['n_test']} samples)")
    print(f"  - word_to_idx.json ({metadata['n_classes']} classes)")
    print(f"  - metadata.json")
    
    return metadata

def main():
    # Configuration
    data_dir = "datasets_local.nosync/mutemotion-output"  # Downloaded WLASL MediaPipe data (LOCAL STORAGE)
    output_dir = "dataset/wlasl_100_processed"
    n_words = 100
    max_sequence_length = 60
    min_samples_per_word = 10
    
    print("=" * 70)
    print("🚀 WLASL-100 Data Processing Pipeline")
    print("=" * 70)
    print(f"📁 Input directory: {data_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Target words: {n_words}")
    print(f"📏 Max sequence length: {max_sequence_length}")
    print(f"📊 Min samples per word: {min_samples_per_word}")
    print("=" * 70)
    
    # Step 1: Load data
    samples, labels, word_counts = load_mediapipe_data(data_dir)
    
    if len(samples) == 0:
        print("\n❌ No data found! Please check:")
        print(f"  1. Dataset downloaded to: {data_dir}")
        print(f"  2. Files are .npy or .pkl format")
        print(f"  3. Directory structure is correct")
        return
    
    # Step 2: Select top 100 words
    samples, labels, word_to_idx, idx_to_word = select_top_100_words(
        samples, labels, word_counts, n_words, min_samples_per_word
    )
    
    # Step 3: Normalize sequence lengths
    X = normalize_sequence_length(samples, max_sequence_length)
    y = np.array(labels)
    
    print(f"\n📊 Final dataset shape: {X.shape}")
    print(f"   - Samples: {X.shape[0]}")
    print(f"   - Sequence length: {X.shape[1]}")
    print(f"   - Features: {X.shape[2]}")
    
    # Step 4: Create splits
    train_data, val_data, test_data = create_train_val_test_splits(X, y)
    
    # Step 5: Save processed data
    metadata = save_processed_data(
        output_dir, train_data, val_data, test_data,
        word_to_idx, idx_to_word
    )
    
    print("\n" + "=" * 70)
    print("✅ DATA PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"📊 Dataset Summary:")
    print(f"   - Classes: {metadata['n_classes']} words")
    print(f"   - Training samples: {metadata['n_train']}")
    print(f"   - Validation samples: {metadata['n_val']}")
    print(f"   - Test samples: {metadata['n_test']}")
    print(f"   - Sequence length: {metadata['sequence_length']}")
    print(f"   - Features per frame: {metadata['n_features']}")
    print("\n🚀 Ready to train! Run:")
    print(f"   python models/train_wlasl100.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
