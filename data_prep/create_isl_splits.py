"""
Create stratified train/val/test splits for ISL image dataset
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

def create_isl_splits(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create stratified splits from processed ISL data
    
    Args:
        input_dir: Directory containing .npz files for each class
        output_dir: Directory to save split data
        train_ratio: Proportion for training (default 0.7)
        val_ratio: Proportion for validation (default 0.15)
        test_ratio: Proportion for testing (default 0.15)
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load all class data
    all_features = []
    all_labels = []
    class_names = []
    
    print(f"\n{'='*60}")
    print(f"CREATING TRAIN/VAL/TEST SPLITS FOR ISL")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split ratios: {train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%}")
    print(f"{'='*60}\n")
    
    # Load data from all .npz files
    npz_files = sorted(input_path.glob('*.npz'))
    
    for npz_file in npz_files:
        if npz_file.name == 'processing_stats.json':
            continue
            
        data = np.load(npz_file)
        features = data['features']
        labels = data['labels']
        
        all_features.append(features)
        all_labels.extend(labels)
        
        class_name = npz_file.stem
        class_names.append(class_name)
        
        print(f"✅ Loaded {class_name}: {len(features)} samples")
    
    # Concatenate all features
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.array(all_labels)
    
    print(f"\n📊 Total: {len(all_features)} samples across {len(class_names)} classes")
    
    # Create label to index mapping
    unique_labels = sorted(set(all_labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    # Convert labels to indices
    y_indices = np.array([label_to_idx[label] for label in all_labels])
    
    # First split: train + val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        all_features, y_indices,
        test_size=test_ratio,
        stratify=y_indices,
        random_state=42
    )
    
    # Second split: train vs val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio_adjusted,
        stratify=y_trainval,
        random_state=42
    )
    
    # Save splits
    print(f"\n💾 Saving splits...")
    
    np.savez_compressed(
        output_path / 'train.npz',
        features=X_train,
        labels=y_train
    )
    print(f"  ✅ Train: {len(X_train)} samples")
    
    np.savez_compressed(
        output_path / 'val.npz',
        features=X_val,
        labels=y_val
    )
    print(f"  ✅ Val: {len(X_val)} samples")
    
    np.savez_compressed(
        output_path / 'test.npz',
        features=X_test,
        labels=y_test
    )
    print(f"  ✅ Test: {len(X_test)} samples")
    
    # Save label mappings
    mappings = {
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'num_classes': len(unique_labels),
        'class_names': unique_labels
    }
    
    with open(output_path / 'label_mappings.json', 'w') as f:
        json.dump(mappings, f, indent=2)
    
    print(f"  ✅ Label mappings saved")
    
    # Print class distribution
    print(f"\n📈 Class distribution:")
    print(f"\n{'Class':<10} {'Train':<10} {'Val':<10} {'Test':<10}")
    print(f"{'-'*40}")
    
    for label in unique_labels[:10]:  # Show first 10 classes
        idx = label_to_idx[label]
        train_count = np.sum(y_train == idx)
        val_count = np.sum(y_val == idx)
        test_count = np.sum(y_test == idx)
        print(f"{label:<10} {train_count:<10} {val_count:<10} {test_count:<10}")
    
    if len(unique_labels) > 10:
        print(f"... ({len(unique_labels) - 10} more classes)")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print(f"SPLIT SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_features)}")
    print(f"Number of classes: {len(unique_labels)}")
    print(f"Feature dimensions: {all_features.shape[1]}")
    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)} ({len(X_train)/len(all_features)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} ({len(X_val)/len(all_features)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} ({len(X_test)/len(all_features)*100:.1f}%)")
    print(f"\nAverage samples per class:")
    print(f"  Train: {len(X_train)/len(unique_labels):.1f}")
    print(f"  Val:   {len(X_val)/len(unique_labels):.1f}")
    print(f"  Test:  {len(X_test)/len(unique_labels):.1f}")
    print(f"{'='*60}\n")
    
    return {
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'num_classes': len(unique_labels),
        'feature_dim': all_features.shape[1]
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create ISL train/val/test splits')
    parser.add_argument(
        '--input_dir',
        type=str,
        default='dataset/keypoints_isl',
        help='Directory containing processed keypoints'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='dataset/splits_isl',
        help='Directory to save splits'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='Test set ratio (default: 0.15)'
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"❌ Error: Ratios must sum to 1.0 (got {total_ratio})")
        exit(1)
    
    # Convert to absolute paths
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / args.input_dir
    output_dir = base_dir / args.output_dir
    
    # Create splits
    stats = create_isl_splits(
        input_dir,
        output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )
    
    print("\n✅ Splits created successfully!")
    print(f"Next step: python models/train_isl.py")
