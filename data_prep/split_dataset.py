"""
Split dataset into train/val/test with stratification.

Usage:
    python data_prep/split_dataset.py --input_dir dataset/keypoints_wlasl/ --output_dir dataset/
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
import shutil

def main(args):
    print("="*60)
    print("DATASET SPLITTING - STRATIFIED TRAIN/VAL/TEST")
    print("="*60)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Load metadata
    metadata_path = input_dir / 'metadata.csv'
    df = pd.read_csv(metadata_path)
    
    print(f"\n📊 Total samples: {len(df)}")
    print(f"📊 Unique labels: {df['label'].nunique()}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts().to_string())
    
    # Filter out classes with too few samples (need at least 6 for 70/15/15 split)
    label_counts = df['label'].value_counts()
    valid_labels = label_counts[label_counts >= 6].index
    df_filtered = df[df['label'].isin(valid_labels)]
    
    if len(df_filtered) < len(df):
        removed = len(df) - len(df_filtered)
        print(f"\n⚠️  Removed {removed} samples from classes with < 6 samples (needed for stratification)")
        print(f"📊 Filtered dataset: {len(df_filtered)} samples, {df_filtered['label'].nunique()} labels")
    
    # Stratified split: 70% train, 15% val, 15% test
    # First split: 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df_filtered, 
        test_size=0.3, 
        stratify=df_filtered['label'], 
        random_state=42
    )
    
    # Second split: 50% of temp = 15% val, 15% test
    try:
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            stratify=temp_df['label'],
            random_state=42
        )
    except ValueError:
        # If stratification fails, do random split
        print("⚠️  Stratification failed for val/test split, using random split")
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42
        )
    
    # Add split column
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    # Combine and save
    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    output_path = output_dir / 'metadata_split.csv'
    final_df.to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print("SPLIT COMPLETE")
    print("="*60)
    print(f"🎯 Train: {len(train_df)} samples ({len(train_df)/len(df_filtered)*100:.1f}%)")
    print(f"🎯 Val:   {len(val_df)} samples ({len(val_df)/len(df_filtered)*100:.1f}%)")
    print(f"🎯 Test:  {len(test_df)} samples ({len(test_df)/len(df_filtered)*100:.1f}%)")
    print(f"\n📄 Saved to: {output_path}")
    
    # Print per-label split
    print(f"\nPer-label split:")
    for label in sorted(final_df['label'].unique()):
        train_count = len(train_df[train_df['label'] == label])
        val_count = len(val_df[val_df['label'] == label])
        test_count = len(test_df[test_df['label'] == label])
        total = train_count + val_count + test_count
        print(f"  {label:15s}: train={train_count:2d}, val={val_count:2d}, test={test_count:2d}, total={total:2d}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset with stratification")
    parser.add_argument('--input_dir', type=str, default='dataset/keypoints_wlasl/',
                        help='Directory containing processed keypoints')
    parser.add_argument('--output_dir', type=str, default='dataset/',
                        help='Output directory for metadata')
    
    args = parser.parse_args()
    main(args)
