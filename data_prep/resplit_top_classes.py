"""
Re-split dataset keeping only top N classes with most samples.
This increases samples per class for better training.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

def main(top_n=5):
    # Load original metadata
    df = pd.read_csv('dataset/keypoints_wlasl/metadata.csv')
    
    # Get top N classes by count
    top_classes = df['label'].value_counts().head(top_n).index.tolist()
    
    print(f"\n{'='*60}")
    print(f"KEEPING TOP {top_n} CLASSES")
    print(f"{'='*60}")
    
    # Filter to top classes
    df_filtered = df[df['label'].isin(top_classes)]
    
    print(f"\nOriginal dataset: {len(df)} samples, {df['label'].nunique()} classes")
    print(f"Filtered dataset: {len(df_filtered)} samples, {df_filtered['label'].nunique()} classes")
    
    print(f"\nClass distribution:")
    for label, count in df_filtered['label'].value_counts().items():
        print(f"  {label:15s}: {count} samples")
    
    # Split: 70/15/15
    train_df, temp_df = train_test_split(
        df_filtered,
        test_size=0.3,
        stratify=df_filtered['label'],
        random_state=42
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['label'],
        random_state=42
    )
    
    # Add split column
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    # Combine
    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    final_df.to_csv('dataset/metadata_split.csv', index=False)
    
    print(f"\n{'='*60}")
    print("SPLIT COMPLETE")
    print(f"{'='*60}")
    print(f"Train: {len(train_df)} samples ({len(train_df)/len(df_filtered)*100:.1f}%)")
    print(f"Val:   {len(val_df)} samples ({len(val_df)/len(df_filtered)*100:.1f}%)")
    print(f"Test:  {len(test_df)} samples ({len(test_df)/len(df_filtered)*100:.1f}%)")
    
    print(f"\nPer-class breakdown:")
    for label in sorted(final_df['label'].unique()):
        train_count = len(train_df[train_df['label'] == label])
        val_count = len(val_df[val_df['label'] == label])
        test_count = len(test_df[test_df['label'] == label])
        total = train_count + val_count + test_count
        print(f"  {label:15s}: train={train_count:2d}, val={val_count:2d}, test={test_count:2d}, total={total:2d}")
    
    print(f"\nAverage samples per class in training: {len(train_df)/train_df['label'].nunique():.1f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_n', type=int, default=5, help='Keep top N classes')
    args = parser.parse_args()
    main(args.top_n)
