import os
import pandas as pd
import argparse
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Main Execution ---

def main(args):
    """
    Splits the combined dataset into training, validation, and test sets.
    """
    print("Splitting dataset into train, validation, and test sets...")
    
    metadata_path = os.path.join(args.input, "metadata_combined.csv")
    if not os.path.exists(metadata_path):
        print(f"Error: Combined metadata not found at {metadata_path}")
        return
        
    df = pd.read_csv(metadata_path)
    
    # Ensure ratios sum to 1
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total_ratio, 1.0):
        print(f"Error: Ratios must sum to 1.0, but got {total_ratio}")
        return

    # Create output directories
    train_dir = os.path.join(args.input, 'train')
    val_dir = os.path.join(args.input, 'val')
    test_dir = os.path.join(args.input, 'test')
    
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    # Stratified split to maintain class distribution
    labels = df['label_encoded']
    
    try:
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=args.test_ratio,
            random_state=42,
            stratify=df['label_encoded']
        )
        
        # Second split: separate train and validation sets
        # Adjust validation ratio for the remaining data
        val_ratio_adjusted = args.val_ratio / (args.train_ratio + args.val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            random_state=42,
            stratify=train_val_df['label_encoded']
        )
    except ValueError as e:
        print(f"\nWARNING: Could not perform stratified split. This is likely because a class has too few samples (error: {e}).")
        print("Proceeding with a non-stratified split. Class distribution may be uneven.")
        
        # Fallback to non-stratified split
        train_val_df, test_df = train_test_split(
            df,
            test_size=args.test_ratio,
            random_state=42
        )
        val_ratio_adjusted = args.val_ratio / (args.train_ratio + args.val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            random_state=42
        )

    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    # Function to copy files
    def copy_files(dataframe, destination_dir):
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"Copying to {os.path.basename(destination_dir)}"):
            src_path = os.path.join(args.input, f"{row['unified_video_id']}.npz")
            dst_path = os.path.join(destination_dir, f"{row['unified_video_id']}.npz")
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
    
    # Copy files to their respective directories
    copy_files(train_df, train_dir)
    copy_files(val_df, val_dir)
    copy_files(test_df, test_dir)
    
    # Save the split metadata files
    train_df.to_csv(os.path.join(train_dir, 'metadata.csv'), index=False)
    val_df.to_csv(os.path.join(val_dir, 'metadata.csv'), index=False)
    test_df.to_csv(os.path.join(test_dir, 'metadata.csv'), index=False)
    
    print("\nDataset splitting complete.")
    print(f"Train, val, and test sets created in: {args.input}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train, validation, and test sets.")
    parser.add_argument('--input', type=str, default='dataset/keypoints_combined', help='Directory of the combined dataset.')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Proportion of data for training.')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Proportion of data for validation.')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Proportion of data for testing.')
    
    # Need numpy for isclose
    import numpy as np
    args = parser.parse_args()
    main(args)
