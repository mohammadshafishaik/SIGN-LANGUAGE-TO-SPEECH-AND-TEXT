import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from scipy.interpolate import interp1d
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
TARGET_FPS = 25
TARGET_SEQ_LENGTH = 100 # Max number of frames per video, can be adjusted
JOINT_ORDER_MAP = {
    # This is a placeholder. You MUST define the mapping from each dataset's
    # joint indices to a common, unified order.
    # For example, if using MediaPipe's 75 landmarks as the standard:
    'common': list(range(75)),
    'WLASL': list(range(75)), # Already in MediaPipe format
    'SignAvatars': list(range(75)), # ASSUMPTION: needs verification
    'ISL': list(range(75)) # Already in MediaPipe format
}

# --- Helper Functions ---

def resample_sequence(sequence, original_fps, target_fps):
    """Resamples a sequence to a target FPS."""
    if original_fps == target_fps:
        return sequence
    
    num_frames = sequence.shape[0]
    original_time = np.linspace(0, num_frames / original_fps, num_frames)
    target_time = np.linspace(0, num_frames / original_fps, int(num_frames * target_fps / original_fps))
    
    resampled = np.zeros((len(target_time), sequence.shape[1], sequence.shape[2]))
    
    for i in range(sequence.shape[1]): # For each joint
        for j in range(sequence.shape[2]): # For each coordinate (x, y, z, dx, dy, dz)
            interpolator = interp1d(original_time, sequence[:, i, j], kind='linear', fill_value='extrapolate')
            resampled[:, i, j] = interpolator(target_time)
            
    return resampled

def pad_or_truncate(sequence, target_length):
    """Pads with zeros or truncates a sequence to a target length."""
    if sequence.shape[0] > target_length:
        return sequence[:target_length]
    else:
        padding = np.zeros((target_length - sequence.shape[0], sequence.shape[1], sequence.shape[2]))
        return np.concatenate([sequence, padding], axis=0)

def remap_joints(sequence, source_dataset):
    """Remaps joints to a common order."""
    # This is where you'd use JOINT_ORDER_MAP if datasets had different skeletons
    # For now, we assume they are all compatible.
    return sequence

# --- Main Execution ---

def main(args):
    """
    Merges multiple preprocessed datasets into a single, unified dataset.
    """
    print("Starting dataset merging process...")
    os.makedirs(args.output, exist_ok=True)
    
    all_features = []
    all_metadata = []
    
    for input_dir in args.inputs:
        source_name = os.path.basename(input_dir).split('_')[-1].upper()
        print(f"Processing source: {source_name}")
        
        metadata_path = os.path.join(input_dir, "metadata.csv")
        if not os.path.exists(metadata_path):
            print(f"Warning: Metadata for {source_name} not found. Skipping.")
            continue
            
        metadata_df = pd.read_csv(metadata_path)
        
        for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc=f"Merging {source_name}"):
            npz_path = os.path.join(input_dir, f"{row['video_id']}.npz")
            if not os.path.exists(npz_path):
                continue
            
            with np.load(npz_path) as npz_file:
                features = npz_file['data']
            
            # 1. Resample to target FPS
            resampled = resample_sequence(features, row['fps'], TARGET_FPS)
            
            # 2. Remap joints to common skeleton (if necessary)
            remapped = remap_joints(resampled, source_name)
            
            # 3. Pad or truncate to target sequence length
            final_sequence = pad_or_truncate(remapped, TARGET_SEQ_LENGTH)
            
            # Save the processed file to the combined directory
            output_npz_path = os.path.join(args.output, f"{row['source']}_{row['video_id']}.npz")
            np.savez_compressed(output_npz_path, data=final_sequence, label=row['label'])
            
            # Append metadata
            new_row = row.to_dict()
            new_row['unified_video_id'] = f"{row['source']}_{row['video_id']}"
            all_metadata.append(new_row)

    # Create and save the combined metadata
    combined_metadata_df = pd.DataFrame(all_metadata)
    
    # Encode labels to integers
    le = LabelEncoder()
    combined_metadata_df['label_encoded'] = le.fit_transform(combined_metadata_df['label'])
    
    # Save the label encoder for later use in inference
    np.save(os.path.join(args.output, 'label_encoder.npy'), le.classes_)
    
    combined_metadata_path = os.path.join(args.output, "metadata_combined.csv")
    combined_metadata_df.to_csv(combined_metadata_path, index=False)
    
    print("\nDataset merging complete.")
    print(f"Total samples: {len(combined_metadata_df)}")
    print(f"Number of unique classes: {len(le.classes_)}")
    print(f"Unified keypoints saved to: {args.output}")
    print(f"Combined metadata saved to: {combined_metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and standardize multiple sign language keypoint datasets.")
    parser.add_argument('--inputs', nargs='+', required=True, help='List of input directories for each dataset (e.g., dataset/keypoints_wlasl).')
    parser.add_argument('--output', type=str, default='dataset/keypoints_combined', help='Directory to save the unified dataset.')
    
    args = parser.parse_args()
    
    # Filter out the signavatars path if it was passed by mistake
    args.inputs = [d for d in args.inputs if 'signavatars' not in d]
    
    main(args)
