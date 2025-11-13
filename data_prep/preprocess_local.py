
import os
import sys
import argparse
from tqdm import tqdm
import pandas as pd

# Add the parent directory to the path to allow importing pose_extractor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pose_extractor.mediapipe_extractor import MediaPipeExtractor

# --- Configuration ---
# Directory where your locally recorded videos are
LOCAL_RAW_DIR = "dataset/raw"
# Directory to save the processed keypoints
OUTPUT_DIR = "dataset/keypoints_isl"
# Metadata file for the local dataset
METADATA_CSV = os.path.join(OUTPUT_DIR, "metadata.csv")

# --- Main Execution ---

def main():
    """
    Processes locally recorded videos using the existing MediaPipeExtractor.
    This script is a wrapper around the logic in `pose_extractor.main`.
    """
    print("Starting preprocessing of local ISL/Telugu dataset...")

    # Check if the raw data directory exists
    if not os.path.isdir(LOCAL_RAW_DIR) or not os.listdir(LOCAL_RAW_DIR):
        print(f"Warning: Raw data directory '{LOCAL_RAW_DIR}' is empty or does not exist.")
        print("Please record data using 'data_collector/collect.py' first.")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # We can reuse the main processing loop from the original extractor script,
    # but we need to adapt it to our directory structure and metadata needs.
    
    # Find all video files in the raw directory
    video_files = [f for f in os.listdir(LOCAL_RAW_DIR) if f.endswith(('.mp4', '.mov', '.avi'))]
    
    if not video_files:
        print(f"No video files found in {LOCAL_RAW_DIR}.")
        return

    print(f"Found {len(video_files)} videos to process.")

    # Initialize the extractor
    extractor = MediaPipeExtractor()
    
    metadata_records = []

    for video_file in tqdm(video_files, desc="Processing Local Videos"):
        video_path = os.path.join(LOCAL_RAW_DIR, video_file)
        
        # The label is typically part of the filename, e.g., "hello_01.mp4"
        label = video_file.split('_')[0]
        video_id = os.path.splitext(video_file)[0]

        # Use the same processing function as in the WLASL script for consistency
        from data_prep.preprocess_wlasl import process_video_file
        
        result = process_video_file(video_path, label, extractor)
        
        if result:
            # Update the source and save
            result['source'] = 'ISL' # Or 'Telugu', or just 'local'
            output_path = os.path.join(OUTPUT_DIR, f"{video_id}.npz")
            np.savez_compressed(output_path, data=result['features'], label=result['label'])
            
            metadata_records.append({
                'video_id': video_id,
                'label': result['label'],
                'frames': result['frames'],
                'fps': result['fps'],
                'source': result['source']
            })

    extractor.close()

    if not metadata_records:
        print("No local videos were processed successfully.")
        return

    # Save metadata
    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(METADATA_CSV, index=False)

    print(f"Preprocessing of local data complete. {len(metadata_records)} videos processed.")
    print(f"Keypoints saved to: {OUTPUT_DIR}")
    print(f"Metadata saved to: {METADATA_CSV}")

if __name__ == "__main__":
    # This script needs access to `preprocess_wlasl` which has numpy, so we import it here
    # to avoid lint errors at the top level before the path is modified.
    import numpy as np
    main()
