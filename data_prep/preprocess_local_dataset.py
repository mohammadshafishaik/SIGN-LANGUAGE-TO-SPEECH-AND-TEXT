import os
import sys
import cv2
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

# Add project root to Python path to find pose_extractor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pose_extractor.mediapipe_extractor import MediaPipeExtractor

# --- Constants ---
MAX_FRAMES = 100
TARGET_FPS = 25

# --- Main Functions ---

def process_video_file(video_path, extractor, max_frames, target_fps):
    """
    Processes a single video file to extract keypoints.
    Returns a numpy array of keypoints or an empty array if processing fails.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return np.array([])

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(original_fps / target_fps) if original_fps > target_fps else 1

        all_keypoints = []
        frame_count = 0
        
        while cap.isOpened() and len(all_keypoints) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                keypoints = extractor.extract_keypoints(frame)
                if keypoints is not None:
                    all_keypoints.append(keypoints)
            
            frame_count += 1
        
        cap.release()

        if not all_keypoints:
            return np.array([])

        num_frames = len(all_keypoints)
        if num_frames < max_frames:
            last_frame_keypoints = all_keypoints[-1]
            padding = [last_frame_keypoints] * (max_frames - num_frames)
            all_keypoints.extend(padding)
        
        return np.array(all_keypoints)

    except Exception as e:
        print(f"\nERROR processing video {video_path}: {e}")
        return np.array([])

def main(args):
    """
    Main function to preprocess the local video dataset.
    """
    print("Starting local dataset preprocessing...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Ensure the input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        print("Please create this directory and place your local video files inside.")
        # Create the directory so the script can be run again later
        os.makedirs(args.input_dir, exist_ok=True)
        return

    # Initialize the MediaPipe extractor
    extractor = MediaPipeExtractor()
    
    processed_files = []
    
    video_files = [f for f in os.listdir(args.input_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

    if not video_files:
        print(f"No video files found in {args.input_dir}. Nothing to process.")
        return

    # Loop through the video files
    for video_file in tqdm(video_files, desc="Processing Local Videos"):
        video_path = os.path.join(args.input_dir, video_file)
        
        # Use the filename (without extension) as the label
        label = os.path.splitext(video_file)[0]
        
        # Process the video
        keypoints = process_video_file(video_path, extractor, MAX_FRAMES, TARGET_FPS)
        
        # Check if the returned keypoints are valid
        if keypoints is None or keypoints.shape[0] == 0:
            print(f"\nWARNING: No valid landmarks extracted from video, skipping: {video_path}")
            continue

        # Save keypoints
        output_filename = f"{label}.npz"
        output_path = os.path.join(args.output_dir, output_filename)
        np.savez_compressed(output_path, data=keypoints)
        
        processed_files.append({
            'video_id': label,
            'label': label,
            'source': 'local'
        })

    if not processed_files:
        print("No videos were successfully processed.")
        return

    # Create and save metadata
    metadata_df = pd.DataFrame(processed_files)
    metadata_path = os.path.join(args.output_dir, 'metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    
    print(f"\nPreprocessing complete. {len(processed_files)} videos processed.")
    print(f"Keypoints saved to: {args.output_dir}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a local directory of sign language videos.")
    parser.add_argument('--input_dir', type=str, default='dataset/raw_local', help='Directory containing the raw local video files.')
    parser.add_argument('--output_dir', type=str, default='dataset/keypoints_local', help='Directory to save the processed keypoints and metadata.')
    
    args = parser.parse_args()
    main(args)
