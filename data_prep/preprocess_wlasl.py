import cv2
import mediapipe as mp
import numpy as np
import os
import json
import argparse
from tqdm import tqdm
import pandas as pd
import sys

# Add project root to Python path to allow importing from pose_extractor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pose_extractor.mediapipe_extractor import MediaPipeExtractor

# --- Constants ---
MAX_FRAMES = 100
TARGET_FPS = 25

# --- Configuration ---
# Assuming the WLASL dataset is cloned into datasets/WLASL
WLASL_ROOT = "datasets/WLASL"
# Pointing to the user's preferred video download location
VIDEO_DIR = os.path.join(WLASL_ROOT, "start_kit", "raw_videos")
METADATA_FILE = os.path.join(WLASL_ROOT, "start_kit", "WLASL_v0.3.json") # Corrected path
OUTPUT_DIR = "dataset/keypoints_wlasl"
METADATA_CSV = os.path.join(OUTPUT_DIR, "metadata.csv")

# --- Helper Functions ---

def process_video_file(video_path, label, extractor):
    """
    Processes a single video file, extracts landmarks, computes features,
    and returns them.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # print(f"Warning: Could not open video file {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    all_landmarks = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = extractor.extract_landmarks(frame)
        landmarks = extractor.get_landmarks_from_results(results)
        all_landmarks.append(landmarks)
        
    cap.release()
    
    if not all_landmarks:
        return None

    landmarks_array = np.array(all_landmarks)
    
    # Normalize landmarks
    normalized_landmarks = []
    for frame_landmarks in landmarks_array:
        norm_lm, _, _ = extractor.normalize_landmarks(frame_landmarks)
        normalized_landmarks.append(norm_lm)
    normalized_landmarks = np.array(normalized_landmarks)

    # Compute velocities (delta)
    velocities = np.diff(normalized_landmarks, axis=0, prepend=normalized_landmarks[0:1])
    
    # Combine features: (x, y, z, dx, dy, dz)
    features = np.concatenate([normalized_landmarks, velocities], axis=-1)

    return {
        'features': features,
        'label': label,
        'frames': num_frames,
        'fps': fps,
        'source': 'WLASL'
    }

def process_video_file(video_path, extractor, max_frames, target_fps):
    """
    Processes a single video file to extract keypoints.
    Returns a numpy array of keypoints or an empty array if processing fails.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # print(f"Error: Could not open video file {video_path}")
            return np.array([])

        all_keypoints = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = extractor.extract_landmarks(frame)
            keypoints = extractor.get_landmarks_from_results(results)
            all_keypoints.append(keypoints)
        
        cap.release()
        
        # If no keypoints were extracted at all, return an empty array
        if not all_keypoints:
            return np.array([])

        # Pad or truncate the sequence of keypoints
        num_frames = len(all_keypoints)
        if num_frames > max_frames:
            all_keypoints = all_keypoints[:max_frames]
        elif num_frames < max_frames:
            # Pad with the last valid frame
            last_frame_keypoints = all_keypoints[-1]
            padding = [last_frame_keypoints] * (max_frames - num_frames)
            all_keypoints.extend(padding)
        
        return np.array(all_keypoints)

    except Exception as e:
        print(f"\nERROR processing video {video_path}: {e}")
        return np.array([])

# --- Main Execution ---

def main():
    """
    Main function to preprocess the WLASL dataset.
    NOW RUNS IN A SINGLE THREAD to avoid multiprocessing errors on macOS.
    """
    print("Starting WLASL dataset preprocessing (single-threaded mode)...")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load WLASL metadata
    with open(METADATA_FILE, 'r') as f:
        wlasl_data = json.load(f)
        
    tasks = []
    for entry in wlasl_data:
        gloss = entry['gloss']
        for instance in entry['instances']:
            video_id = instance['video_id']
            tasks.append((video_id, gloss))
            
    print(f"Found {len(tasks)} videos to process.")

    # Create a single extractor instance to be used for all videos.
    extractor = MediaPipeExtractor()
    
    metadata_records = []
    
    # Process videos one by one in a simple loop
    for video_id, label in tqdm(tasks, desc="Processing Videos"):
        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            # Also check for .swf, as some files might have this extension
            video_path = os.path.join(VIDEO_DIR, f"{video_id}.swf")
            if not os.path.exists(video_path):
                # print(f"Warning: Video file not found, skipping: {video_path}")
                continue
            
        # Process video - Pass all required arguments
        keypoints = process_video_file(video_path, extractor, MAX_FRAMES, TARGET_FPS)
        
        # --- CRITICAL FIX ---
        # Check if the returned keypoints are valid before saving.
        if keypoints is not None and len(keypoints) > 0:
            # Save features to .npz file
            output_path = os.path.join(OUTPUT_DIR, f"{video_id}.npz")
            np.savez_compressed(output_path, data=keypoints, label=label)
            # Append metadata for CSV
            metadata_records.append({
                'video_id': video_id,
                'label': label,
                'frames': len(keypoints),
                'fps': TARGET_FPS,  # Use the target FPS for consistency
                'source': 'WLASL'
            })

    # Clean up the extractor
    extractor.close()

    if not metadata_records:
        print("No videos were processed successfully. Please ensure videos were downloaded correctly.")
        return

    # Save metadata to CSV
    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(METADATA_CSV, index=False)
    
    print(f"Preprocessing complete. {len(metadata_records)} videos processed.")
    print(f"Keypoints saved to: {OUTPUT_DIR}")
    print(f"Metadata saved to: {METADATA_CSV}")

if __name__ == "__main__":
    main()
