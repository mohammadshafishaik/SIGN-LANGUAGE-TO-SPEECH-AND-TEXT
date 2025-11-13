"""
Enhanced WLASL video preprocessing with MediaPipe pose + hand extraction.
Extracts 3D keypoints + velocity + acceleration features.

Usage:
    python data_prep/preprocess_wlasl_enhanced.py --input_dir dataset/raw/ --output_dir dataset/keypoints_wlasl/
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import argparse
from pathlib import Path
from tqdm import tqdm
import json

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    """Extract pose + both hands keypoints (75 landmarks × 3 coords = 225 features)."""
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def compute_derivatives(sequence):
    """Compute velocity and acceleration from position sequence."""
    if len(sequence) < 2:
        return sequence, np.zeros_like(sequence), np.zeros_like(sequence)
    
    # Velocity (first derivative)
    velocity = np.diff(sequence, axis=0)
    velocity = np.vstack([velocity[0], velocity])  # Pad first frame
    
    # Acceleration (second derivative)
    acceleration = np.diff(velocity, axis=0)
    acceleration = np.vstack([acceleration[0], acceleration])  # Pad first frame
    
    return sequence, velocity, acceleration

def process_video(video_path, holistic, max_frames=150):
    """
    Process a single video and extract keypoint sequence.
    
    Returns:
        keypoints: (T, 225) - position features
        velocity: (T, 225) - velocity features
        acceleration: (T, 225) - acceleration features
        success: bool - whether extraction succeeded
    """
    cap = cv2.VideoCapture(str(video_path))
    keypoints_list = []
    
    frame_count = 0
    success_count = 0
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = holistic.process(frame_rgb)
        
        # Extract keypoints
        kp = extract_keypoints(results)
        
        # Only keep frames with successful detection
        if results.pose_landmarks is not None:
            keypoints_list.append(kp)
            success_count += 1
        
        frame_count += 1
    
    cap.release()
    
    # Check if we got enough valid frames
    if len(keypoints_list) < 10:  # Minimum 10 frames
        return None, None, None, False
    
    # Convert to numpy array
    keypoints = np.array(keypoints_list)
    
    # Compute derivatives
    position, velocity, acceleration = compute_derivatives(keypoints)
    
    # Check for NaN values
    if np.isnan(position).any() or np.isnan(velocity).any() or np.isnan(acceleration).any():
        return None, None, None, False
    
    return position, velocity, acceleration, True

def main(args):
    print("="*70)
    print("WLASL ENHANCED PREPROCESSING - 3D KEYPOINTS + VELOCITY + ACCELERATION")
    print("="*70)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all video files
    video_files = list(input_dir.glob('*.mp4')) + list(input_dir.glob('*.avi')) + list(input_dir.glob('*.mov'))
    print(f"\n📂 Found {len(video_files)} videos in {input_dir}")
    
    if len(video_files) == 0:
        print("❌ No videos found! Please download videos first using download_wlasl_videos.py")
        return
    
    # Load metadata if available
    metadata_path = input_dir / 'wlasl_downloaded_metadata.csv'
    if metadata_path.exists():
        metadata_df = pd.read_csv(metadata_path)
        print(f"📊 Loaded metadata: {len(metadata_df)} entries")
        # Create a lookup dict
        metadata_lookup = {row['filename']: row['gloss'] for _, row in metadata_df.iterrows()}
    else:
        print("⚠️  No metadata file found. Will extract labels from filenames.")
        metadata_lookup = {}
    
    # Initialize MediaPipe
    print("\n🔧 Initializing MediaPipe Holistic...")
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )
    
    # Process videos
    processed_data = []
    failed_videos = []
    
    print(f"\n🎬 Processing {len(video_files)} videos...")
    for video_file in tqdm(video_files, desc="Extracting keypoints"):
        # Get label from metadata or filename
        filename = video_file.name
        if filename in metadata_lookup:
            label = metadata_lookup[filename]
        else:
            # Extract from filename (format: label_videoid.mp4)
            label = filename.split('_')[0]
        
        # Process video
        position, velocity, acceleration, success = process_video(video_file, holistic, max_frames=args.max_frames)
        
        if success:
            # Save features
            video_id = video_file.stem
            npy_path = output_dir / f"{video_id}.npy"
            
            # Stack all features: (T, 225*3 = 675)
            features = np.concatenate([position, velocity, acceleration], axis=1)
            np.save(npy_path, features)
            
            processed_data.append({
                'video_id': video_id,
                'filename': filename,
                'label': label,
                'keypoints_path': str(npy_path),
                'num_frames': len(position),
                'feature_dim': features.shape[1]
            })
        else:
            failed_videos.append(filename)
    
    holistic.close()
    
    # Save metadata
    df = pd.DataFrame(processed_data)
    metadata_output = output_dir / 'metadata.csv'
    df.to_csv(metadata_output, index=False)
    
    # Print statistics
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"✅ Successfully processed: {len(processed_data)} videos")
    print(f"❌ Failed: {len(failed_videos)} videos")
    print(f"📊 Unique labels: {df['label'].nunique()}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts().to_string())
    print(f"\n📁 Features saved to: {output_dir}")
    print(f"📄 Metadata saved to: {metadata_output}")
    print(f"🎯 Feature dimension: {df['feature_dim'].iloc[0] if len(df) > 0 else 'N/A'} (position + velocity + acceleration)")
    print("="*70)
    
    if failed_videos:
        print(f"\n⚠️  Failed videos ({len(failed_videos)}):")
        for vid in failed_videos[:10]:
            print(f"  - {vid}")
        if len(failed_videos) > 10:
            print(f"  ... and {len(failed_videos) - 10} more")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced WLASL preprocessing")
    parser.add_argument('--input_dir', type=str, default='dataset/raw/',
                        help='Directory containing raw videos')
    parser.add_argument('--output_dir', type=str, default='dataset/keypoints_wlasl/',
                        help='Output directory for keypoint features')
    parser.add_argument('--max_frames', type=int, default=150,
                        help='Maximum frames to process per video')
    
    args = parser.parse_args()
    main(args)
