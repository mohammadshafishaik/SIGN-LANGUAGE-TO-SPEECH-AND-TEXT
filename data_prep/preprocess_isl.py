"""
ISL (Indian Sign Language) Dataset Preprocessor
Extracts 3D keypoints using MediaPipe from ISL videos

Usage:
    python data_prep/preprocess_isl.py
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic

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
    
    # Velocity
    velocity = np.diff(sequence, axis=0)
    velocity = np.vstack([velocity[0], velocity])
    
    # Acceleration
    acceleration = np.diff(velocity, axis=0)
    acceleration = np.vstack([acceleration[0], acceleration])
    
    return sequence, velocity, acceleration

def process_video(video_path, holistic, max_frames=150):
    """Process a single video and extract keypoint sequence."""
    cap = cv2.VideoCapture(str(video_path))
    keypoints_list = []
    
    success_count = 0
    frame_count = 0
    
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
    if len(keypoints_list) < 10:
        return None, None, None, False
    
    # Convert to numpy array
    keypoints = np.array(keypoints_list)
    
    # Compute derivatives
    position, velocity, acceleration = compute_derivatives(keypoints)
    
    # Clip extreme values
    position = np.clip(position, -10, 10)
    velocity = np.clip(velocity, -10, 10)
    acceleration = np.clip(acceleration, -10, 10)
    
    # Check for NaN
    if np.isnan(position).any() or np.isnan(velocity).any() or np.isnan(acceleration).any():
        return None, None, None, False
    
    return position, velocity, acceleration, True

def main():
    print("="*70)
    print("ISL (INDIAN SIGN LANGUAGE) PREPROCESSING")
    print("="*70)
    
    input_dir = Path('datasets/ISL')
    output_dir = Path('dataset/keypoints_isl')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(input_dir.rglob(ext)))
    
    print(f"\n📂 Found {len(video_files)} videos")
    
    if len(video_files) == 0:
        print("❌ No videos found! Please extract the downloaded ZIP file first.")
        return
    
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
        # Extract label from path (parent directory name or filename)
        label = video_file.parent.name if video_file.parent != input_dir else video_file.stem.split('_')[0]
        
        # Process video
        position, velocity, acceleration, success = process_video(video_file, holistic, max_frames=150)
        
        if success:
            # Save features
            video_id = video_file.stem
            npy_path = output_dir / f"{video_id}.npy"
            
            # Stack all features: (T, 225*3 = 675)
            features = np.concatenate([position, velocity, acceleration], axis=1)
            np.save(npy_path, features)
            
            processed_data.append({
                'video_id': video_id,
                'filename': video_file.name,
                'label': label,
                'keypoints_path': str(npy_path),
                'num_frames': len(position),
                'feature_dim': features.shape[1]
            })
        else:
            failed_videos.append(video_file.name)
    
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
    print(f"\nLabel distribution (top 20):")
    print(df['label'].value_counts().head(20).to_string())
    print(f"\n📁 Features saved to: {output_dir}")
    print(f"📄 Metadata saved to: {metadata_output}")
    print(f"🎯 Feature dimension: {df['feature_dim'].iloc[0] if len(df) > 0 else 'N/A'}")
    print("="*70)
    
    if failed_videos and len(failed_videos) < 20:
        print(f"\n⚠️  Failed videos:")
        for vid in failed_videos:
            print(f"  - {vid}")

if __name__ == "__main__":
    main()
