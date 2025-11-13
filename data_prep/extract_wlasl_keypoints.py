"""
Preprocess WLASL videos to extract keypoint sequences
Processes videos from dataset/raw/ and saves to dataset/keypoints_wlasl/
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
from project_paths import DATASET_DIR
from tqdm import tqdm

# Paths (portable)
VIDEO_DIR = DATASET_DIR / 'raw'
OUTPUT_DIR = DATASET_DIR / 'keypoints_wlasl'

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_features(frame):
    """Extract 144D features from a single frame"""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    
    # Process with MediaPipe
    hand_results = hands.process(image_rgb)
    pose_results = pose.process(image_rgb)
    
    # Extract features (144D)
    features = []
    
    # Hand landmarks (21 points × 3 coords × 2 hands = 126D)
    for hand_idx in range(2):
        if hand_results.multi_hand_landmarks and hand_idx < len(hand_results.multi_hand_landmarks):
            landmarks = hand_results.multi_hand_landmarks[hand_idx]
            for lm in landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * 63)
    
    # Upper body pose (6 points × 3 coords = 18D)
    if pose_results.pose_landmarks:
        pose_indices = [11, 12, 13, 14, 15, 16]  # Shoulders, elbows, wrists
        for idx in pose_indices:
            lm = pose_results.pose_landmarks.landmark[idx]
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 18)
    
    return np.array(features, dtype=np.float32)

def process_video(video_path, max_frames=50):
    """Process a video and extract keypoint sequence"""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Could not open: {video_path}")
        return None
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly if video is too long
    if total_frames > max_frames:
        frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
    else:
        frame_indices = range(total_frames)
    
    keypoints = []
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Extract features
        features = extract_features(frame)
        keypoints.append(features)
    
    cap.release()
    
    if len(keypoints) == 0:
        return None
    
    return np.array(keypoints, dtype=np.float32)

def main():
    """Process all videos in dataset/raw/"""
    print("="*70)
    print("🎬 WLASL VIDEO PREPROCESSING")
    print("="*70)
    
    # Get all video files
    video_files = sorted(VIDEO_DIR.glob('*.mp4'))
    
    if len(video_files) == 0:
        print(f"❌ No videos found in {VIDEO_DIR}")
        return
    
    print(f"📁 Found {len(video_files)} videos")
    print(f"📂 Output: {OUTPUT_DIR}")
    
    # Extract unique words
    words = set()
    for video_file in video_files:
        word = video_file.stem.split('_')[0]
        words.add(word)
    
    print(f"🔤 Words to process: {len(words)}")
    print(f"   {sorted(list(words))[:10]}{'...' if len(words) > 10 else ''}")
    
    # Process each video
    processed = 0
    failed = 0
    
    print("\n🚀 Processing videos...")
    for video_file in tqdm(video_files, desc="Extracting keypoints"):
        try:
            # Extract keypoints
            keypoints = process_video(video_file)
            
            if keypoints is None:
                failed += 1
                continue
            
            # Save keypoints
            output_file = OUTPUT_DIR / f"{video_file.stem}.npy"
            np.save(output_file, keypoints)
            
            processed += 1
            
        except Exception as e:
            print(f"\n❌ Error processing {video_file.name}: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"✓ Processed: {processed} videos")
    print(f"✗ Failed: {failed} videos")
    print(f"📁 Keypoints saved to: {OUTPUT_DIR}")
    print("="*70)
    print("\n💡 Next steps:")
    print("   1. Run: python models/train_wlasl.py")
    print("   2. This will train an LSTM model on the keypoint sequences")
    print("   3. Model will be saved to checkpoints/wlasl_best.keras")
    print("="*70)

if __name__ == '__main__':
    main()
