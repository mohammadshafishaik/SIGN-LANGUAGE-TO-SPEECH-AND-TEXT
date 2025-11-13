"""
ISL Image Dataset Preprocessing
Extracts 3D keypoints from static ISL hand gesture images using MediaPipe
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
from tqdm import tqdm

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

def extract_keypoints_from_image(image_path, hands_detector, pose_detector):
    """Extract 3D keypoints from a single image"""
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with hands
    hands_results = hands_detector.process(image_rgb)
    
    # Process with pose (upper body for context)
    pose_results = pose_detector.process(image_rgb)
    
    # Initialize features array
    features = []
    
    # Extract hand landmarks (21 landmarks × 3 coords × 2 hands = 126 features)
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks[:2]:  # Max 2 hands
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
        
        # Pad if only one hand detected
        if len(hands_results.multi_hand_landmarks) == 1:
            features.extend([0.0] * 63)  # Pad with zeros
    else:
        # No hands detected - pad with zeros
        features.extend([0.0] * 126)
    
    # Extract pose landmarks (upper body only: shoulders, elbows, wrists = 6 points × 3 = 18 features)
    if pose_results.pose_landmarks:
        upper_body_indices = [11, 12, 13, 14, 15, 16]  # Shoulders, elbows, wrists
        for idx in upper_body_indices:
            lm = pose_results.pose_landmarks.landmark[idx]
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 18)
    
    # Total features: 126 (hands) + 18 (upper body) = 144 features
    
    # Clip extreme values
    features = np.clip(features, -10, 10)
    
    return np.array(features, dtype=np.float32)


def process_isl_dataset(dataset_dir, output_dir):
    """
    Process ISL image dataset
    
    Args:
        dataset_dir: Path to datasets/ISL/Indian/
        output_dir: Path to save processed keypoints
    """
    
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize MediaPipe detectors
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    pose = mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Get all class directories
    class_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    print(f"\n{'='*60}")
    print(f"ISL IMAGE DATASET PREPROCESSING")
    print(f"{'='*60}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of classes: {len(class_dirs)}")
    print(f"{'='*60}\n")
    
    stats = {
        'total_images': 0,
        'processed_images': 0,
        'failed_images': 0,
        'classes': []
    }
    
    # Process each class
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # SKIP if NPZ already exists
        class_output_path = output_path / f"{class_name}.npz"
        if class_output_path.exists():
            print(f"\n⏭️  Skipping class: {class_name} (already processed)")
            continue
        
        print(f"\n📁 Processing class: {class_name}")
        
        # Get all images in this class
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        
        class_features = []
        class_labels = []
        processed = 0
        failed = 0
        
        # Process each image
        for img_path in tqdm(image_files, desc=f"  Extracting keypoints"):
            stats['total_images'] += 1
            
            features = extract_keypoints_from_image(img_path, hands, pose)
            
            if features is not None and not np.isnan(features).any():
                class_features.append(features)
                class_labels.append(class_name)
                processed += 1
                stats['processed_images'] += 1
            else:
                failed += 1
                stats['failed_images'] += 1
        
        # Save class data
        if class_features:
            class_output_path = output_path / f"{class_name}.npz"
            np.savez_compressed(
                class_output_path,
                features=np.array(class_features),
                labels=np.array(class_labels)
            )
            
            print(f"  ✅ Saved: {len(class_features)} samples")
            print(f"  ❌ Failed: {failed} samples")
            
            stats['classes'].append({
                'name': class_name,
                'processed': processed,
                'failed': failed,
                'total': len(image_files)
            })
        else:
            print(f"  ⚠️  No valid samples extracted!")
    
    # Clean up
    hands.close()
    pose.close()
    
    # Save processing stats
    stats_path = output_path / 'processing_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total images: {stats['total_images']}")
    print(f"Successfully processed: {stats['processed_images']} ({stats['processed_images']/stats['total_images']*100:.1f}%)")
    print(f"Failed: {stats['failed_images']} ({stats['failed_images']/stats['total_images']*100:.1f}%)")
    print(f"Classes: {len(stats['classes'])}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process ISL image dataset')
    parser.add_argument(
        '--input_dir',
        type=str,
        default='datasets/ISL/Indian',
        help='Path to ISL dataset directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='dataset/keypoints_isl',
        help='Path to save processed keypoints'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / args.input_dir
    output_dir = base_dir / args.output_dir
    
    # Process dataset
    stats = process_isl_dataset(str(input_dir), str(output_dir))
    
    print("\n✅ Ready for training!")
    print(f"Next step: python data_prep/create_splits.py --input_dir {output_dir}")
