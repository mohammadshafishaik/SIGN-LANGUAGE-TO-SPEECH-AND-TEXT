"""
Test if webcam extraction produces features that match training data distribution
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import tensorflow as tf
import json

print("\n" + "="*70)
print("TESTING FEATURE EXTRACTION: Webcam vs Training Data")
print("="*70)

# Load a training image and extract features
img_path = list(Path("datasets/ISL/Indian/1").glob("*.jpg"))[0]
image = cv2.imread(str(img_path))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

def extract_features(image_rgb, static_mode):
    """Extract features same way as training"""
    with mp_hands.Hands(
        static_image_mode=static_mode,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands, \
    mp_pose.Pose(
        static_image_mode=static_mode,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        hands_results = hands.process(image_rgb)
        pose_results = pose.process(image_rgb)
        
        features = []
        
        # Hand landmarks
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks[:2]:
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])
            if len(hands_results.multi_hand_landmarks) == 1:
                features.extend([0.0] * 63)
        else:
            features.extend([0.0] * 126)
        
        # Pose landmarks
        if pose_results.pose_landmarks:
            upper_body_indices = [11, 12, 13, 14, 15, 16]
            for idx in upper_body_indices:
                lm = pose_results.pose_landmarks.landmark[idx]
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * 18)
        
        features = np.clip(features, -10, 10)
        return np.array(features)

print("\n1. Extracting features from training image...")
features_extracted = extract_features(image_rgb, static_mode=True)
print(f"   Shape: {features_extracted.shape}")
print(f"   Mean: {features_extracted.mean():.4f}")
print(f"   Std: {features_extracted.std():.4f}")
print(f"   First 10 values: {features_extracted[:10]}")

# Load the ACTUAL training features for this class
print("\n2. Loading preprocessed training features...")
train_data = np.load("dataset/keypoints_isl/1.npz")['features']
print(f"   Shape: {train_data.shape}")
print(f"   Mean: {train_data.mean():.4f}")
print(f"   Std: {train_data.std():.4f}")
print(f"   First sample first 10 values: {train_data[0][:10]}")

# Test with model
print("\n3. Testing with model...")
model = tf.keras.models.load_model("checkpoints/isl_best.keras")

with open("dataset/splits_isl/label_mappings.json") as f:
    label_info = json.load(f)
idx_to_label = {int(k): v for k, v in label_info['idx_to_label'].items()}

# Predict on extracted features
pred_extracted = model.predict(features_extracted.reshape(1, -1), verbose=0)[0]
top_5_idx = np.argsort(pred_extracted)[-5:][::-1]

print("\n   Prediction on NEWLY EXTRACTED features:")
for i, idx in enumerate(top_5_idx):
    label = idx_to_label[idx]
    conf = pred_extracted[idx]
    marker = "✓" if label == '1' else "✗"
    num_or_let = "[NUM]" if label in ['1','2','3','4','5','6','7','8','9'] else "[LTR]"
    print(f"   {i+1}. {marker} {num_or_let} {label}: {conf*100:.1f}%")

# Predict on TRAINING features
sample_train = train_data[0]
pred_train = model.predict(sample_train.reshape(1, -1), verbose=0)[0]
top_5_train = np.argsort(pred_train)[-5:][::-1]

print("\n   Prediction on ORIGINAL TRAINING features:")
for i, idx in enumerate(top_5_train):
    label = idx_to_label[idx]
    conf = pred_train[idx]
    marker = "✓" if label == '1' else "✗"
    num_or_let = "[NUM]" if label in ['1','2','3','4','5','6','7','8','9'] else "[LTR]"
    print(f"   {i+1}. {marker} {num_or_let} {label}: {conf*100:.1f}%")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)

if idx_to_label[top_5_idx[0]] == '1':
    print("✓ NEWLY EXTRACTED features predict '1' correctly!")
    print("  → Skeleton extraction is working fine")
    print("  → Problem must be in WEBCAM capture or hand pose")
else:
    print(f"❌ NEWLY EXTRACTED features predict '{idx_to_label[top_5_idx[0]]}' not '1'!")
    print("  → There's an issue with feature extraction")
    print("  → Even reprocessing training images fails")
    
print("="*70 + "\n")
