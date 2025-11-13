"""
CRITICAL BUG FOUND: MediaPipe Settings Mismatch

TRAINING:     static_image_mode = TRUE
INFERENCE:    static_image_mode = FALSE

This causes different landmark coordinates!
The model was trained on one type of data but gets another type at inference.
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

print("\n" + "="*70)
print("🔴 CRITICAL BUG DIAGNOSIS")
print("="*70)
print("\nTesting MediaPipe with BOTH settings on same image...")
print("="*70 + "\n")

# Load a sample image
img_path = Path("datasets/ISL/Indian/1")
sample_img = list(img_path.glob("*.jpg"))[0]
image = cv2.imread(str(sample_img))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mp_hands = mp.solutions.hands

# Test 1: Training settings (static=True)
print("TEST 1: Training Settings (static_image_mode=True)")
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
) as hands:
    result_static = hands.process(image_rgb)
    if result_static.multi_hand_landmarks:
        landmarks_static = []
        for lm in result_static.multi_hand_landmarks[0].landmark:
            landmarks_static.extend([lm.x, lm.y, lm.z])
        print(f"  ✓ Hand detected: {len(landmarks_static)} values")
        print(f"  Sample landmarks: {landmarks_static[:9]}")
    else:
        landmarks_static = None
        print(f"  ❌ No hand detected")

# Test 2: Webcam settings (static=False)
print("\nTEST 2: Webcam Settings (static_image_mode=False)")
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    result_tracking = hands.process(image_rgb)
    if result_tracking.multi_hand_landmarks:
        landmarks_tracking = []
        for lm in result_tracking.multi_hand_landmarks[0].landmark:
            landmarks_tracking.extend([lm.x, lm.y, lm.z])
        print(f"  ✓ Hand detected: {len(landmarks_tracking)} values")
        print(f"  Sample landmarks: {landmarks_tracking[:9]}")
    else:
        landmarks_tracking = None
        print(f"  ❌ No hand detected")

# Compare
print("\n" + "="*70)
print("COMPARISON:")
print("="*70)

if landmarks_static and landmarks_tracking:
    diff = np.linalg.norm(np.array(landmarks_static) - np.array(landmarks_tracking))
    print(f"Landmark difference: {diff:.4f}")
    if diff > 0.01:
        print(f"\n🔴 PROBLEM CONFIRMED:")
        print(f"   Different settings = Different landmarks!")
        print(f"   Model trained on static=True but webcam uses static=False")
        print(f"   This causes WRONG predictions!")
    else:
        print(f"\n✓ Landmarks are similar (diff < 0.01)")
else:
    print("Could not compare - detection failed in one mode")

print("\n" + "="*70)
print("SOLUTION:")
print("="*70)
print("""
Fix the webcam inference code to use:
    static_image_mode = True  (NOT False!)
    
This will make MediaPipe extract landmarks the SAME WAY
as during training, fixing the prediction issues!
""")
print("="*70 + "\n")
