"""
Real-Time ISL Recognition - FIXED VERSION
Uses static_image_mode=True to match training exactly
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from pathlib import Path
import json

# Load model and label mappings
print("Loading ISL model...")
model_path = Path(__file__).parent.parent / "checkpoints" / "isl_best.keras"
model = tf.keras.models.load_model(model_path)
print(f"Model loaded: {model_path}")

label_path = Path(__file__).parent.parent / "dataset" / "splits_isl" / "label_mappings.json"
with open(label_path, 'r') as f:
    label_info = json.load(f)
idx_to_label = {int(k): v for k, v in label_info['idx_to_label'].items()}
print(f"Labels loaded: {len(idx_to_label)} classes\n")

# Separate numbers and letters
numbers = [str(i) for i in range(1, 10)]

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(image, hands, pose):
    """Extract 144D features from image - SAME AS TRAINING"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process hands
    hands_results = hands.process(image_rgb)
    
    # Process pose
    pose_results = pose.process(image_rgb)
    
    # Extract features (identical to training code)
    features = []
    
    # Hand landmarks (126 features)
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks[:2]:
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
        if len(hands_results.multi_hand_landmarks) == 1:
            features.extend([0.0] * 63)
    else:
        features.extend([0.0] * 126)
    
    # Pose landmarks (18 features - upper body only)
    if pose_results.pose_landmarks:
        upper_body_indices = [11, 12, 13, 14, 15, 16]
        for idx in upper_body_indices:
            lm = pose_results.pose_landmarks.landmark[idx]
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 18)
    
    # Clip values
    features = np.clip(features, -10, 10)
    
    return np.array(features), hands_results, pose_results

def main():
    print("\n" + "="*70)
    print("ISL RECOGNITION - FIXED VERSION (Top 5 Predictions)")
    print("="*70)
    print("FIX: Now using static_image_mode=TRUE (matches training!)")
    print("Opening webcam...")
    print("Recognizing: A-Z (26 letters), 1-9 (9 numbers)")
    print("Press 'q' to quit")
    print("="*70)
    print("\nTIPS FOR NUMBERS:")
    print("  1: Index finger STRAIGHT UP (vertical, not sideways!)")
    print("  2: Peace sign (V shape)")
    print("  5: Open palm (easiest - try this first!)")
    print("  9: OK sign (thumb+index circle)")
    print("="*70 + "\n")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize MediaPipe - FIXED: static_image_mode=TRUE
    with mp_hands.Hands(
        static_image_mode=True,  # ← FIXED! Was False, now True
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands, \
    mp_pose.Pose(
        static_image_mode=True,  # ← FIXED! Was False, now True
        min_detection_confidence=0.5
    ) as pose:
        
        frame_count = 0
        predictions_list = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Mirror frame
            frame = cv2.flip(frame, 1)
            
            # Extract features every 5 frames (slower but more accurate with static mode)
            frame_count += 1
            if frame_count % 5 == 0:
                try:
                    features, hands_results, pose_results = extract_keypoints(frame, hands, pose)
                    
                    if hands_results.multi_hand_landmarks:
                        # Make prediction
                        features_batch = features.reshape(1, -1)
                        predictions = model.predict(features_batch, verbose=0)[0]
                        
                        # Get top 5 predictions
                        top_5_idx = np.argsort(predictions)[-5:][::-1]
                        predictions_list = [
                            (idx_to_label[idx], predictions[idx]) 
                            for idx in top_5_idx
                        ]
                    else:
                        predictions_list = []
                        
                except Exception as e:
                    predictions_list = []
                    print(f"Warning - Error: {str(e)}")
            
            # Draw hand landmarks
            if 'hands_results' in locals() and hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
            
            # Display predictions
            h, w = frame.shape[:2]
            
            if predictions_list:
                # Main prediction box
                box_height = 280
                cv2.rectangle(frame, (10, 10), (650, box_height), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 10), (650, box_height), (0, 255, 0), 3)
                
                # Title
                cv2.putText(frame, "Top 5 Predictions (FIXED):", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (255, 255, 0), 2)
                
                # Display top 5
                y_offset = 80
                for i, (label, conf) in enumerate(predictions_list):
                    # Determine if number or letter
                    is_number = label in numbers
                    type_text = "[NUM]" if is_number else "[LTR]"
                    
                    # Color based on confidence and rank
                    if i == 0:  # Top prediction
                        if conf > 0.8:
                            color = (0, 255, 0)  # Green
                        elif conf > 0.5:
                            color = (0, 255, 255)  # Yellow
                        else:
                            color = (0, 165, 255)  # Orange
                        thickness = 2
                        font_scale = 0.9
                    else:
                        color = (180, 180, 180)  # Gray
                        thickness = 1
                        font_scale = 0.7
                    
                    # Draw text first (left side)
                    text = f"{i+1}. {type_text} {label}"
                    cv2.putText(frame, text, 
                               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                               font_scale, color, thickness)
                    
                    # Draw confidence bar (right side, after text)
                    bar_start_x = 250
                    bar_width = int(conf * 350)
                    cv2.rectangle(frame, (bar_start_x, y_offset-18), 
                                 (bar_start_x + bar_width, y_offset-5), color, -1)
                    
                    # Draw percentage text on the bar
                    conf_text = f"{conf:.0%}"
                    cv2.putText(frame, conf_text, 
                               (bar_start_x + bar_width + 10, y_offset-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, color, 1)
                    
                    y_offset += 45
            else:
                # No hand detected
                cv2.rectangle(frame, (10, 10), (500, 120), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 10), (500, 120), (0, 0, 255), 2)
                cv2.putText(frame, "No hand detected", 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, (0, 0, 255), 2)
                cv2.putText(frame, "Show hand to camera", 
                           (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 1)
            
            # Instructions box
            cv2.rectangle(frame, (w-400, h-100), (w-10, h-10), (0, 0, 0), -1)
            cv2.rectangle(frame, (w-400, h-100), (w-10, h-10), (255, 255, 255), 2)
            
            cv2.putText(frame, "Press 'q' to quit", 
                       (w-380, h-70), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            
            # Legend
            cv2.putText(frame, "[NUM] = Number (1-9)", 
                       (w-380, h-45), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 255), 1)
            cv2.putText(frame, "[LTR] = Letter (A-Z)", 
                       (w-380, h-25), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 1)
            
            # Show frame
            cv2.imshow('ISL Recognition - FIXED (static_image_mode=TRUE)', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam closed. Thanks for testing!")

if __name__ == "__main__":
    main()
