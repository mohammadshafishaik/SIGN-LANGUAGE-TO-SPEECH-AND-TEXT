"""
🎯 TEST WLASL TRAINED MODEL
Loads your Kaggle-trained model and tests it with webcam
"""

import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
import time

print("="*80)
print("🎯 WLASL MODEL TESTING")
print("="*80)
print()

# ============================================================================
# STEP 1: LOAD MODEL AND LABELS
# ============================================================================
print("📥 Loading trained model...")

MODEL_PATH = 'checkpoints/wlasl_100_best.keras'
LABELS_PATH = 'checkpoints/wlasl_labels.txt'

try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\n💡 Make sure you downloaded the model from Kaggle to:")
    print(f"   {MODEL_PATH}")
    exit(1)

# Load labels
with open(LABELS_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

print(f"✅ Labels loaded: {len(labels)} classes")
print(f"\n📊 Sample classes: {labels[:10]}")
print()

# ============================================================================
# STEP 2: SETUP MEDIAPIPE
# ============================================================================
print("🎥 Initializing MediaPipe...")

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints(results):
    """Extract keypoints matching training format"""
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark[:10]]) if results.face_landmarks else np.zeros((10, 3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    
    keypoints = np.concatenate([pose, face, lh, rh])
    if keypoints.shape[0] < 104:
        padding = np.zeros((104 - keypoints.shape[0], 3))
        keypoints = np.concatenate([keypoints, padding])
    
    return keypoints

print("✅ MediaPipe ready!")
print()

# ============================================================================
# STEP 3: REAL-TIME TESTING
# ============================================================================
print("🎬 Starting webcam testing...")
print("="*80)
print("CONTROLS:")
print("  - SPACE: Start recording sign (hold for 2 seconds)")
print("  - Q: Quit")
print("="*80)
print()

cap = cv2.VideoCapture(0)
sequence = []
predictions_history = []
recording = False
start_time = 0
MAX_FRAMES = 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = holistic.process(frame_rgb)
    
    # Draw landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
    # Recording logic
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '):  # Start recording
        if not recording:
            recording = True
            sequence = []
            start_time = time.time()
            print("🔴 RECORDING... (hold for 2 seconds)")
    
    if recording:
        # Collect frames
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        
        # Show recording indicator
        cv2.putText(frame, "RECORDING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.circle(frame, (580, 30), 10, (0, 0, 255), -1)
        
        # After 2 seconds or MAX_FRAMES, make prediction
        if time.time() - start_time >= 2 or len(sequence) >= MAX_FRAMES:
            recording = False
            
            # Prepare input
            if len(sequence) < MAX_FRAMES:
                # Pad sequence
                while len(sequence) < MAX_FRAMES:
                    sequence.append(np.zeros((104, 3)))
            
            sequence = sequence[:MAX_FRAMES]
            input_data = np.array([sequence], dtype=np.float32)
            input_data = input_data.reshape(1, MAX_FRAMES, -1)
            
            # Normalize (same as training)
            input_data = (input_data - input_data.mean()) / (input_data.std() + 1e-8)
            
            # Predict
            predictions = model.predict(input_data, verbose=0)[0]
            top_3_idx = np.argsort(predictions)[-3:][::-1]
            
            print("\n" + "="*60)
            print("🎯 PREDICTION RESULTS:")
            print("="*60)
            for i, idx in enumerate(top_3_idx, 1):
                confidence = predictions[idx] * 100
                print(f"  {i}. {labels[idx]:20s}: {confidence:5.2f}%")
            print("="*60)
            print("\nPress SPACE to record another sign, Q to quit\n")
            
            predictions_history.append(labels[top_3_idx[0]])
    
    # Display current prediction if available
    if predictions_history:
        latest = predictions_history[-1]
        cv2.putText(frame, f"Last: {latest}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Instructions
    cv2.putText(frame, "Press SPACE to sign, Q to quit", (10, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow('WLASL Sign Recognition', frame)
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()

print("\n✅ Testing complete!")
if predictions_history:
    print(f"\n📊 Total predictions made: {len(predictions_history)}")
    print(f"   Signs detected: {', '.join(predictions_history[:10])}")
