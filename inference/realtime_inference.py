import cv2
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from collections import deque
import sys
import os
import pyttsx3

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pose_extractor.mediapipe_extractor import MediaPipeExtractor

# --- Constants ---
MAX_FRAMES = 100
PREDICTION_INTERVAL = 30  # Make a prediction every N frames
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to display prediction

def standardize_data(keypoints):
    """
    Applies the same standardization as used in training.
    Now with improved robustness for real-time data.
    """
    # Replace any NaNs with zeros first
    keypoints = np.nan_to_num(keypoints, nan=0.0, posinf=0.0, neginf=0.0)
    
    mean = np.mean(keypoints, axis=(0, 1), keepdims=True)
    std = np.std(keypoints, axis=(0, 1), keepdims=True)
    
    # Avoid division by zero - use a larger epsilon
    std = np.where(std < 1e-3, 1.0, std)
    
    standardized = (keypoints - mean) / std
    
    # Final cleanup - replace any remaining NaNs or infs
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)
    
    return standardized

def realtime_inference(model_path):
    """
    Main function to run real-time sign language recognition from webcam.
    """
    # --- Initialization ---
    print("Initializing components...")
    
    # 1. Load the trained model
    if not os.path.exists(model_path):
        print(f"Error loading model: No file or directory found at {model_path}. Please ensure the model exists.")
        return
    
    def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0):
        pass  # Dummy function for loading
    
    with tf.keras.utils.custom_object_scope({'transformer_encoder_block': transformer_encoder_block}):
        model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
    
    # 2. Load label mapping
    metadata_path = 'dataset/keypoints_combined/metadata_combined.csv'
    if not os.path.exists(metadata_path):
        print(f"Error: Combined metadata not found at {metadata_path}")
        return
    
    df = pd.read_csv(metadata_path)
    label_map = pd.Series(df.label.values, index=df.label_encoded).to_dict()
    print(f"Loaded {len(label_map)} unique labels")
    
    # 3. Initialize MediaPipe Extractor
    extractor = MediaPipeExtractor()
    
    # 4. Initialize Text-to-Speech
    tts_engine = pyttsx3.init()
    
    # 5. Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return
    
    # --- Variables for Inference Loop ---
    sequence_buffer = deque(maxlen=MAX_FRAMES)
    frame_count = 0
    current_prediction = "None"
    current_confidence = 0.0
    last_spoken_prediction = ""
    
    print("\n--- Starting Real-Time Inference ---")
    print("Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam")
            break
        
        # --- 1. Extract Keypoints ---
        mp_results = extractor.extract_landmarks(frame)
        landmarks = extractor.get_landmarks_from_results(mp_results)
        normalized_landmarks, _, _ = extractor.normalize_landmarks(landmarks)
        
        # Check if we detected any landmarks
        has_pose = mp_results.pose_landmarks is not None
        has_left_hand = mp_results.left_hand_landmarks is not None
        has_right_hand = mp_results.right_hand_landmarks is not None
        
        # Add to sequence buffer
        sequence_buffer.append(normalized_landmarks)
        frame_count += 1
        
        # --- 2. Make Prediction at intervals ---
        if frame_count % PREDICTION_INTERVAL == 0 and len(sequence_buffer) >= MAX_FRAMES:
            print(f"[DEBUG] Making prediction at frame {frame_count}...")
            
            # Prepare data for prediction
            sequence_array = np.array(list(sequence_buffer))
            
            # Additional data cleaning before standardization
            sequence_array = np.nan_to_num(sequence_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            print(f"[DEBUG] Sequence contains NaN: {np.isnan(sequence_array).any()}")
            print(f"[DEBUG] Sequence contains Inf: {np.isinf(sequence_array).any()}")
            
            standardized_sequence = standardize_data(sequence_array)
            
            print(f"[DEBUG] Standardized contains NaN: {np.isnan(standardized_sequence).any()}")
            print(f"[DEBUG] Standardized contains Inf: {np.isinf(standardized_sequence).any()}")
            
            input_data = np.expand_dims(standardized_sequence, axis=0)
            
            print(f"[DEBUG] Input shape: {input_data.shape}")
            print(f"[DEBUG] Input data range: [{np.min(input_data):.4f}, {np.max(input_data):.4f}]")
            
            # Predict
            prediction = model.predict(input_data, verbose=0)
            predicted_class_index = np.argmax(prediction)
            confidence = np.max(prediction)
            
            print(f"[DEBUG] Raw prediction: {prediction}")
            print(f"[DEBUG] Predicted class: {predicted_class_index}, Confidence: {confidence:.4f}")
            
            # Always update the prediction, even if confidence is low
            predicted_label = label_map.get(predicted_class_index, "Unknown")
            current_prediction = predicted_label
            current_confidence = confidence
            
            # Speak if confidence is above threshold and prediction changed
            if not np.isnan(confidence) and confidence > CONFIDENCE_THRESHOLD:
                if predicted_label != last_spoken_prediction:
                    print(f"[SPEAKING] {predicted_label} (Confidence: {confidence:.2%})")
                    tts_engine.say(predicted_label)
                    tts_engine.runAndWait()
                    last_spoken_prediction = predicted_label
            else:
                if np.isnan(confidence):
                    print(f"[DEBUG] Confidence is NaN - model output is corrupted")
                else:
                    print(f"[DEBUG] Confidence too low: {confidence:.4f} < {CONFIDENCE_THRESHOLD}")
        
        # --- 3. Display information on frame ---
        # Status text
        status_text = "Collecting" if (has_pose or has_left_hand or has_right_hand) else "Idle"
        cv2.putText(frame, f"Status: {status_text}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Sequence length
        cv2.putText(frame, f"Frames: {len(sequence_buffer)}/{MAX_FRAMES}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Detection status
        cv2.putText(frame, f"Pose: {'Found' if has_pose else 'Not Found'}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"L.Hand: {'Found' if has_left_hand else 'Not Found'}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"R.Hand: {'Found' if has_right_hand else 'Not Found'}", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Prediction
        cv2.putText(frame, f"Prediction: {current_prediction}", (10, 190), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Confidence
        if not np.isnan(current_confidence):
            cv2.putText(frame, f"Confidence: {current_confidence:.2%}", (10, 220), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display the frame
        cv2.imshow('Real-Time Sign Language Translator', frame)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print("--- Inference Stopped ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run real-time sign language inference from a webcam.")
    parser.add_argument('--model_path', type=str, 
                        default='models/saved_models/best_model_pretrained_wlasl.h5', 
                        help='Path to the trained .h5 model file.')
    
    args = parser.parse_args()
    realtime_inference(args.model_path)
