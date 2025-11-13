import os
import sys
import cv2
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
import pyttsx3

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pose_extractor.mediapipe_extractor import MediaPipeExtractor

# --- Constants ---
MAX_FRAMES = 100
TARGET_FPS = 25

# --- Data Processing Functions (mirrors training script) ---

def process_video_file(video_path, extractor, max_frames, target_fps):
    """
    Processes a single video file to extract keypoints.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(original_fps / target_fps) if original_fps > target_fps else 1

        all_keypoints = []
        frame_count = 0
        
        while cap.isOpened() and len(all_keypoints) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Use the correct method name: extract_landmarks
                results = extractor.extract_landmarks(frame)
                # Process results to get a numpy array
                keypoints = extractor.get_landmarks_from_results(results)
                # Normalize the keypoints
                normalized_keypoints, _, _ = extractor.normalize_landmarks(keypoints)
                
                # Append the (75, 3) array directly, without flattening
                all_keypoints.append(normalized_keypoints)
            
            frame_count += 1
        
        cap.release()

        if not all_keypoints:
            return None

        num_frames = len(all_keypoints)
        if num_frames < max_frames:
            last_frame_keypoints = all_keypoints[-1]
            padding = [last_frame_keypoints] * (max_frames - num_frames)
            all_keypoints.extend(padding)
        
        return np.array(all_keypoints)

    except Exception as e:
        print(f"\nERROR processing video {video_path}: {e}")
        return None

def standardize_data(keypoints):
    """
    Applies the same standardization as used in training.
    """
    mean = np.mean(keypoints, axis=(0, 1), keepdims=True)
    std = np.std(keypoints, axis=(0, 1), keepdims=True)
    std[std == 0] = 1e-6
    return (keypoints - mean) / std

# --- Main Inference Function ---

def main(args):
    """
    Runs inference on a single video file.
    """
    print("--- Starting Inference ---")

    # --- 1. Load Model ---
    print(f"Loading model from: {args.model_path}")
    if not os.path.exists(args.model_path):
        print(f"FATAL: Model file not found at {args.model_path}")
        print("Please ensure you have run the training script successfully.")
        return
    
    # The custom transformer_encoder_block is needed for the model to load
    # We define a dummy function here as it's part of the saved model architecture
    def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0):
        pass # This doesn't need to do anything, just exist for loading

    with tf.keras.utils.custom_object_scope({'transformer_encoder_block': transformer_encoder_block}):
        model = tf.keras.models.load_model(args.model_path)
    print("Model loaded successfully.")

    # --- 2. Load Label Mapping ---
    print("Loading label mapping...")
    metadata_path = 'dataset/keypoints_combined/metadata_combined.csv'
    if not os.path.exists(metadata_path):
        print(f"FATAL: Combined metadata not found at {metadata_path}")
        return
        
    df = pd.read_csv(metadata_path)
    # Create a mapping from encoded label (integer) to human-readable label (string)
    label_map = pd.Series(df.label.values, index=df.label_encoded).to_dict()
    print(f"Found {len(label_map)} unique labels.")

    # --- 3. Initialize TTS Engine ---
    print("Initializing Text-to-Speech engine...")
    tts_engine = pyttsx3.init()

    # --- 4. Process Input Video ---
    if not os.path.exists(args.video_path):
        print(f"FATAL: Input video not found at {args.video_path}")
        return

    print(f"Processing video: {args.video_path}")
    extractor = MediaPipeExtractor()
    keypoints = process_video_file(args.video_path, extractor, MAX_FRAMES, TARGET_FPS)

    if keypoints is None:
        print("Could not extract keypoints from the video. Aborting.")
        return

    # --- 5. Standardize and Prepare Data ---
    standardized_keypoints = standardize_data(keypoints)
    # Add a batch dimension
    input_data = np.expand_dims(standardized_keypoints, axis=0)

    # Ensure there are no NaNs in the input data
    input_data = np.nan_to_num(input_data)

    # --- 6. Make Prediction ---
    print("Making prediction...")
    prediction = model.predict(input_data)
    predicted_class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    # --- 7. Decode and Announce Prediction ---
    predicted_label = label_map.get(predicted_class_index, "Unknown")
    
    print("\n--- Prediction Result ---")
    print(f"Predicted Sign: '{predicted_label}'")
    # Handle potential nan confidence
    if not np.isnan(confidence):
        print(f"Confidence: {confidence:.2%}")
    else:
        print("Confidence: Not available")

    # Announce the result
    tts_engine.say(f"I think the sign is {predicted_label}")
    tts_engine.runAndWait()
    
    print("\n--- Inference Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sign language to speech inference on a video file.")
    parser.add_argument('video_path', type=str, help='Path to the input video file.')
    parser.add_argument('--model_path', type=str, 
                        default='models/saved_models/best_model_pretrained_wlasl.h5', 
                        help='Path to the trained .h5 model file.')
    args = parser.parse_args()
    main(args)
