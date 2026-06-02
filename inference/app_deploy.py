"""
Deployment-Ready Web App for SignSpeak AI (A-Z, 1-9)
Uses browser webcam (client-side) and server-side model inference.
Works on cloud platforms like Render, Railway, etc.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe import tasks
import tensorflow as tf
from flask import Flask, render_template, jsonify, request
import threading
import base64
from pathlib import Path
import json
import time
import os

app = Flask(__name__)

# Global state
class State:
    def __init__(self):
        self.model = None
        self.idx_to_label = {}
        self.hand_landmarker = None
        self.pose_landmarker = None
        self.predictions = []
        self.lock = threading.Lock()

state = State()


def extract_keypoints_from_frame(frame):
    """Extract 144D features using new MediaPipe Tasks API"""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    hands_result = None
    pose_result = None
    
    try:
        if state.hand_landmarker is not None:
            hands_result = state.hand_landmarker.detect(mp_image)
    except Exception as e:
        print(f"Hand detection error: {e}")
    
    try:
        if state.pose_landmarker is not None:
            pose_result = state.pose_landmarker.detect(mp_image)
    except Exception as e:
        print(f"Pose detection error: {e}")
    
    features = []
    
    # Hand landmarks (126 features)
    if hands_result and hands_result.hand_landmarks:
        for hand_landmarks in hands_result.hand_landmarks[:2]:
            for lm in hand_landmarks:
                features.extend([lm.x, lm.y, lm.z])
    
    while len(features) < 126:
        features.extend([0.0] * 21 * 3)
        if len(features) >= 126:
            features = features[:126]
    
    # Pose landmarks (18 features)
    if pose_result and pose_result.pose_landmarks:
        pose_indices = [11, 12, 13, 14, 15, 16]
        for idx in pose_indices:
            if idx < len(pose_result.pose_landmarks[0]):
                lm = pose_result.pose_landmarks[0][idx]
                features.extend([lm.x, lm.y, lm.z])
            else:
                features.extend([0.0, 0.0, 0.0])
    else:
        features.extend([0.0] * 18)
    
    return np.array(features[:144], dtype=np.float32), hands_result, pose_result


def draw_landmarks_on_frame(frame, hands_result, pose_result):
    """Draw skeleton on frame"""
    h, w, _ = frame.shape
    
    if hands_result and hands_result.hand_landmarks:
        hand_connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17),
        ]
        
        for hand_landmarks in hands_result.hand_landmarks:
            for start_idx, end_idx in hand_connections:
                if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            
            for lm in hand_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
    
    if pose_result and pose_result.pose_landmarks:
        pose_connections = [(11,12),(11,13),(13,15),(12,14),(14,16)]
        landmarks = pose_result.pose_landmarks[0]
        
        for start_idx, end_idx in pose_connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (200, 0, 0), 2)
        
        for idx in [11, 12, 13, 14, 15, 16]:
            if idx < len(landmarks):
                lm = landmarks[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)


def load_model():
    """Load SignSpeak AI model"""
    print("🚀 Loading SignSpeak AI System...")
    print("="*60)
    
    base_dir = Path(__file__).parent.parent / "checkpoints"
    model_candidates = [
        base_dir / "isl_best_v2.keras",
        base_dir / "isl_best_fixed.keras",
        base_dir / "isl_best.keras",
    ]
    
    model_path = None
    for candidate in model_candidates:
        if candidate.exists():
            model_path = candidate
            break
    
    if model_path is None:
        raise FileNotFoundError("No model found in checkpoints/")
    
    try:
        state.model = tf.keras.models.load_model(str(model_path))
    except Exception:
        state.model = tf.keras.models.load_model(str(model_path), compile=False)
    print(f"✓ Model: {model_path}")
    
    # Load labels
    label_json_path = Path(__file__).parent.parent / "dataset" / "splits_isl" / "label_mappings.json"
    label_txt_path = Path(__file__).parent.parent / "checkpoints" / "labels.txt"
    if label_json_path.exists():
        with open(label_json_path, 'r') as f:
            label_info = json.load(f)
        state.idx_to_label = {int(k): v for k, v in label_info['idx_to_label'].items()}
        print(f"✓ Labels: {len(state.idx_to_label)} classes")
    elif label_txt_path.exists():
        with open(label_txt_path, 'r') as f:
            labels = [line.strip() for line in f if line.strip()]
        state.idx_to_label = {i: label for i, label in enumerate(labels)}
        print(f"✓ Labels: {len(state.idx_to_label)} loaded from labels.txt")
    else:
        print("✗ No label file found!")
        state.idx_to_label = {}
    
    # Initialize MediaPipe
    mp_models_dir = Path(__file__).parent.parent / "models" / "mediapipe"
    
    hand_model_path = mp_models_dir / "hand_landmarker.task"
    if hand_model_path.exists():
        hand_options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(hand_model_path)),
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        state.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        print("✓ Hand Landmarker initialized")
    
    pose_model_path = mp_models_dir / "pose_landmarker.task"
    if pose_model_path.exists():
        pose_options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_model_path)),
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        state.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
        print("✓ Pose Landmarker initialized")
    
    print("🎉 System ready!\n")


@app.route('/')
def index():
    return render_template('deploy.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Receive a frame from the browser webcam, process it, and return predictions + annotated frame"""
    data = request.json
    if not data or 'frame' not in data:
        return jsonify({'error': 'No frame data'}), 400
    
    try:
        # Decode base64 frame from browser
        frame_data = data['frame'].split(',')[1] if ',' in data['frame'] else data['frame']
        frame_bytes = base64.b64decode(frame_data)
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid frame'}), 400
        
        # Extract features and predict
        features, hands_result, pose_result = extract_keypoints_from_frame(frame)
        predictions = state.model.predict(features.reshape(1, -1), verbose=0)[0]
        
        # Get top 5
        top_indices = np.argsort(predictions)[::-1][:5]
        top_preds = [
            {
                'label': state.idx_to_label.get(idx, f'class_{idx}'),
                'confidence': float(predictions[idx])
            }
            for idx in top_indices
        ]
        
        # Draw landmarks on frame
        draw_landmarks_on_frame(frame, hands_result, pose_result)
        
        # Encode annotated frame back
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'frame': frame_b64,
            'predictions': top_preds
        })
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check for deployment"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': state.model is not None,
        'labels': len(state.idx_to_label)
    })


if __name__ == '__main__':
    load_model()
    
    port = int(os.environ.get('PORT', 8080))
    
    print("="*70)
    print("🎨 SIGNSPEAK AI - DEPLOYMENT MODE")
    print("="*70)
    print(f"📱 Open: http://localhost:{port}")
    print("📹 Using browser webcam (client-side)")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
