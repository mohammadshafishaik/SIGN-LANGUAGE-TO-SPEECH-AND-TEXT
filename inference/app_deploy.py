"""
Deployment-Ready Web App for SignSpeak AI (A-Z, 1-9)
Uses browser webcam (client-side) and server-side model inference.
Works on cloud platforms like Render, Railway, etc.

Architecture:
- Browser captures video at full quality and shows it directly
- Only small JPEG snapshots are sent for inference (~100ms apart)
- Server returns predictions + landmark coordinates (NOT re-encoded frames)
- Browser draws landmarks on a transparent canvas overlay
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
        self.lock = threading.Lock()

state = State()


def extract_keypoints_from_frame(frame):
    """Extract 144D features using MediaPipe Tasks API.
    Returns features array, hands_result, pose_result."""
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


def extract_landmark_coords(hands_result, pose_result):
    """Extract landmark coordinates as lightweight JSON-serializable lists.
    Returns dict with 'hands' and 'pose' keys containing [x, y] pairs."""
    landmarks = {}

    # Hands: list of hands, each hand is a list of [x, y] pairs
    if hands_result and hands_result.hand_landmarks:
        landmarks['hands'] = []
        for hand in hands_result.hand_landmarks[:2]:
            landmarks['hands'].append([[round(lm.x, 4), round(lm.y, 4)] for lm in hand])

    # Pose: list of [x, y] for the 6 upper-body landmarks (indices 11-16)
    if pose_result and pose_result.pose_landmarks:
        pose_pts = []
        for idx in [11, 12, 13, 14, 15, 16]:
            if idx < len(pose_result.pose_landmarks[0]):
                lm = pose_result.pose_landmarks[0][idx]
                pose_pts.append([round(lm.x, 4), round(lm.y, 4)])
        if pose_pts:
            landmarks['pose'] = pose_pts

    return landmarks if landmarks else None


def load_model():
    """Load SignSpeak AI model and MediaPipe landmarkers."""
    print("Loading SignSpeak AI System...")
    print("=" * 60)

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
    print(f"  Model: {model_path}")

    # Load labels
    label_json_path = Path(__file__).parent.parent / "dataset" / "splits_isl" / "label_mappings.json"
    label_txt_path = Path(__file__).parent.parent / "checkpoints" / "labels.txt"
    if label_json_path.exists():
        with open(label_json_path, 'r') as f:
            label_info = json.load(f)
        state.idx_to_label = {int(k): v for k, v in label_info['idx_to_label'].items()}
        print(f"  Labels: {len(state.idx_to_label)} classes")
    elif label_txt_path.exists():
        with open(label_txt_path, 'r') as f:
            labels = [line.strip() for line in f if line.strip()]
        state.idx_to_label = {i: label for i, label in enumerate(labels)}
        print(f"  Labels: {len(state.idx_to_label)} loaded from labels.txt")
    else:
        print("  No label file found!")
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
        print("  Hand Landmarker initialized")

    pose_model_path = mp_models_dir / "pose_landmarker.task"
    if pose_model_path.exists():
        pose_options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_model_path)),
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        state.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
        print("  Pose Landmarker initialized")

    print("  System ready!\n")


@app.route('/')
def index():
    return render_template('deploy.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Receive a JPEG frame, run inference, return predictions + landmark coords.

    Unlike the old approach, we do NOT re-encode the annotated frame.
    Instead we return lightweight landmark coordinates that the browser
    draws on a transparent canvas overlay. This cuts response size by ~95%
    and eliminates the server-side cv2.imencode bottleneck.
    """
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

        # Get top 5 predictions
        top_indices = np.argsort(predictions)[::-1][:5]
        top_preds = [
            {
                'label': state.idx_to_label.get(idx, f'class_{idx}'),
                'confidence': float(predictions[idx])
            }
            for idx in top_indices
        ]

        # Extract landmark coordinates (tiny JSON, not a re-encoded image)
        landmark_data = extract_landmark_coords(hands_result, pose_result)

        response = {'predictions': top_preds}
        if landmark_data:
            response['landmarks'] = landmark_data

        return jsonify(response)

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check for deployment platforms."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': state.model is not None,
        'labels': len(state.idx_to_label)
    })


if __name__ == '__main__':
    load_model()

    port = int(os.environ.get('PORT', 8080))

    print("=" * 60)
    print("  SIGNSPEAK AI - DEPLOYMENT MODE")
    print("=" * 60)
    print(f"  Open: http://localhost:{port}")
    print("  Camera: browser-side (full quality)")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
