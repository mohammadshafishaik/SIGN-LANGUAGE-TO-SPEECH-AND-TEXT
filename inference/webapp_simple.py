"""
Simple Working Web App for SignSpeak AI (A-Z, 1-9)
Uses the new MediaPipe Tasks API (0.10.21+)
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
import pyttsx3
from pathlib import Path
import json
import time

app = Flask(__name__)

# Global state
class State:
    def __init__(self):
        self.model = None
        self.idx_to_label = {}
        self.hand_landmarker = None
        self.pose_landmarker = None
        self.cap = None
        self.current_frame_jpg = None
        self.predictions = []
        self.sentence = []
        self.speech_enabled = True
        self.volume = 0.8
        self.auto_speak = False
        self.lock = threading.Lock()

state = State()

# Speech lock
speech_lock = threading.Lock()

def speak_async(text):
    """Speak text in background - fixed version"""
    def _speak():
        try:
            with speech_lock:
                # Create fresh engine for each speech
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.setProperty('volume', state.volume)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
        except Exception as e:
            print(f"Speech error: {e}")
    
    if state.speech_enabled and text:
        threading.Thread(target=_speak, daemon=True).start()

def extract_keypoints(image):
    """Extract 144D features using new MediaPipe Tasks API"""
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # Process hands and pose
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
    
    # Extract features
    features = []
    
    # Hand landmarks (126 features = 2 hands * 21 landmarks * 3 coords)
    if hands_result and hands_result.hand_landmarks:
        for hand_landmarks in hands_result.hand_landmarks[:2]:
            for lm in hand_landmarks:
                features.extend([lm.x, lm.y, lm.z])
    
    # Pad if needed
    while len(features) < 126:
        features.extend([0.0] * 21 * 3)
        if len(features) >= 126:
            features = features[:126]
    
    # Pose landmarks (18 features = 6 landmarks * 3 coords)
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

def draw_landmarks(frame, hands_result, pose_result):
    """Draw skeleton on frame using the new API results"""
    h, w, _ = frame.shape
    
    # Draw hands
    if hands_result and hands_result.hand_landmarks:
        # Hand connections (MediaPipe hand topology)
        hand_connections = [
            (0,1),(1,2),(2,3),(3,4),  # Thumb
            (0,5),(5,6),(6,7),(7,8),  # Index
            (0,9),(9,10),(10,11),(11,12),  # Middle
            (0,13),(13,14),(14,15),(15,16),  # Ring
            (0,17),(17,18),(18,19),(19,20),  # Pinky
            (5,9),(9,13),(13,17),  # Palm
        ]
        
        for hand_landmarks in hands_result.hand_landmarks:
            # Draw connections
            for start_idx, end_idx in hand_connections:
                if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            
            # Draw landmarks
            for lm in hand_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                cv2.circle(frame, (x, y), 6, (0, 200, 0), 1)
    
    # Draw pose (upper body)
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
    
    # Load model - try compatible versions first
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
        print(f"✓ Labels: {len(state.idx_to_label)} classes (A-Z, 1-9)")
    elif label_txt_path.exists():
        with open(label_txt_path, 'r') as f:
            labels = [line.strip() for line in f if line.strip()]
        state.idx_to_label = {i: label for i, label in enumerate(labels)}
        print(f"✓ Labels: {len(state.idx_to_label)} loaded from labels.txt")
    else:
        print("✗ No label file found! Please provide label_mappings.json or labels.txt.")
        state.idx_to_label = {}
    
    # Initialize MediaPipe with new Tasks API
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
    else:
        print(f"⚠ Hand model not found at {hand_model_path}")
    
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
    else:
        print(f"⚠ Pose model not found at {pose_model_path}")
    
    # Open webcam
    state.cap = cv2.VideoCapture(0)
    state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("✓ Webcam ready")
    
    print("🎉 System ready!\n")

@app.route('/')
def index():
    return render_template('deploy.html')

@app.route('/get_frame')
def get_frame():
    """Get current frame with predictions"""
    ret, frame = state.cap.read()
    if not ret:
        return jsonify({'error': 'Camera error'}), 500
    
    # Mirror
    frame = cv2.flip(frame, 1)
    
    # Extract features
    features, hands_result, pose_result = extract_keypoints(frame)
    
    # Predict
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
    
    # Draw landmarks
    draw_landmarks(frame, hands_result, pose_result)
    
    # Encode frame
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    
    with state.lock:
        state.predictions = top_preds
    
    return jsonify({
        'frame': frame_b64,
        'predictions': top_preds,
        'sentence': state.sentence
    })

@app.route('/add_word', methods=['POST'])
def add_word():
    """Add top prediction to sentence"""
    with state.lock:
        if state.predictions:
            word = state.predictions[0]['label']
            state.sentence.append(word)
            if state.auto_speak:
                speak_async(word)
    
    return jsonify({'sentence': state.sentence})

@app.route('/delete_last', methods=['POST'])
def delete_last():
    """Delete last word"""
    with state.lock:
        if state.sentence:
            state.sentence.pop()
    
    return jsonify({'sentence': state.sentence})

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    """Clear sentence"""
    with state.lock:
        state.sentence = []
    
    return jsonify({'sentence': state.sentence})

@app.route('/speak_sentence', methods=['POST'])
def speak_sentence():
    """Speak full sentence"""
    with state.lock:
        sentence_text = ' '.join(state.sentence)
    
    if sentence_text:
        speak_async(sentence_text)
    
    return jsonify({'success': True})

@app.route('/speak_word', methods=['POST'])
def speak_word():
    """Speak specific word"""
    data = request.json
    word = data.get('word', '')
    
    if word:
        speak_async(word)
    
    return jsonify({'success': True})

@app.route('/update_settings', methods=['POST'])
def update_settings():
    """Update settings"""
    data = request.json
    
    with state.lock:
        if 'speech_enabled' in data:
            state.speech_enabled = data['speech_enabled']
        if 'volume' in data:
            state.volume = max(0.0, min(1.0, data['volume']))
        if 'auto_speak' in data:
            state.auto_speak = data['auto_speak']
    
    return jsonify({'success': True})

@app.route('/get_latest_prediction', methods=['GET'])
def get_latest_prediction():
    """Get the latest prediction for analytics"""
    with state.lock:
        if len(state.predictions) > 0:
            latest = state.predictions[-1]
            return jsonify({
                'label': latest['label'],
                'confidence': latest['confidence']
            })
        return jsonify({'label': None, 'confidence': 0})

@app.route('/ask_gemini', methods=['POST'])
def ask_gemini():
    """Proxy for Gemini AI to avoid CORS issues"""
    import requests as req
    
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    GEMINI_API_KEY = 'AIzaSyBRS9fcrlIDYN1ySBP0nocriJTm3t70z0g'
    
    try:
        response = req.post(
            f'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}',
            json={
                'contents': [{
                    'parts': [{
                        'text': f'You are a helpful Indian Sign Language (ISL) AI assistant. {question}'
                    }]
                }],
                'generationConfig': {
                    'temperature': 0.7,
                    'maxOutputTokens': 200,
                }
            },
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('candidates') and result['candidates'][0].get('content'):
                text = result['candidates'][0]['content']['parts'][0]['text']
                return jsonify({'answer': text})
            else:
                return jsonify({'error': 'No valid response from Gemini'}), 500
        else:
            return jsonify({'error': f'Gemini API error: {response.status_code}'}), response.status_code
    
    except Exception as e:
        print(f"Gemini error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    
    print("="*70)
    print("🎨 SIGNSPEAK AI - WEB APPLICATION")
    print("="*70)
    print("📱 Open: http://localhost:8080")
    print("🎤 Speech: Enabled")
    print("📹 Webcam: Ready")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=False)
