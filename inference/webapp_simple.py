"""
Simple Working Web App for SignSpeak AI (A-Z, 1-9)
Uses the working realtime_isl_FIXED.py logic
"""

import cv2
import numpy as np
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    mp = None
    HAS_MEDIAPIPE = False
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
        self.mp_hands = None
        self.mp_pose = None
        self.hands = None
        self.pose = None
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
    """Extract 144D features - same as training"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process (guard if MediaPipe is unavailable)
    hands_results = None
    pose_results = None
    if HAS_MEDIAPIPE and state.hands is not None and state.pose is not None:
        try:
            hands_results = state.hands.process(image_rgb)
            pose_results = state.pose.process(image_rgb)
        except Exception as e:
            # Fallback to None if processing fails
            print(f"MediaPipe processing error: {e}")
            hands_results = None
            pose_results = None
    
    # Extract features
    features = []
    
    # Hand landmarks (126 features)
    if hands_results and getattr(hands_results, 'multi_hand_landmarks', None):
        for hand_landmarks in hands_results.multi_hand_landmarks[:2]:
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
    
    # Pad if needed
    while len(features) < 126:
        features.extend([0.0] * 21 * 3)
        if len(features) >= 126:
            features = features[:126]
    
    # Pose landmarks (18 features)
    if pose_results and getattr(pose_results, 'pose_landmarks', None):
        pose_indices = [11, 12, 13, 14, 15, 16]
        for idx in pose_indices:
            lm = pose_results.pose_landmarks.landmark[idx]
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 18)
    
    return np.array(features[:144], dtype=np.float32), hands_results, pose_results

def draw_landmarks(frame, hands_results, pose_results):
    """Draw skeleton on frame"""
    if not HAS_MEDIAPIPE or mp is None:
        return
    mp_drawing = mp.solutions.drawing_utils
    
    # Draw hands
    if hands_results and getattr(hands_results, 'multi_hand_landmarks', None):
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, state.mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2)
            )
    
    # Draw pose
    if pose_results and getattr(pose_results, 'pose_landmarks', None):
        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks, state.mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(200, 0, 0), thickness=2)
        )

def load_model():
    """Load SignSpeak AI model"""
    print("🚀 Loading SignSpeak AI System...")
    print("="*60)
    
    # Load model
    model_path = Path(__file__).parent.parent / "checkpoints" / "isl_best.keras"
    state.model = tf.keras.models.load_model(str(model_path))
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
    
    # Initialize MediaPipe
    state.mp_hands = mp.solutions.hands
    state.mp_pose = mp.solutions.pose
    state.hands = state.mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    )
    state.pose = state.mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5
    )
    print("✓ MediaPipe initialized")
    
    # Open webcam
    state.cap = cv2.VideoCapture(0)
    state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("✓ Webcam ready")
    
    print("🎉 System ready!\n")

@app.route('/')
def index():
    return render_template('webapp.html')

@app.route('/get_frame')
def get_frame():
    """Get current frame with predictions"""
    ret, frame = state.cap.read()
    if not ret:
        return jsonify({'error': 'Camera error'}), 500
    
    # Mirror
    frame = cv2.flip(frame, 1)
    
    # Extract features
    features, hands_results, pose_results = extract_keypoints(frame)
    
    # Predict
    predictions = state.model.predict(features.reshape(1, -1), verbose=0)[0]
    
    # Get top 5
    top_indices = np.argsort(predictions)[::-1][:5]
    top_preds = [
        {
            'label': state.idx_to_label[idx],
            'confidence': float(predictions[idx])
        }
        for idx in top_indices
    ]
    
    # Draw landmarks
    draw_landmarks(frame, hands_results, pose_results)
    
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
    import requests
    
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    GEMINI_API_KEY = 'AIzaSyBRS9fcrlIDYN1ySBP0nocriJTm3t70z0g'
    
    try:
        response = requests.post(
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
