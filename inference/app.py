"""
Stunning ISL Recognition Web App
Beautiful, glossy UI with speech output and all controls
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, render_template, Response, jsonify, request
import threading
import queue
import time
import base64
import pyttsx3
from collections import deque
from pathlib import Path
from project_paths import CHECKPOINTS_DIR, ensure_default_dirs

app = Flask(__name__)

# Global state
class AppState:
    def __init__(self):
        self.model = None
        self.labels = []
        self.mp_hands = None
        self.mp_pose = None
        self.hands = None
        self.pose = None
        self.cap = None
        self.current_frame = None
        self.predictions = []
        self.sentence = []
        self.is_speaking = False
        self.speech_enabled = True
        self.volume = 0.8
        self.speech_threshold = 0.75
        self.auto_speak = True
        self.last_spoken_time = 0
        self.speech_cooldown = 2.0
        self.prediction_history = deque(maxlen=5)  # Smooth predictions
        self.lock = threading.Lock()
        
state = AppState()

# Speech engine (runs in separate thread)
speech_queue = queue.Queue()

def speech_worker():
    """Background thread for speech output"""
    engine = pyttsx3.init()
    
    while True:
        try:
            text, volume = speech_queue.get(timeout=1)
            if text is None:
                break
                
            engine.setProperty('rate', 150)
            engine.setProperty('volume', volume)
            state.is_speaking = True
            engine.say(text)
            engine.runAndWait()
            state.is_speaking = False
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Speech error: {e}")
            state.is_speaking = False

# Start speech thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def speak_text(text):
    """Add text to speech queue"""
    if state.speech_enabled and not state.is_speaking:
        speech_queue.put((text, state.volume))

def load_model():
    """Load ISL model and initialize MediaPipe"""
    print("🚀 Loading ISL Recognition System...")
    
    # Load model
    ensure_default_dirs()
    model_path = CHECKPOINTS_DIR / 'isl_best.keras'
    state.model = tf.keras.models.load_model(str(model_path))
    print(f"✓ Model loaded: {model_path}")
    
    # Load labels
    labels_path = CHECKPOINTS_DIR / 'labels.txt'
    with open(labels_path, 'r') as f:
        state.labels = [line.strip() for line in f.readlines()]
    print(f"✓ Labels loaded: {len(state.labels)} classes")
    
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
    
    # Initialize webcam
    state.cap = cv2.VideoCapture(0)
    if not state.cap.isOpened():
        raise Exception("Cannot open webcam")
    state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("✓ Webcam initialized")
    
    print("🎉 System ready!")

def extract_features(image):
    """Extract 144D features from image"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    
    # Process with MediaPipe
    hand_results = state.hands.process(image_rgb)
    pose_results = state.pose.process(image_rgb)
    
    # Extract features (144D)
    features = []
    
    # Hand landmarks (21 points × 3 coords × 2 hands = 126D)
    for hand_idx in range(2):
        if hand_results.multi_hand_landmarks and hand_idx < len(hand_results.multi_hand_landmarks):
            landmarks = hand_results.multi_hand_landmarks[hand_idx]
            for lm in landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * 63)
    
    # Upper body pose (6 points × 3 coords = 18D)
    if pose_results.pose_landmarks:
        pose_indices = [11, 12, 13, 14, 15, 16]  # Shoulders, elbows, wrists
        for idx in pose_indices:
            lm = pose_results.pose_landmarks.landmark[idx]
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 18)
    
    return np.array(features, dtype=np.float32)

def process_frame():
    """Process current frame and update predictions"""
    ret, frame = state.cap.read()
    if not ret:
        return
    
    # Mirror frame
    frame = cv2.flip(frame, 1)
    
    # Extract features
    features = extract_features(frame)
    
    # Get predictions
    predictions = state.model.predict(features.reshape(1, -1), verbose=0)[0]
    
    # Get top 5
    top_indices = np.argsort(predictions)[::-1][:5]
    top_predictions = [
        {
            'label': state.labels[idx],
            'confidence': float(predictions[idx])
        }
        for idx in top_indices
    ]
    
    # Smooth predictions
    state.prediction_history.append(top_predictions[0])
    
    # Auto-speak if enabled and confident
    if state.auto_speak and len(state.prediction_history) >= 3:
        recent_labels = [p['label'] for p in state.prediction_history]
        if len(set(recent_labels)) == 1:  # All same prediction
            top_pred = top_predictions[0]
            current_time = time.time()
            
            if (top_pred['confidence'] >= state.speech_threshold and 
                current_time - state.last_spoken_time >= state.speech_cooldown):
                speak_text(top_pred['label'])
                state.last_spoken_time = current_time
    
    # Draw skeleton on frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = state.hands.process(frame_rgb)
    pose_results = state.pose.process(frame_rgb)
    
    # Draw hand landmarks
    if hand_results.multi_hand_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, state.mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2)
            )
    
    # Draw pose landmarks
    if pose_results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks, state.mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(200, 0, 0), thickness=2)
        )
    
    # Store state
    with state.lock:
        state.current_frame = frame
        state.predictions = top_predictions

@app.route('/')
def index():
    return render_template('app.html')

@app.route('/get_frame')
def get_frame():
    """Get current frame as base64"""
    process_frame()
    
    with state.lock:
        if state.current_frame is not None:
            _, buffer = cv2.imencode('.jpg', state.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            return jsonify({
                'frame': frame_base64,
                'predictions': state.predictions,
                'sentence': state.sentence,
                'is_speaking': state.is_speaking
            })
    
    return jsonify({'error': 'No frame available'}), 500

@app.route('/get_state')
def get_state():
    """Get current app state"""
    with state.lock:
        return jsonify({
            'predictions': state.predictions,
            'sentence': state.sentence,
            'is_speaking': state.is_speaking,
            'speech_enabled': state.speech_enabled,
            'volume': state.volume,
            'speech_threshold': state.speech_threshold,
            'auto_speak': state.auto_speak
        })

@app.route('/add_to_sentence', methods=['POST'])
def add_to_sentence():
    """Add prediction to sentence"""
    data = request.json
    label = data.get('label')
    
    with state.lock:
        if label and label not in ['', ' ']:
            state.sentence.append(label)
    
    return jsonify({'sentence': state.sentence})

@app.route('/delete_last', methods=['POST'])
def delete_last():
    """Delete last word from sentence"""
    with state.lock:
        if state.sentence:
            state.sentence.pop()
    
    return jsonify({'sentence': state.sentence})

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    """Clear entire sentence"""
    with state.lock:
        state.sentence = []
    
    return jsonify({'sentence': state.sentence})

@app.route('/speak_sentence', methods=['POST'])
def speak_sentence():
    """Speak the full sentence"""
    with state.lock:
        sentence_text = ' '.join(state.sentence)
    
    if sentence_text:
        speak_text(sentence_text)
    
    return jsonify({'success': True})

@app.route('/speak_word', methods=['POST'])
def speak_word():
    """Speak a specific word"""
    data = request.json
    word = data.get('word')
    
    if word:
        speak_text(word)
    
    return jsonify({'success': True})

@app.route('/update_settings', methods=['POST'])
def update_settings():
    """Update app settings"""
    data = request.json
    
    with state.lock:
        if 'speech_enabled' in data:
            state.speech_enabled = data['speech_enabled']
        if 'volume' in data:
            state.volume = max(0.0, min(1.0, data['volume']))
        if 'speech_threshold' in data:
            state.speech_threshold = max(0.0, min(1.0, data['speech_threshold']))
        if 'auto_speak' in data:
            state.auto_speak = data['auto_speak']
    
    return jsonify({'success': True})

if __name__ == '__main__':
    load_model()
    print("\n" + "="*60)
    print("🎨 ISL RECOGNITION WEB APP")
    print("="*60)
    print("📱 Open: http://localhost:8080")
    print("🎤 Speech: Enabled")
    print("📹 Webcam: Ready")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
