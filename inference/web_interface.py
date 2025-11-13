#!/usr/bin/env python3
"""
ISL Recognition Web Interface with Speech
- Beautiful web UI showing live predictions
- Real-time video feed
- Speech output
- Sentence builder
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
import time
import json
from queue import Queue
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pose_extractor.mediapipe_extractor import MediaPipeExtractor

app = Flask(__name__)

# ===== CONFIGURATION =====
MODEL_PATH = "checkpoints/isl_best.keras"
CLASS_NAMES = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
               'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
               'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Global state
state = {
    'predictions': [],
    'sentence': '',
    'fps': 0,
    'speaking': False,
    'last_spoken': '',
    'confidence': 0.0
}
state_lock = threading.Lock()

# Speech engine
speech_engine = None
speech_queue = Queue()

def init_speech():
    """Initialize speech engine in separate thread"""
    global speech_engine
    speech_engine = pyttsx3.init()
    speech_engine.setProperty('rate', 150)
    speech_engine.setProperty('volume', 0.9)
    
def speech_worker():
    """Background thread for speech"""
    global speech_engine, state
    init_speech()
    
    while True:
        text = speech_queue.get()
        if text is None:
            break
        
        with state_lock:
            state['speaking'] = True
            state['last_spoken'] = text
        
        try:
            if text.isdigit():
                speech_text = f"Number {text}"
            else:
                speech_text = f"Letter {text}"
            speech_engine.say(speech_text)
            speech_engine.runAndWait()
        except Exception as e:
            print(f"Speech error: {e}")
        
        with state_lock:
            state['speaking'] = False
        
        time.sleep(0.5)

# Start speech thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

# Load model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
extractor = MediaPipeExtractor()
print("Model loaded!")

def generate_frames():
    """Generate video frames with predictions"""
    global state
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_count = 0
    fps_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # Process every 5 frames
        predictions = []
        if frame_count % 5 == 0:
            try:
                results = extractor.extract_landmarks(frame)
                landmarks = extractor.get_landmarks_from_results(results)
                
                if landmarks is not None:
                    norm_landmarks, _, _ = extractor.normalize_landmarks(landmarks)
                    features = norm_landmarks.reshape(1, -1)
                    
                    preds = model.predict(features, verbose=0)[0]
                    top_indices = np.argsort(preds)[::-1][:5]
                    predictions = [
                        {
                            'label': CLASS_NAMES[i],
                            'confidence': float(preds[i]),
                            'type': 'number' if CLASS_NAMES[i].isdigit() else 'letter'
                        }
                        for i in top_indices
                    ]
                    
                    with state_lock:
                        state['predictions'] = predictions
                        state['confidence'] = predictions[0]['confidence']
                    
                    # Draw landmarks on frame
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            extractor.mp_draw.draw_landmarks(
                                frame, hand_landmarks, extractor.mp_hands.HAND_CONNECTIONS
                            )
                else:
                    with state_lock:
                        state['predictions'] = []
                        state['confidence'] = 0.0
            except Exception as e:
                print(f"Frame processing error: {e}")
                with state_lock:
                    state['predictions'] = []
                    state['confidence'] = 0.0
        
        # Calculate FPS
        if time.time() - fps_time > 1.0:
            fps = frame_count / (time.time() - fps_time)
            with state_lock:
                state['fps'] = fps
            frame_count = 0
            fps_time = time.time()
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_state')
def get_state():
    """Get current state (predictions, sentence, etc.)"""
    with state_lock:
        return jsonify(state)

@app.route('/add_to_sentence')
def add_to_sentence():
    """Add current prediction to sentence"""
    with state_lock:
        if state['predictions'] and state['predictions'][0]['confidence'] > 0.5:
            label = state['predictions'][0]['label']
            state['sentence'] += label
            speech_queue.put(label)
            return jsonify({'success': True, 'sentence': state['sentence']})
    return jsonify({'success': False})

@app.route('/delete_last')
def delete_last():
    """Delete last character from sentence"""
    with state_lock:
        if state['sentence']:
            state['sentence'] = state['sentence'][:-1]
        return jsonify({'success': True, 'sentence': state['sentence']})

@app.route('/clear_sentence')
def clear_sentence():
    """Clear entire sentence"""
    with state_lock:
        state['sentence'] = ''
    return jsonify({'success': True})

@app.route('/speak_sentence')
def speak_sentence():
    """Speak entire sentence"""
    with state_lock:
        sentence = state['sentence']
    if sentence:
        speech_queue.put(sentence)
    return jsonify({'success': True})

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌐 ISL RECOGNITION WEB INTERFACE")
    print("="*70)
    print("\n🚀 Starting server...")
    print("\n📱 Open in browser: http://localhost:8080")
    print("\n⚡ Features:")
    print("   - Live video feed with hand tracking")
    print("   - Real-time predictions with confidence")
    print("   - Sentence builder")
    print("   - Text-to-speech")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=False, threaded=True, host='0.0.0.0', port=8080)
