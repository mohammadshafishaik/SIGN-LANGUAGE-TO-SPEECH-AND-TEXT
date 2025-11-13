#!/usr/bin/env python3
"""
Simplified ISL Web Interface with Speech
Shows webcam + predictions + speech
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
import sys
import os

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pose_extractor.mediapipe_extractor import MediaPipeExtractor

app = Flask(__name__)

# Configuration
MODEL_PATH = "checkpoints/isl_best.keras"
CLASS_NAMES = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
               'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
               'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Load model and extractor once
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
extractor = MediaPipeExtractor()
print("✅ Model loaded!")

# Global state
current_predictions = []
current_sentence = ""

def generate_frames():
    """Generate video frames"""
    global current_predictions
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # Process every 5 frames
        if frame_count % 5 == 0:
            try:
                results = extractor.extract_landmarks(frame)
                landmarks = extractor.get_landmarks_from_results(results)
                
                if landmarks is not None:
                    norm_landmarks, _, _ = extractor.normalize_landmarks(landmarks)
                    features = norm_landmarks.reshape(1, -1)
                    
                    preds = model.predict(features, verbose=0)[0]
                    top_indices = np.argsort(preds)[::-1][:5]
                    
                    current_predictions = [
                        {
                            'label': CLASS_NAMES[i],
                            'confidence': float(preds[i]),
                            'type': 'number' if CLASS_NAMES[i].isdigit() else 'letter'
                        }
                        for i in top_indices
                    ]
                    
                    # Draw landmarks
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            extractor.mp_draw.draw_landmarks(
                                frame, hand_landmarks, extractor.mp_hands.HAND_CONNECTIONS
                            )
                else:
                    current_predictions = []
            except Exception as e:
                print(f"Error: {e}")
                current_predictions = []
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index_simple.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_predictions')
def get_predictions():
    return jsonify({
        'predictions': current_predictions,
        'sentence': current_sentence
    })

@app.route('/add_prediction')
def add_prediction():
    global current_sentence
    if current_predictions and current_predictions[0]['confidence'] > 0.5:
        current_sentence += current_predictions[0]['label']
    return jsonify({'sentence': current_sentence})

@app.route('/delete_last')
def delete_last():
    global current_sentence
    if current_sentence:
        current_sentence = current_sentence[:-1]
    return jsonify({'sentence': current_sentence})

@app.route('/clear')
def clear():
    global current_sentence
    current_sentence = ""
    return jsonify({'sentence': current_sentence})

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🎥 ISL WEB INTERFACE (SIMPLE)")
    print("="*70)
    print("\n📱 Open: http://localhost:8080\n")
    
    app.run(debug=False, threaded=True, host='0.0.0.0', port=8080)
