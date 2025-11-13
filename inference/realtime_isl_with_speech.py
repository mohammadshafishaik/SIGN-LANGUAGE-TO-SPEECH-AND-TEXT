#!/usr/bin/env python3
"""
ISL Real-Time Recognition with TEXT-TO-SPEECH
- Recognizes ISL signs (A-Z, 1-9)
- Speaks predictions out loud
- Shows visual feedback in webcam
"""

import cv2
import numpy as np
import tensorflow as tf
from pose_extractor.mediapipe_extractor import MediaPipeExtractor
import pyttsx3
import time
import threading

# ===== CONFIGURATION =====
MODEL_PATH = "checkpoints/isl_best.keras"
CLASS_NAMES = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
               'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
               'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Speech settings
CONFIDENCE_THRESHOLD = 0.85  # Only speak if >85% confident
SPEECH_COOLDOWN = 2.0  # Seconds between speeches

# Colors (BGR format)
COLOR_BG = (20, 20, 20)
COLOR_NUMBER = (0, 100, 255)  # Orange
COLOR_LETTER = (0, 255, 100)  # Green
COLOR_TEXT = (255, 255, 255)
COLOR_BAR_BG = (50, 50, 50)

class SpeechEngine:
    """Handles text-to-speech in background thread"""
    
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed
        self.engine.setProperty('volume', 0.9)  # Volume
        self.last_spoken = ""
        self.last_speech_time = 0
        self.speaking = False
        
    def speak(self, text, confidence):
        """Speak text if conditions are met"""
        current_time = time.time()
        
        # Check conditions
        if self.speaking:
            return False
        if confidence < CONFIDENCE_THRESHOLD:
            return False
        if text == self.last_spoken and (current_time - self.last_speech_time) < SPEECH_COOLDOWN:
            return False
        
        # Speak in background thread
        self.last_spoken = text
        self.last_speech_time = current_time
        
        thread = threading.Thread(target=self._speak_async, args=(text,))
        thread.daemon = True
        thread.start()
        return True
    
    def _speak_async(self, text):
        """Internal method to speak asynchronously"""
        self.speaking = True
        try:
            # Format speech text
            if text.isdigit():
                speech_text = f"Number {text}"
            else:
                speech_text = f"Letter {text}"
            
            self.engine.say(speech_text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Speech error: {e}")
        finally:
            self.speaking = False

def draw_ui(frame, predictions, sentence, fps, speech_indicator):
    """Draw beautiful UI on frame"""
    h, w = frame.shape[:2]
    
    # Create overlay panel (right side)
    panel_width = 400
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_width, 0), (w, h), COLOR_BG, -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    y_offset = 30
    
    # Title
    cv2.putText(frame, "ISL RECOGNITION + SPEECH", 
                (w - panel_width + 10, y_offset),
                cv2.FONT_HERSHEY_BOLD, 0.7, COLOR_TEXT, 2)
    y_offset += 40
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", 
                (w - panel_width + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_offset += 30
    
    # Speech indicator
    if speech_indicator:
        cv2.circle(frame, (w - panel_width + 20, y_offset - 5), 8, (0, 255, 0), -1)
        cv2.putText(frame, "SPEAKING", 
                    (w - panel_width + 35, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    y_offset += 30
    
    # Separator
    cv2.line(frame, (w - panel_width + 10, y_offset), 
             (w - 10, y_offset), (100, 100, 100), 1)
    y_offset += 20
    
    # Current sentence
    cv2.putText(frame, "SENTENCE:", 
                (w - panel_width + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_offset += 30
    
    # Display sentence (word wrap)
    sentence_display = sentence if len(sentence) <= 20 else sentence[-20:]
    cv2.rectangle(frame, (w - panel_width + 10, y_offset - 25),
                  (w - 10, y_offset + 10), (50, 50, 50), -1)
    cv2.putText(frame, sentence_display, 
                (w - panel_width + 15, y_offset),
                cv2.FONT_HERSHEY_BOLD, 0.8, (0, 255, 255), 2)
    y_offset += 50
    
    # Separator
    cv2.line(frame, (w - panel_width + 10, y_offset), 
             (w - 10, y_offset), (100, 100, 100), 1)
    y_offset += 20
    
    # Top 5 predictions
    cv2.putText(frame, "TOP PREDICTIONS:", 
                (w - panel_width + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_offset += 25
    
    for i, (label, conf) in enumerate(predictions[:5]):
        # Determine color
        is_number = label.isdigit()
        color = COLOR_NUMBER if is_number else COLOR_LETTER
        prefix = "[NUM]" if is_number else "[LTR]"
        
        # Draw rank
        rank_text = f"{i+1}."
        cv2.putText(frame, rank_text,
                    (w - panel_width + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
        
        # Draw label
        label_text = f"{prefix} {label}"
        cv2.putText(frame, label_text,
                    (w - panel_width + 40, y_offset),
                    cv2.FONT_HERSHEY_BOLD, 0.6, color, 2)
        
        # Draw confidence bar
        bar_x = w - panel_width + 150
        bar_y = y_offset - 12
        bar_width = 200
        bar_height = 15
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_width, bar_y + bar_height),
                      COLOR_BAR_BG, -1)
        
        # Filled bar
        fill_width = int(bar_width * conf)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill_width, bar_y + bar_height),
                      color, -1)
        
        # Percentage text
        conf_text = f"{conf*100:.1f}%"
        cv2.putText(frame, conf_text,
                    (bar_x + bar_width + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
        
        y_offset += 35
    
    # Instructions at bottom
    y_offset = h - 120
    cv2.line(frame, (w - panel_width + 10, y_offset), 
             (w - 10, y_offset), (100, 100, 100), 1)
    y_offset += 25
    
    instructions = [
        "SPACE: Add to sentence",
        "BACKSPACE: Delete last",
        "C: Clear sentence",
        "Q: Quit"
    ]
    
    for instruction in instructions:
        cv2.putText(frame, instruction,
                    (w - panel_width + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        y_offset += 20
    
    return frame

def main():
    print("="*70)
    print("🎤 ISL REAL-TIME RECOGNITION WITH SPEECH")
    print("="*70)
    print()
    
    # Load model
    print("📦 Loading ISL model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
    
    # Initialize components
    print("🎥 Initializing webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("🤖 Initializing MediaPipe...")
    extractor = MediaPipeExtractor(static_image_mode=True)
    
    print("🔊 Initializing speech engine...")
    speech = SpeechEngine()
    
    print("\n✅ All systems ready!")
    print("\n📝 CONTROLS:")
    print("   SPACE      - Add current prediction to sentence")
    print("   BACKSPACE  - Delete last character")
    print("   C          - Clear sentence")
    print("   Q          - Quit")
    print("\n🎤 Speech will activate automatically for confident predictions (>85%)")
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    # State
    sentence = ""
    frame_count = 0
    fps_time = time.time()
    fps = 0
    speech_indicator = False
    speech_indicator_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # Process every 5 frames for stability
        if frame_count % 5 == 0:
            # Extract features
            results = extractor.extract_landmarks(frame)
            landmarks = extractor.get_landmarks_from_results(results)
            
            if landmarks is not None:
                norm_landmarks, _, _ = extractor.normalize_landmarks(landmarks)
                features = norm_landmarks.reshape(1, -1)
                
                # Predict
                preds = model.predict(features, verbose=0)[0]
                
                # Get top 5
                top_indices = np.argsort(preds)[::-1][:5]
                predictions = [(CLASS_NAMES[i], preds[i]) for i in top_indices]
                
                # Auto-speak top prediction if confident
                top_label, top_conf = predictions[0]
                if speech.speak(top_label, top_conf):
                    speech_indicator = True
                    speech_indicator_time = time.time()
            else:
                predictions = [("No hand detected", 0.0)] * 5
        
        # Update speech indicator
        if speech_indicator and (time.time() - speech_indicator_time) > 0.5:
            speech_indicator = False
        
        # Calculate FPS
        if time.time() - fps_time > 1.0:
            fps = frame_count / (time.time() - fps_time)
            frame_count = 0
            fps_time = time.time()
        
        # Draw UI
        frame = draw_ui(frame, predictions, sentence, fps, speech_indicator)
        
        # Show frame
        cv2.imshow("ISL Recognition + Speech", frame)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space - add to sentence
            if predictions[0][1] > 0.5:  # If confident enough
                sentence += predictions[0][0]
                # Speak the added character
                speech.speak(predictions[0][0], 1.0)
                speech_indicator = True
                speech_indicator_time = time.time()
        elif key == 8:  # Backspace
            if len(sentence) > 0:
                sentence = sentence[:-1]
        elif key == ord('c'):  # Clear
            sentence = ""
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    
    print("\n✅ Session ended")
    if sentence:
        print(f"📝 Final sentence: {sentence}")

if __name__ == "__main__":
    main()
