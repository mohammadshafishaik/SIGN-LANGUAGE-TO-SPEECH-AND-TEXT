# 🤟 ISL Recognition - Stunning Web Interface

**Real-time Indian Sign Language Recognition with AI-powered Speech Output**

## 🎨 Features

### ✨ Beautiful UI
- **Glossy glass-morphism design** with gradient purple theme
- **Animated background** with floating particles
- **Real-time video feed** with skeleton overlay
- **Smooth animations** and transitions
- **Responsive design** for all screen sizes

### 🎤 Speech Controls
- **Text-to-speech output** for predictions
- **Volume control** (0-100%)
- **Confidence threshold** slider (50-95%)
- **Auto-speak mode** for confident predictions
- **Click-to-speak** individual words
- **Speak full sentences** with one click

### 📊 Real-time Recognition
- **Top 5 predictions** with confidence scores
- **Color-coded confidence bars** (green/orange/red)
- **Smooth prediction smoothing** (reduces jitter)
- **Live skeleton visualization** (hands + upper body)
- **Status indicators** (recognizing/speaking)

### ✍️ Sentence Builder
- **Add predictions** to build sentences
- **Delete last word** or clear all
- **Interactive word chips** (click to speak)
- **Keyboard shortcuts** for fast interaction

## 🚀 Quick Start

### 1. Start the Web App

```bash
cd "$PROJECT_ROOT"  # From the project root
source venv/bin/activate
python inference/app.py
```

### 2. Open in Browser

Visit: **http://localhost:8080**

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Space** | Add top prediction to sentence |
| **Backspace** | Delete last word |
| **S** | Speak full sentence |
| **C** | Clear sentence |

## 🎯 Current Model

### WLASL Dataset
Currently recognizing **25 words**:
- all, before, black, book, candy
- chair, clothes, computer, cool, cousin
- deaf, drink, fine, finish, go
- help, hot, like, many, no
- thin, walk, who, year, yes

### Model Architecture
- **Input**: Video sequences → MediaPipe keypoints
- **Features**: 144D (hand + pose landmarks)
- **Model**: LSTM (Bidirectional)
- **Output**: Word classification

## 📈 Training on More Words

### Step 1: Extract Keypoints from Videos

```bash
python data_prep/extract_wlasl_keypoints.py
```

This processes all videos in `dataset/raw/` and extracts keypoint sequences.

**What it does:**
- Loads each video file
- Extracts hand + pose landmarks for each frame
- Saves sequences as `.npy` files in `dataset/keypoints_wlasl/`

**Expected output:**
```
📁 Found 143 videos
🔤 Words to process: 25
🚀 Processing videos...
✅ PREPROCESSING COMPLETE!
✓ Processed: 143 videos
```

### Step 2: Train LSTM Model

```bash
python models/train_wlasl.py
```

**What it does:**

**Model Architecture:**
```
- Masking Layer (ignore padding)
- Bidirectional LSTM (256 units)
- Bidirectional LSTM (128 units)
# Now handled automatically via project_paths.CHECKPOINTS_DIR
- Bidirectional LSTM (64 units)
- Dense (128 units)
- Dense (64 units)
- Output (num_classes)
```

**Expected results:**
- Training accuracy: ~90-95%
- Validation accuracy: ~85-90%
- Training time: ~30-60 minutes
cd "$PROJECT_ROOT"  # From the project root
### Step 3: Update Web App

After training, update `inference/app.py`:

```python
# Change model path (line ~88)
model_path = '/Users/shaikshafi/Documents/ML PROJECT/checkpoints/wlasl_best.keras'

# Change labels path (line ~92)
labels_path = '/Users/shaikshafi/Documents/ML PROJECT/checkpoints/wlasl_labels.txt'
```

Then restart the server and you'll have word-level recognition!

## 📥 Download More WLASL Data

To expand beyond 25 words, download more WLASL videos:

```bash
python data_collector/download_wlasl.py --num_words 100
```

**Options:**
- `--num_words 100`: Download 100 most common words (~2GB)
- `--num_words 300`: Download 300 words (~6GB)
- `--num_words 2000`: Download full dataset (~40GB)

Then repeat Steps 1-3 above to train on the larger dataset.

## 🎨 UI Customization

### Change Theme Colors

Edit `inference/templates/app.html`:

```css
/* Line ~16: Background gradient */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* Line ~299: Confidence bar colors */
.confidence-high { background: linear-gradient(90deg, #10b981, #34d399); }
.confidence-med { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.confidence-low { background: linear-gradient(90deg, #ef4444, #f87171); }
```

### Adjust Speech Settings

Default values in `inference/app.py`:

```python
self.volume = 0.8              # 80% volume
self.speech_threshold = 0.75   # Speak when >75% confident
self.speech_cooldown = 2.0     # 2 seconds between auto-speech
```

## 🔧 Technical Details

### Architecture

```
Browser (HTML/CSS/JS)
    ↓ AJAX polling (10 FPS)
Flask Server (Python)
    ↓ processes frames
MediaPipe (Hand + Pose)
    ↓ extracts 144D features
LSTM Model (Keras)
    ↓ predicts word
Text-to-Speech (pyttsx3)
    ↓ speaks result
```

### Why AJAX Polling Instead of Streaming?

Previous attempts with Flask video streaming caused MediaPipe threading conflicts:
- Flask spawns multiple threads for requests
- MediaPipe expects sequential frame processing
- Result: "Packet timestamp mismatch" errors

**Solution:** AJAX polling at 10 FPS
- Browser requests frame every 100ms
- Flask processes frame synchronously
- No threading conflicts
- Smooth, reliable operation

### Feature Extraction

**144D feature vector:**
- Hand landmarks: 21 points × 3 coords × 2 hands = **126D**
- Pose landmarks: 6 points × 3 coords = **18D** (shoulders, elbows, wrists)
- Total: **144D** per frame

**Sequence processing:**
- Videos → Fixed-length sequences (50 frames)
- Shorter videos: Padded with zeros
- Longer videos: Evenly sampled
- Masking layer: Ignores padding during training

## 📊 Model Performance

### Current (25 words)
- **Validation Accuracy**: TBD (need to train)
- **Inference Speed**: ~10 FPS
- **Model Size**: ~5-10 MB

### Expected (100 words)
- **Validation Accuracy**: 85-90%
- **Training Time**: ~1 hour
- **Dataset Size**: ~2GB

### Expected (300 words)
- **Validation Accuracy**: 80-85%
- **Training Time**: ~3-4 hours
- **Dataset Size**: ~6GB

## 🐛 Troubleshooting

### Port already in use
```bash
# Change port in app.py (line ~341)
app.run(host='0.0.0.0', port=8081, debug=False, threaded=True)
```

### Webcam not detected
```bash
# Test webcam
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

# Try different camera index
# Edit app.py line ~105
state.cap = cv2.VideoCapture(1)  # Try 1 instead of 0
```

### Speech not working (macOS)
```bash
# Reinstall pyttsx3 with macOS support
pip uninstall pyttsx3
pip install pyttsx3 pyobjc
```

### Low recognition accuracy
- Ensure good lighting
- Keep hand clearly visible
- Use correct finger orientations
- Train with more data
- Adjust confidence threshold

## 📚 Project Structure

```
ML PROJECT/
├── inference/
│   ├── app.py                  # Main Flask web app
│   └── templates/
│       └── app.html            # Stunning UI
├── models/
│   └── train_wlasl.py          # LSTM training script
├── data_prep/
│   └── extract_wlasl_keypoints.py  # Video preprocessing
├── data_collector/
│   └── download_wlasl.py       # Download more data
├── dataset/
│   ├── raw/                    # Video files (*.mp4)
│   └── keypoints_wlasl/        # Extracted keypoints (*.npy)
└── checkpoints/
    ├── wlasl_best.keras        # Trained model
    └── wlasl_labels.txt        # Word labels
```

## 🎯 Roadmap

- [x] Beautiful web interface
- [x] Real-time recognition
- [x] Text-to-speech output
- [x] Sentence builder
- [x] Volume controls
- [ ] Train on 100 words
- [ ] Train on 300 words
- [ ] Add user authentication
- [ ] Save/load sentences
- [ ] Export to text file
- [ ] Multi-language support
- [ ] Mobile app version

## 🙏 Credits

- **WLASL Dataset**: [WLASL](https://dxli94.github.io/WLASL/)
- **MediaPipe**: Google MediaPipe for hand/pose tracking
- **Flask**: Web framework
- **pyttsx3**: Text-to-speech library
- **TensorFlow/Keras**: Deep learning framework

## 📄 License

This project is for educational purposes.

---

**Enjoy your stunning ISL Recognition web app!** 🎉

For questions or issues, check the troubleshooting section above.
