# ISL Recognition - Sign Language to Speech

Real-time Indian Sign Language (ISL) recognition system with speech output using deep learning and MediaPipe.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- 🎯 **Real-time Recognition**: Recognizes 35 ISL gestures (A-Z, 1-9)
- 🎤 **Speech Output**: Text-to-speech for recognized signs
- 🌐 **Web Interface**: Beautiful, responsive UI with live video
- 📊 **High Accuracy**: 85-90% validation accuracy
- 🚀 **Fast**: Real-time inference at 10+ FPS
- 💻 **Cross-platform**: Works on macOS, Linux, Windows

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Webcam
- 4GB RAM minimum

### Installation

```bash
# Clone repository
git clone https://github.com/mohammadshafishaik/SIGN-LANGUAGE-TO-SPEECH-AND-TEXT.git
cd isl-recognition

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
# Simple method
./run_app.sh

# Or manually
source venv/bin/activate
export PYTHONPATH=.
python inference/webapp_simple.py
```

Open your browser to: **http://localhost:8080**

## 📸 Screenshots

### Web Interface
- Real-time video feed with hand tracking
- Top 5 predictions with confidence scores
- Speech controls and sentence builder
- Responsive design for all devices

## 🎯 Recognized Gestures

**Letters**: A-Z (26 letters)  
**Numbers**: 1-9 (9 numbers)  
**Total**: 35 gestures

## 🏗️ Architecture

### Model
- **Input**: 144D features (hand + pose landmarks)
- **Architecture**: Dense layers with BatchNorm + Dropout
- **Framework**: TensorFlow/Keras
- **Size**: ~1MB

### Pipeline
1. **Capture**: Webcam video at 30 FPS
2. **Extract**: MediaPipe hand and pose landmarks
3. **Predict**: Deep learning model inference
4. **Speak**: Text-to-speech output

## 📁 Project Structure

```
isl-recognition/
├── inference/              # Web app and inference
│   ├── webapp_simple.py   # Main application ⭐
│   ├── app.py             # Alternative app
│   └── templates/         # HTML templates
├── models/                # Training scripts
│   ├── train_isl.py      # Train ISL model
│   └── train_wlasl.py    # Train WLASL model
├── data_collector/        # Data collection tools
├── checkpoints/           # Model files (download separately)
├── requirements.txt       # Dependencies
├── run_app.sh            # Quick start script
└── README.md             # This file
```

## 📥 Download Model

The trained model is not included in the repository (too large for Git).

**Option 1: Use Pre-trained Model**

Download from releases:
```bash
# Download model files
wget https://github.com/YOUR_USERNAME/isl-recognition/releases/download/v1.0/isl_best.keras
wget https://github.com/YOUR_USERNAME/isl-recognition/releases/download/v1.0/labels.txt

# Move to checkpoints
mv isl_best.keras checkpoints/
mv labels.txt checkpoints/
```

**Option 2: Train Your Own**

```bash
# Collect data
python data_collector/collect.py --phrase "A"

# Train model
python models/train_isl.py
```

## 🔧 Configuration

Edit `inference/webapp_simple.py` to customize:

```python
# Model paths
MODEL_PATH = 'checkpoints/isl_best.keras'
LABELS_PATH = 'checkpoints/labels.txt'

# Speech settings
SPEECH_ENABLED = True
VOLUME = 0.8
AUTO_SPEAK = False

# Inference settings
CONFIDENCE_THRESHOLD = 0.5
TOP_K_PREDICTIONS = 5
```

## 🎓 Training

### Collect Data

```bash
python data_collector/collect.py --phrase "hello" --duration 3
```

### Train Model

```bash
python models/train_isl.py
```

Training takes 30-60 minutes on CPU, 5-10 minutes on GPU.

## 🌐 Expanding to 100+ Words

Want more vocabulary? Integrate WLASL dataset:

```bash
# Upgrade TensorFlow
pip install --upgrade tensorflow-macos==2.16.1

# Integrate 100-word model
python integrate_kaggle_model.py \
  --model path/to/wlasl_100_best.keras \
  --labels path/to/labels_wlasl_100.txt
```

See `COMPLETE_GUIDE.md` for details.

## 📊 Performance

| Model | Classes | Accuracy | Size | Speed |
|-------|---------|----------|------|-------|
| ISL-35 | 35 | 85-90% | 1MB | 10+ FPS |
| WLASL-30 | 30 | 90-95% | 12MB | 10+ FPS |
| WLASL-100 | 100 | 85-90% | 70MB | 8+ FPS |

## 🐛 Troubleshooting

### Webcam not detected
```bash
# Test webcam
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### Port already in use
```bash
# Kill process on port 8080
lsof -ti:8080 | xargs kill -9
```

### Speech not working (macOS)
```bash
pip uninstall pyttsx3
pip install pyttsx3 pyobjc
```

### Model not found
Download model files to `checkpoints/` directory.

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **MediaPipe** by Google for hand tracking
- **WLASL Dataset** for sign language videos
- **TensorFlow** for deep learning framework

## 📞 Support

- 📧 Email: sk.mohammadshafi3044@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/mohammadshafishaik/SIGN-LANGUAGE-TO-SPEECH-AND-TEXT)
- 📖 Docs: See `COMPLETE_GUIDE.md`

## 🎯 Roadmap

- [ ] Mobile app (iOS/Android)
- [ ] More languages (ASL, BSL)
- [ ] Continuous sign recognition
- [ ] Sentence grammar correction
- [ ] Cloud deployment

---

**Made with ❤️ for the deaf and hard-of-hearing community**

⭐ Star this repo if you find it helpful!
