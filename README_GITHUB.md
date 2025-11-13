# 🤟 ISL Recognition - Sign Language to Speech

Real-time Indian Sign Language (ISL) recognition system with speech output using deep learning and computer vision.

## 🎯 Features

- **Real-time Recognition**: Recognizes 35 ISL gestures (A-Z, 1-9) in real-time
- **Speech Output**: Text-to-speech conversion for recognized signs
- **Beautiful Web UI**: Modern, responsive web interface with live video feed
- **High Accuracy**: Deep learning model with 85-90% accuracy
- **MediaPipe Integration**: Advanced hand and pose tracking
- **Sentence Builder**: Build and speak complete sentences

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Webcam
- macOS (M1/M2/M4) or Linux

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/isl-recognition.git
cd isl-recognition

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

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

## 📊 Model Architecture

- **Input**: 144D features (hand + pose landmarks from MediaPipe)
- **Architecture**: Dense layers with BatchNorm + Dropout
- **Output**: 35 classes (A-Z, 1-9)
- **Framework**: TensorFlow/Keras

## 🎨 Web Interface

- **Real-time video feed** with skeleton overlay
- **Top 5 predictions** with confidence scores
- **Speech controls** (volume, auto-speak, manual speak)
- **Sentence builder** with keyboard shortcuts
- **Responsive design** for all screen sizes

## 📁 Project Structure

```
isl-recognition/
├── inference/              # Web app and inference scripts
│   ├── webapp_simple.py   # Main web application
│   ├── app.py             # Alternative web app
│   └── templates/         # HTML templates
├── models/                # Model training scripts
│   ├── train_isl.py      # ISL model training
│   └── train_wlasl.py    # WLASL model training
├── data_collector/        # Data collection tools
├── checkpoints/           # Trained models (not in git)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Training Your Own Model

### Collect Data

```bash
python data_collector/collect.py --phrase "hello"
```

### Train Model

```bash
python models/train_isl.py
```

## 🌐 Expanding to 100+ Words

To use the WLASL dataset with 100+ words:

```bash
# Upgrade TensorFlow
pip install --upgrade tensorflow-macos==2.16.1

# Integrate Kaggle model
python integrate_kaggle_model.py \
  --model path/to/wlasl_100_best.keras \
  --labels path/to/labels_wlasl_100.txt
```

See `COMPLETE_GUIDE.md` for detailed instructions.

## 📋 Requirements

- Python 3.11+
- TensorFlow 2.15+
- OpenCV
- MediaPipe
- Flask
- pyttsx3 (text-to-speech)

## 🎯 Accuracy

- **ISL (35 classes)**: 85-90% validation accuracy
- **WLASL-30**: 90-95% validation accuracy
- **WLASL-100**: 85-90% validation accuracy

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MediaPipe** by Google for hand and pose tracking
- **WLASL Dataset** for sign language videos
- **TensorFlow/Keras** for deep learning framework

## 📞 Contact

For questions or issues, please open an issue on GitHub.

---

**Made with ❤️ for the deaf and hard-of-hearing community**
