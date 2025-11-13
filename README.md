# Real-Time Sign Language to Speech Translator

This project translates sign language into speech in real time using 3D pose estimation and deep learning.

## Abstract
This project presents an end-to-end system for translating continuous sign language into spoken language. It uses MediaPipe Holistic to extract 3D landmarks for the full body, hands, and face from a webcam feed. By computing temporal features like velocity and acceleration, the system can better understand the dynamics of signing. The core of the project includes models for both isolated and continuous sign recognition, a real-time segmentation module to detect the start and end of signs, and a text-to-speech engine for voice output. The system is designed for edge deployment and includes a thorough evaluation of feature importance and ethical considerations.

## Features
- **Real-Time 3D Landmark Extraction:** Full body, hands, and face keypoints using MediaPipe Holistic.
- **Temporal Feature Computation:** Velocity and acceleration of joints for dynamic sign representation.
- **Sign Segmentation:** Automatic detection of the start and end of sign phrases in a continuous stream.
- **Dual-Model Approach:**
    - Baseline LSTM for isolated sign classification.
    - Advanced Transformer/CTC models for continuous sequence-to-sequence translation.
- **Text-to-Speech (TTS):** Real-time audio feedback in multiple languages (e.g., English, Telugu).
- **Edge Deployment Ready:** Model conversion scripts for TFLite (Raspberry Pi) and Core ML (iPhone).
- **Ablation Studies:** Jupyter notebook to compare 2D, 3D, and 3D+velocity features.
- **Ethical Evaluation:** A dedicated report on dataset consent, bias, and mitigation strategies.

## Setup
1.  **Install Python 3.11:** This project is optimized for Python 3.11, especially for TensorFlow on Apple Silicon.
    ```bash
    brew install python@3.11
    ```
2.  **Run the setup script:** This will create a virtual environment and install all necessary dependencies.
    ```bash
    chmod +x env_setup.sh
    ./env_setup.sh
    ```
3.  **Activate the virtual environment:**
    ```bash
    source venv/bin/activate
    ```

## Run Commands

### 1. Data Collection
Record your own samples for each phrase. A window will appear for recording.
```bash
# Usage: python data_collector/collect.py --phrase "your phrase"
python data_collector/collect.py --phrase "hello"
python data_collector/collect.py --phrase "thank you"
```

### 2. Keypoint Extraction
Process the recorded videos to extract and save keypoints.
```bash
python pose_extractor/mediapipe_extractor.py --video_dir dataset/raw --output_dir dataset/keypoints
```

### 3. Train the Model
Train the baseline LSTM model on the extracted keypoints.
```bash
python models/baseline_lstm.py --train
```

### 4. Run Real-Time Inference
Launch the application to see live translation from your webcam.
```bash
python inference/realtime_inference.py
```

## Ablation Table
The following table will be populated by running the `notebooks/ablation.ipynb` notebook. It shows the performance improvement as we add more complex features.

| Feature Type        | Accuracy | F1-Score (Weighted) |
|---------------------|----------|---------------------|
| 2D                  | *TBD*    | *TBD*               |
| 3D                  | *TBD*    | *TBD*               |
| 3D + Velocity       | *TBD*    | *TBD*               |
| 3D + Velo + Accel   | *TBD*    | *TBD*               |


## Ethics
...

## Future Work
...
