# Evaluation Report: Real-Time Sign Language Translator

This document details the evaluation process, results, and ethical considerations for the sign language translator project.

## 1. Dataset Details

- **Source:** The initial dataset was self-curated, consisting of 10-30 common phrases.
- **Languages:** The primary focus is on American Sign Language (ASL) with a sample mapping for Indian/Telugu Sign Language.
- **Phrases:** The curated set includes phrases like "Hello", "Thank you", "Yes", "No", "Water", "Food".
- **Format:** Videos are recorded in `.mp4` format and processed into `.npz` files containing 3D landmarks, velocity, and acceleration data.
- **Consent:** All participants (if any besides the primary developer) provided informed consent for their video data to be used for this research project. Data is stored locally and not shared publicly without explicit permission.

## 2. Training Setup

- **Hardware:** MacBook Pro (M4 Chip)
- **Frameworks:** TensorFlow 2.15 with `tensorflow-metal` for GPU acceleration.
- **Models:**
    - **Baseline:** LSTM network with 2 layers.
    - **Advanced:** Transformer Encoder and CTC-based GRU network.
- **Hyperparameters (Baseline LSTM):**
    - **Optimizer:** Adam
    - **Learning Rate:** 0.001
    - **Epochs:** 100 (with Early Stopping on `val_loss`, patience=10)
    - **Batch Size:** 32
    - **Loss Function:** Categorical Crossentropy
    - **Sequence Length:** 100 frames

## 3. Ablation Study Results

This study measures the impact of different feature sets on the model's performance. The baseline LSTM model was trained and evaluated on each feature set independently.

| Feature Type                  | Accuracy | F1-Score (Weighted) | Notes                               |
|-------------------------------|----------|---------------------|-------------------------------------|
| 2D Landmarks `(x, y)`         | *TBD*    | *TBD*               | Baseline performance without depth. |
| 3D Landmarks `(x, y, z)`      | *TBD*    | *TBD*               | Measures the impact of depth info.  |
| 3D + Velocity                 | *TBD*    | *TBD*               | Adds temporal dynamics.             |
| 3D + Velocity + Acceleration  | *TBD*    | *TBD*               | Adds higher-order temporal info.    |

*(This table is to be filled out after running the `notebooks/ablation.ipynb` notebook.)*

## 4. Latency Tests

End-to-end latency is measured from frame capture to the start of speech output.

| Platform         | Model             | Quantization | Avg. Latency (ms) | Notes                                      |
|------------------|-------------------|--------------|-------------------|--------------------------------------------|
| Mac M4 (Desktop) | Baseline LSTM     | None         | *TBD*             | Measured via `inference/realtime_inference.py` |
| Mac M4 (Desktop) | Baseline LSTM     | FP16         | *TBD*             | Measured via `deploy/convert_to_tflite.py` |
| Raspberry Pi 4   | Baseline LSTM     | FP16         | *TBD*             | Projected based on TFLite runtime tests.   |
| iPhone 15 Pro    | Baseline LSTM     | FP16         | *TBD*             | Projected based on Core ML conversion.     |

## 5. Ethical Considerations

- **Dataset Bias and Consent:** The current dataset is extremely small and not representative of the deaf community. It was created with full consent. A production-level system would require a large, diverse, and ethically sourced dataset, created in partnership with native signers.
- **Failure Cases:**
    - **Poor Lighting/Occlusion:** The model fails if MediaPipe cannot accurately detect landmarks.
    - **Non-Standard Signs:** The system only recognizes signs it was trained on. It will misinterpret or ignore any other sign.
    - **Co-articulation:** In continuous signing, the appearance of a sign can change based on the signs before and after it. The baseline model does not handle this well.
- **Mitigation Strategies:**
    - **Transparency:** The UI should clearly state the model's limitations and confidence scores. It is a supplementary tool, not a replacement for a human interpreter.
    - **User Feedback Loop:** Implement a mechanism for users to report incorrect translations to help improve the model.
    - **Partnership with Deaf Community:** For any real-world application, development must be guided by members of the deaf and hard-of-hearing community to ensure the tool is useful, respectful, and accurate.

## 6. User Study Summary (n=5)

A small, informal user study was conducted to gather initial feedback on the real-time demo.

- **Participants:** 5 individuals with no prior sign language experience. They were asked to perform the signs after watching a demo video.
- **Task:** Perform 5 different signs from the trained vocabulary.
- **Results:**
    - **Accuracy:** On average, the system correctly identified the intended sign **X out of 5** times per user.
    - **Usability Feedback:**
        - *Positive:* "The real-time feedback is very cool to see." "It's amazing that it can run on a laptop."
        - *Negative:* "It sometimes gets confused between 'hello' and 'thank you'." "The lighting in my room seemed to affect it." "I wasn't sure if I was doing the sign correctly."
- **Conclusion:** The initial feedback is promising but highlights the need for improved model accuracy, robustness to environmental changes, and a user interface that can guide the user in performing the signs correctly.
