"""
Test Number Recognition - Shows actual predictions for sample images
This helps understand what the model learned for numbers vs letters
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import random

# Load model and labels
print("\n" + "="*70)
print("ISL NUMBER RECOGNITION TEST")
print("="*70)

model_path = Path("checkpoints/isl_best.keras")
model = tf.keras.models.load_model(model_path)
print(f"✓ Model loaded")

label_path = Path("dataset/splits_isl/label_mappings.json")
with open(label_path, 'r') as f:
    label_info = json.load(f)
idx_to_label = {int(k): v for k, v in label_info['idx_to_label'].items()}
print(f"✓ Labels loaded: {len(idx_to_label)} classes\n")

# Test each number with real samples from dataset
dataset_dir = Path("dataset/keypoints_isl")

print("="*70)
print("TESTING NUMBERS WITH ACTUAL DATASET SAMPLES:")
print("="*70)
print("(Showing what the model predicts for its own training data)\n")

for num in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
    npz_path = dataset_dir / f"{num}.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        features = data['features']
        
        # Test 5 random samples from this number class
        sample_indices = random.sample(range(len(features)), min(5, len(features)))
        
        predictions_summary = []
        for idx in sample_indices:
            sample = features[idx].reshape(1, -1)
            pred = model.predict(sample, verbose=0)[0]
            top_3_idx = np.argsort(pred)[-3:][::-1]
            top_3 = [(idx_to_label[i], pred[i]) for i in top_3_idx]
            predictions_summary.append(top_3)
        
        # Analyze: How often does the model predict the correct number?
        correct_count = 0
        letter_wins = []
        for preds in predictions_summary:
            top_pred = preds[0][0]
            if top_pred == num:
                correct_count += 1
            else:
                letter_wins.append(top_pred)
        
        accuracy = correct_count / len(predictions_summary) * 100
        
        print(f"\nNumber '{num}':")
        print(f"  Accuracy on own dataset: {accuracy:.0f}% ({correct_count}/{len(predictions_summary)} samples)")
        
        if letter_wins:
            from collections import Counter
            most_common = Counter(letter_wins).most_common(3)
            print(f"  Often confused with: {', '.join([f'{label} ({count}x)' for label, count in most_common])}")
        
        # Show one example prediction
        example = predictions_summary[0]
        print(f"  Example prediction:")
        for i, (label, conf) in enumerate(example):
            marker = "✓" if label == num else "✗"
            num_or_let = "[NUM]" if label in ['1','2','3','4','5','6','7','8','9'] else "[LTR]"
            print(f"    {i+1}. {marker} {num_or_let} {label}: {conf*100:.1f}%")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)

print("""
If numbers show LOW accuracy on their own training data, it means:
1. The ISL dataset has very similar poses for numbers and letters
2. The model learned that these poses are ambiguous
3. Real-world webcam detection will struggle too

SOLUTION:
We need to either:
A) Accept that ISL numbers look like letters (this is how ISL works!)
B) Use a different dataset with more distinct number signs
C) Add temporal information (sequence/motion) to distinguish them
""")

print("\nRun this script to see the confusion patterns!")
print("="*70 + "\n")
