#!/usr/bin/env python3
"""
🚀 KAGGLE MODEL INTEGRATION SCRIPT

Automatically integrates your Kaggle-trained model with the ISL web app.

Usage:
    python integrate_kaggle_model.py
    
Or with custom paths:
    python integrate_kaggle_model.py --model path/to/model.keras --labels path/to/labels.txt
"""

import os
import sys
import argparse
from pathlib import Path
import tensorflow as tf
import numpy as np

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def find_kaggle_model():
    """Auto-detect downloaded Kaggle model"""
    print_info("Searching for Kaggle model files...")
    
    search_paths = [
        Path.cwd() / 'kaggle_output',
        Path.cwd() / 'checkpoints',
        Path.home() / 'Downloads',
        Path.cwd(),
    ]
    
    model_files = []
    label_files = []
    
    for search_path in search_paths:
        if not search_path.exists():
            continue
            
        # Find model files
        for pattern in ['*.keras', '*.h5']:
            model_files.extend(search_path.glob(pattern))
        
        # Find label files
        for pattern in ['*label*.txt', 'labels.txt', 'wlasl_labels.txt']:
            label_files.extend(search_path.glob(pattern))
    
    # Filter for WLASL models
    wlasl_models = [f for f in model_files if 'wlasl' in f.name.lower()]
    wlasl_labels = [f for f in label_files if 'wlasl' in f.name.lower() or 'label' in f.name.lower()]
    
    if wlasl_models:
        print_success(f"Found {len(wlasl_models)} model file(s)")
        for m in wlasl_models:
            print(f"  - {m}")
    
    if wlasl_labels:
        print_success(f"Found {len(wlasl_labels)} label file(s)")
        for l in wlasl_labels:
            print(f"  - {l}")
    
    return wlasl_models, wlasl_labels

def verify_model(model_path):
    """Verify model can be loaded and check architecture"""
    print_info(f"Verifying model: {model_path.name}")
    
    try:
        model = tf.keras.models.load_model(str(model_path))
        print_success("Model loaded successfully!")
        
        # Check input shape
        input_shape = model.input_shape
        output_shape = model.output_shape
        
        print(f"  Input shape:  {input_shape}")
        print(f"  Output shape: {output_shape}")
        print(f"  Parameters:   {model.count_params():,}")
        
        # Expected shapes for WLASL model
        expected_frames = 30
        expected_features = 312  # 104 landmarks × 3 coords
        
        if len(input_shape) == 3:
            _, frames, features = input_shape
            
            if frames == expected_frames and features == expected_features:
                print_success(f"✓ Model architecture matches! ({frames} frames, {features} features)")
                return model, True
            else:
                print_warning(f"Model shape differs: {frames} frames, {features} features")
                print_warning(f"Expected: {expected_frames} frames, {expected_features} features")
                return model, False
        else:
            print_warning(f"Unexpected input shape: {input_shape}")
            return model, False
            
    except Exception as e:
        print_error(f"Failed to load model: {e}")
        return None, False

def load_labels(labels_path):
    """Load and verify labels"""
    print_info(f"Loading labels: {labels_path.name}")
    
    try:
        with open(labels_path, 'r') as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
        
        print_success(f"Loaded {len(labels)} labels")
        
        # Show first 10 labels
        print("  First 10 words:")
        for i, label in enumerate(labels[:10], 1):
            print(f"    {i:2d}. {label}")
        
        if len(labels) > 10:
            print(f"    ... ({len(labels) - 10} more)")
        
        return labels
        
    except Exception as e:
        print_error(f"Failed to load labels: {e}")
        return None

def test_model(model, labels):
    """Test model with dummy data"""
    print_info("Testing model with sample data...")
    
    try:
        # Create dummy input
        input_shape = model.input_shape
        batch_size = 1
        
        if len(input_shape) == 3:
            _, frames, features = input_shape
            dummy_input = np.random.randn(batch_size, frames, features).astype(np.float32)
        else:
            print_error("Unexpected input shape for testing")
            return False
        
        # Run prediction
        predictions = model.predict(dummy_input, verbose=0)
        
        # Get top 5 predictions
        top_5_idx = np.argsort(predictions[0])[-5:][::-1]
        
        print_success("Model inference successful!")
        print("  Top 5 predictions (random input):")
        for i, idx in enumerate(top_5_idx, 1):
            confidence = predictions[0][idx] * 100
            word = labels[idx] if idx < len(labels) else f"class_{idx}"
            print(f"    {i}. {word:20s} ({confidence:5.2f}%)")
        
        return True
        
    except Exception as e:
        print_error(f"Model test failed: {e}")
        return False

def integrate_with_webapp(model_path, labels_path, checkpoints_dir):
    """Copy files to checkpoints and update web app"""
    print_info("Integrating with web app...")
    
    # Create checkpoints directory if needed
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model
    target_model = checkpoints_dir / 'wlasl_model.keras'
    import shutil
    shutil.copy2(model_path, target_model)
    print_success(f"Model copied to: {target_model}")
    
    # Copy labels
    target_labels = checkpoints_dir / 'wlasl_labels.txt'
    shutil.copy2(labels_path, target_labels)
    print_success(f"Labels copied to: {target_labels}")
    
    # Also keep backup of original ISL model
    isl_model = checkpoints_dir / 'isl_best.keras'
    if isl_model.exists():
        backup = checkpoints_dir / 'isl_best_backup.keras'
        if not backup.exists():
            shutil.copy2(isl_model, backup)
            print_info(f"Original ISL model backed up to: {backup.name}")
    
    return target_model, target_labels

def create_webapp_config(model_path, labels_path, num_classes):
    """Create configuration file for web app"""
    config_path = Path('webapp_config.py')
    
    config_content = f'''"""
Web App Configuration - Auto-generated by integrate_kaggle_model.py
"""

from pathlib import Path

# Model paths
MODEL_PATH = Path('{model_path}')
LABELS_PATH = Path('{labels_path}')

# Model info
NUM_CLASSES = {num_classes}
INPUT_FRAMES = 30
INPUT_FEATURES = 312  # 104 landmarks × 3 coords

# Feature extraction
USE_HANDS = True
USE_POSE = True
USE_FACE = False

# Inference settings
CONFIDENCE_THRESHOLD = 0.5
TOP_K_PREDICTIONS = 5
SMOOTHING_WINDOW = 5

# Speech settings
SPEECH_ENABLED = True
AUTO_SPEAK_THRESHOLD = 0.75
SPEECH_COOLDOWN = 2.0
DEFAULT_VOLUME = 0.8

print(f"✓ Loaded config: {{NUM_CLASSES}} classes")
'''
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print_success(f"Configuration saved to: {config_path}")
    return config_path

def main():
    parser = argparse.ArgumentParser(description='Integrate Kaggle-trained model with ISL web app')
    parser.add_argument('--model', type=str, help='Path to model file (.keras or .h5)')
    parser.add_argument('--labels', type=str, help='Path to labels file (.txt)')
    parser.add_argument('--checkpoints', type=str, default='checkpoints', help='Checkpoints directory')
    parser.add_argument('--skip-test', action='store_true', help='Skip model testing')
    
    args = parser.parse_args()
    
    print_header("KAGGLE MODEL INTEGRATION")
    
    # Step 1: Find model files
    if args.model and args.labels:
        model_path = Path(args.model)
        labels_path = Path(args.labels)
        
        if not model_path.exists():
            print_error(f"Model file not found: {model_path}")
            sys.exit(1)
        if not labels_path.exists():
            print_error(f"Labels file not found: {labels_path}")
            sys.exit(1)
            
        print_success("Using provided model and labels")
    else:
        print_info("Auto-detecting Kaggle model files...")
        model_files, label_files = find_kaggle_model()
        
        if not model_files:
            print_error("No model files found!")
            print("\nPlease download your model from Kaggle first:")
            print("  1. Go to your Kaggle notebook")
            print("  2. Download the output files")
            print("  3. Place them in ./kaggle_output/ or ./checkpoints/")
            print("\nOr specify paths manually:")
            print("  python integrate_kaggle_model.py --model path/to/model.keras --labels path/to/labels.txt")
            sys.exit(1)
        
        if not label_files:
            print_error("No label files found!")
            sys.exit(1)
        
        # Use the first found files
        model_path = model_files[0]
        labels_path = label_files[0]
        
        print_success(f"Selected model: {model_path.name}")
        print_success(f"Selected labels: {labels_path.name}")
    
    # Step 2: Verify model
    print_header("VERIFYING MODEL")
    model, is_compatible = verify_model(model_path)
    
    if model is None:
        print_error("Model verification failed!")
        sys.exit(1)
    
    if not is_compatible:
        print_warning("Model architecture differs from expected format")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    # Step 3: Load labels
    print_header("LOADING LABELS")
    labels = load_labels(labels_path)
    
    if labels is None:
        print_error("Failed to load labels!")
        sys.exit(1)
    
    num_classes = len(labels)
    
    # Step 4: Test model
    if not args.skip_test:
        print_header("TESTING MODEL")
        if not test_model(model, labels):
            print_error("Model test failed!")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(0)
    
    # Step 5: Integrate with web app
    print_header("INTEGRATING WITH WEB APP")
    checkpoints_dir = Path(args.checkpoints)
    target_model, target_labels = integrate_with_webapp(model_path, labels_path, checkpoints_dir)
    
    # Step 6: Create config
    config_path = create_webapp_config(target_model, target_labels, num_classes)
    
    # Success!
    print_header("INTEGRATION COMPLETE!")
    
    print_success("Your Kaggle model is now integrated!")
    print()
    print(f"📊 Model Info:")
    print(f"   Classes: {num_classes}")
    print(f"   Model: {target_model.name}")
    print(f"   Labels: {target_labels.name}")
    print()
    print(f"🚀 Next Steps:")
    print(f"   1. Start the web app:")
    print(f"      python inference/app.py")
    print()
    print(f"   2. Open in browser:")
    print(f"      http://localhost:8080")
    print()
    print(f"   3. Test with your webcam!")
    print()
    print(f"💡 Tips:")
    print(f"   - Make sure lighting is good")
    print(f"   - Keep hands clearly visible")
    print(f"   - Sign slowly and clearly")
    print(f"   - Check the word list in {target_labels.name}")
    print()
    
    # Ask if user wants to start the app
    response = input("Start the web app now? (y/n): ")
    if response.lower() == 'y':
        print()
        print_info("Starting web app...")
        os.system("python inference/app.py")

if __name__ == '__main__':
    main()
