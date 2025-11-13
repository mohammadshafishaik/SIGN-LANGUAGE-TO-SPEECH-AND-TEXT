#!/usr/bin/env python3
"""
Convert Keras 3.x model to Keras 2.x compatible format
"""

import sys
import os

# Temporarily set environment to use Keras 3 compatibility
os.environ['TF_USE_LEGACY_KERAS'] = '0'

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    # Try to load with compatibility mode
    model_path = sys.argv[1] if len(sys.argv) > 1 else "~/Downloads/wlasl_30_best.keras"
    model_path = os.path.expanduser(model_path)
    
    print(f"\nAttempting to load: {model_path}")
    
    # Try different loading methods
    try:
        # Method 1: Direct load
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✓ Model loaded successfully!")
        
        # Save in compatible format
        output_path = model_path.replace('.keras', '_converted.h5')
        model.save(output_path, save_format='h5')
        print(f"✓ Converted model saved to: {output_path}")
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\nThe model was trained with Keras 3.x (from Kaggle/Colab)")
        print("Your local environment has Keras 2.15")
        print("\nSOLUTION: Use the model directly in Colab or upgrade to TensorFlow 2.16+")
        
except ImportError as e:
    print(f"Import error: {e}")
