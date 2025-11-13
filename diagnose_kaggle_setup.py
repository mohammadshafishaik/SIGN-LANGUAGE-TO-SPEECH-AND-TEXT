#!/usr/bin/env python3
"""
🔍 KAGGLE SETUP DIAGNOSTIC TOOL

Checks your Kaggle API setup and model files.
"""

import os
import sys
from pathlib import Path
import subprocess

def check_mark(condition):
    return "✅" if condition else "❌"

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check_kaggle_cli():
    """Check if Kaggle CLI is installed"""
    print_section("KAGGLE CLI")
    
    try:
        result = subprocess.run(['kaggle', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Kaggle CLI installed: {version}")
            return True
        else:
            print(f"❌ Kaggle CLI not working properly")
            return False
    except FileNotFoundError:
        print(f"❌ Kaggle CLI not installed")
        print(f"\n   Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"❌ Error checking Kaggle CLI: {e}")
        return False

def check_kaggle_credentials():
    """Check if Kaggle API credentials are configured"""
    print_section("KAGGLE CREDENTIALS")
    
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    
    if kaggle_json.exists():
        print(f"✅ Credentials file exists: {kaggle_json}")
        
        # Check permissions
        stat = kaggle_json.stat()
        mode = oct(stat.st_mode)[-3:]
        
        if mode == '600':
            print(f"✅ Permissions correct: {mode}")
        else:
            print(f"⚠️  Permissions should be 600, currently: {mode}")
            print(f"   Fix with: chmod 600 {kaggle_json}")
        
        # Try to list datasets
        try:
            result = subprocess.run(['kaggle', 'datasets', 'list', '--mine'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ API authentication successful")
                return True
            else:
                print(f"❌ API authentication failed")
                print(f"   Error: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Could not test API: {e}")
            return False
    else:
        print(f"❌ Credentials file not found: {kaggle_json}")
        print(f"\n   Setup instructions:")
        print(f"   1. Go to https://www.kaggle.com/account")
        print(f"   2. Click 'Create New API Token'")
        print(f"   3. Move kaggle.json to ~/.kaggle/")
        print(f"   4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False

def check_tensorflow():
    """Check TensorFlow installation"""
    print_section("TENSORFLOW")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow installed: {tf.__version__}")
        
        # Check GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU available: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"   - {gpu}")
        else:
            print(f"ℹ️  No GPU (CPU only)")
        
        return True
    except ImportError:
        print(f"❌ TensorFlow not installed")
        print(f"\n   Install with: pip install tensorflow")
        return False

def check_model_files():
    """Check for model files"""
    print_section("MODEL FILES")
    
    search_paths = [
        Path.cwd() / 'kaggle_output',
        Path.cwd() / 'checkpoints',
        Path.home() / 'Downloads',
    ]
    
    found_models = []
    found_labels = []
    
    for search_path in search_paths:
        if not search_path.exists():
            continue
        
        # Find models
        for pattern in ['*.keras', '*.h5']:
            found_models.extend(search_path.glob(pattern))
        
        # Find labels
        for pattern in ['*label*.txt', 'labels.txt']:
            found_labels.extend(search_path.glob(pattern))
    
    if found_models:
        print(f"✅ Found {len(found_models)} model file(s):")
        for model in found_models:
            size_mb = model.stat().st_size / (1024 * 1024)
            print(f"   - {model.name} ({size_mb:.1f} MB)")
    else:
        print(f"❌ No model files found")
        print(f"\n   Searched in:")
        for path in search_paths:
            print(f"   - {path}")
    
    if found_labels:
        print(f"✅ Found {len(found_labels)} label file(s):")
        for labels in found_labels:
            print(f"   - {labels.name}")
    else:
        print(f"❌ No label files found")
    
    return len(found_models) > 0 and len(found_labels) > 0

def check_webapp():
    """Check web app files"""
    print_section("WEB APP")
    
    app_file = Path('inference/app.py')
    checkpoints = Path('checkpoints')
    
    if app_file.exists():
        print(f"✅ Web app found: {app_file}")
    else:
        print(f"❌ Web app not found: {app_file}")
    
    if checkpoints.exists():
        print(f"✅ Checkpoints directory exists")
        
        # List existing models
        models = list(checkpoints.glob('*.keras')) + list(checkpoints.glob('*.h5'))
        if models:
            print(f"   Existing models:")
            for model in models:
                print(f"   - {model.name}")
    else:
        print(f"⚠️  Checkpoints directory not found")
    
    return app_file.exists()

def list_kaggle_resources():
    """List user's Kaggle notebooks and datasets"""
    print_section("YOUR KAGGLE RESOURCES")
    
    try:
        # List notebooks
        print("\n📓 Your Notebooks:")
        result = subprocess.run(['kaggle', 'kernels', 'list', '--mine', '--page-size', '10'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("Could not list notebooks")
        
        # List datasets
        print("\n📊 Your Datasets:")
        result = subprocess.run(['kaggle', 'datasets', 'list', '--mine', '--page-size', '10'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("Could not list datasets")
            
    except Exception as e:
        print(f"Could not list resources: {e}")

def main():
    print("🔍 KAGGLE SETUP DIAGNOSTIC")
    print("="*60)
    
    checks = {
        'Kaggle CLI': check_kaggle_cli(),
        'Kaggle Credentials': check_kaggle_credentials(),
        'TensorFlow': check_tensorflow(),
        'Model Files': check_model_files(),
        'Web App': check_webapp(),
    }
    
    # List Kaggle resources if credentials work
    if checks['Kaggle Credentials']:
        list_kaggle_resources()
    
    # Summary
    print_section("SUMMARY")
    
    for check_name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10s} {check_name}")
    
    all_passed = all(checks.values())
    
    print()
    if all_passed:
        print("🎉 All checks passed! You're ready to download and integrate your model.")
        print()
        print("Next steps:")
        print("  1. Download your model from Kaggle")
        print("  2. Run: python integrate_kaggle_model.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print()
        print("Quick fixes:")
        if not checks['Kaggle CLI']:
            print("  - Install Kaggle: pip install kaggle")
        if not checks['Kaggle Credentials']:
            print("  - Setup credentials: https://www.kaggle.com/account")
        if not checks['TensorFlow']:
            print("  - Install TensorFlow: pip install tensorflow")
        if not checks['Model Files']:
            print("  - Download your model from Kaggle")
    
    print()

if __name__ == '__main__':
    main()
