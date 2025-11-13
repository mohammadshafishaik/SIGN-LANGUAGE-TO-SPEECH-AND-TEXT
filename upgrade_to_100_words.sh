#!/bin/bash

# 🚀 Upgrade ISL Recognition to 100 Words

echo "=================================="
echo "  UPGRADE TO 100 WORDS"
echo "=================================="
echo ""

# Stop any running app
echo "🛑 Stopping any running app..."
lsof -ti:8080 | xargs kill -9 2>/dev/null
sleep 2

# Activate venv
source venv/bin/activate

echo "✅ Virtual environment activated"
echo ""

# Step 1: Upgrade TensorFlow
echo "=================================="
echo "  STEP 1: UPGRADE TENSORFLOW"
echo "=================================="
echo ""

CURRENT_TF=$(python -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null)
echo "Current TensorFlow: $CURRENT_TF"

if [[ "$CURRENT_TF" < "2.16" ]]; then
    echo ""
    echo "⚠️  TensorFlow needs upgrade for Keras 3 compatibility"
    echo "This will take 2-3 minutes..."
    echo ""
    read -p "Upgrade TensorFlow to 2.16.1? (y/n): " upgrade
    
    if [ "$upgrade" = "y" ]; then
        echo "📦 Upgrading TensorFlow..."
        pip install --upgrade tensorflow-macos==2.16.1
        
        if [ $? -eq 0 ]; then
            echo "✅ TensorFlow upgraded successfully!"
        else
            echo "❌ Upgrade failed!"
            exit 1
        fi
    else
        echo "⚠️  Skipping upgrade - model may not load"
    fi
else
    echo "✅ TensorFlow version is compatible"
fi

echo ""

# Step 2: Find Kaggle model
echo "=================================="
echo "  STEP 2: FIND KAGGLE MODEL"
echo "=================================="
echo ""

echo "Looking for WLASL models in Downloads..."
MODELS=$(find ~/Downloads -name "wlasl_100*.keras" 2>/dev/null)

if [ -z "$MODELS" ]; then
    echo "❌ No WLASL-100 models found in Downloads"
    echo ""
    echo "Please download your model from Kaggle first:"
    echo "1. Go to your Kaggle notebook"
    echo "2. Download wlasl_100_best.keras"
    echo "3. Download labels_wlasl_100.txt"
    echo "4. Place them in ~/Downloads/"
    echo ""
    exit 1
fi

echo "✅ Found models:"
echo "$MODELS" | while read model; do
    size=$(du -h "$model" | cut -f1)
    echo "  - $(basename "$model") ($size)"
done

echo ""
echo "Select model:"
select MODEL_PATH in $MODELS; do
    if [ -n "$MODEL_PATH" ]; then
        break
    fi
done

echo ""
echo "Selected: $(basename "$MODEL_PATH")"

# Find corresponding labels
LABEL_PATH=$(find ~/Downloads -name "labels_wlasl_100*.txt" 2>/dev/null | head -1)

if [ -z "$LABEL_PATH" ]; then
    echo "❌ Labels file not found!"
    echo "Looking for: labels_wlasl_100.txt in ~/Downloads"
    exit 1
fi

echo "Labels: $(basename "$LABEL_PATH")"
echo ""

# Step 3: Integrate model
echo "=================================="
echo "  STEP 3: INTEGRATE MODEL"
echo "=================================="
echo ""

python integrate_kaggle_model.py \
    --model "$MODEL_PATH" \
    --labels "$LABEL_PATH"

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "  ✅ SUCCESS!"
    echo "=================================="
    echo ""
    echo "Your ISL Recognition now supports 100 words!"
    echo ""
    read -p "Start the web app now? (y/n): " start_app
    
    if [ "$start_app" = "y" ]; then
        echo ""
        echo "🚀 Starting web app..."
        echo "Open: http://localhost:8080"
        echo ""
        ./run_app.sh
    fi
else
    echo ""
    echo "❌ Integration failed!"
    echo "Check the error messages above"
fi
