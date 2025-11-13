#!/bin/bash

# 🚀 QUICK START SCRIPT
# Automatically sets up and runs your ISL Recognition app

set -e

echo "=================================="
echo "  ISL RECOGNITION - QUICK START"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠ Virtual environment not found${NC}"
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Check TensorFlow
echo ""
echo -e "${BLUE}Checking TensorFlow...${NC}"
TF_VERSION=$(python -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null || echo "not_installed")

if [ "$TF_VERSION" = "not_installed" ]; then
    echo -e "${YELLOW}⚠ TensorFlow not installed${NC}"
    echo "Installing TensorFlow..."
    pip install tensorflow-macos tensorflow-metal
else
    echo -e "${GREEN}✓ TensorFlow $TF_VERSION${NC}"
fi

# Check if model exists
echo ""
echo -e "${BLUE}Checking model files...${NC}"

if [ -f "checkpoints/isl_best.keras" ]; then
    echo -e "${GREEN}✓ ISL model found${NC}"
    MODEL_EXISTS=true
else
    echo -e "${YELLOW}⚠ No model found in checkpoints/${NC}"
    MODEL_EXISTS=false
fi

# Check for Kaggle models
KAGGLE_MODELS=$(find ~/Downloads -name "wlasl*.keras" 2>/dev/null | head -1)

if [ -n "$KAGGLE_MODELS" ] && [ "$MODEL_EXISTS" = false ]; then
    echo ""
    echo -e "${BLUE}Found Kaggle models in Downloads!${NC}"
    echo ""
    echo "Do you want to integrate a Kaggle model?"
    echo "1) Yes - Use Kaggle model (30-100 words)"
    echo "2) No - Train a new model locally"
    echo "3) Skip - Just run with existing setup"
    echo ""
    read -p "Choice (1-3): " choice
    
    case $choice in
        1)
            echo ""
            echo "Available Kaggle models:"
            ls -lh ~/Downloads/wlasl*.keras 2>/dev/null | awk '{print $9, "(" $5 ")"}'
            echo ""
            read -p "Enter model filename (e.g., wlasl_30_best.keras): " model_name
            
            MODEL_PATH="$HOME/Downloads/$model_name"
            LABEL_NAME=$(echo $model_name | sed 's/_best.keras//' | sed 's/wlasl/labels_wlasl/')
            LABEL_PATH="$HOME/Downloads/${LABEL_NAME}.txt"
            
            if [ -f "$MODEL_PATH" ] && [ -f "$LABEL_PATH" ]; then
                echo ""
                echo "Integrating model..."
                python integrate_kaggle_model.py --model "$MODEL_PATH" --labels "$LABEL_PATH"
                MODEL_EXISTS=true
            else
                echo -e "${YELLOW}⚠ Model or labels not found${NC}"
                echo "Model: $MODEL_PATH"
                echo "Labels: $LABEL_PATH"
            fi
            ;;
        2)
            echo ""
            echo "Training new model..."
            echo "This will take 30-60 minutes."
            read -p "Continue? (y/n): " confirm
            if [ "$confirm" = "y" ]; then
                python models/train_wlasl100.py
                MODEL_EXISTS=true
            fi
            ;;
        3)
            echo "Skipping..."
            ;;
    esac
fi

# Start the app
echo ""
echo "=================================="
echo "  STARTING WEB APP"
echo "=================================="
echo ""

if [ "$MODEL_EXISTS" = true ]; then
    echo -e "${GREEN}✓ Model ready!${NC}"
    echo ""
    echo "Starting web server..."
    echo "Open in browser: ${BLUE}http://localhost:8080${NC}"
    echo ""
    echo "Press Ctrl+C to stop"
    echo ""
    
    python inference/app.py
else
    echo -e "${YELLOW}⚠ No model available${NC}"
    echo ""
    echo "Options:"
    echo "1. Download model from Kaggle"
    echo "2. Train a new model: python models/train_wlasl100.py"
    echo "3. Use existing ISL model if available"
    echo ""
fi
