#!/bin/bash

# 🚀 QUICK KAGGLE MODEL DOWNLOAD SCRIPT
# Downloads your trained model from Kaggle

set -e  # Exit on error

echo "=================================="
echo "  KAGGLE MODEL DOWNLOAD"
echo "=================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo -e "${RED}❌ Kaggle CLI not installed${NC}"
    echo ""
    echo "Install with:"
    echo "  pip install kaggle"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ Kaggle CLI found${NC}"

# Check credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo -e "${RED}❌ Kaggle credentials not found${NC}"
    echo ""
    echo "Setup instructions:"
    echo "  1. Go to https://www.kaggle.com/account"
    echo "  2. Click 'Create New API Token'"
    echo "  3. Move kaggle.json to ~/.kaggle/"
    echo "  4. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ Kaggle credentials found${NC}"
echo ""

# Create output directory
mkdir -p kaggle_output
cd kaggle_output

echo "=================================="
echo "  YOUR KAGGLE RESOURCES"
echo "=================================="
echo ""

echo -e "${BLUE}📓 Your Notebooks:${NC}"
kaggle kernels list --mine --page-size 10

echo ""
echo -e "${BLUE}📊 Your Datasets:${NC}"
kaggle datasets list --mine --page-size 10

echo ""
echo "=================================="
echo "  DOWNLOAD OPTIONS"
echo "=================================="
echo ""

echo "Choose download method:"
echo ""
echo "1) Download from Kaggle Notebook output"
echo "2) Download from Kaggle Dataset"
echo "3) Manual download instructions"
echo "4) Exit"
echo ""

read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo -e "${BLUE}Enter your notebook name (e.g., username/notebook-name):${NC}"
        read notebook_name
        
        echo ""
        echo "Downloading notebook output..."
        kaggle kernels output "$notebook_name" -p .
        
        echo ""
        echo -e "${GREEN}✓ Download complete!${NC}"
        echo ""
        echo "Files downloaded to: $(pwd)"
        ls -lh
        ;;
        
    2)
        echo ""
        echo -e "${BLUE}Enter your dataset name (e.g., username/dataset-name):${NC}"
        read dataset_name
        
        echo ""
        echo "Downloading dataset..."
        kaggle datasets download "$dataset_name" -p .
        
        echo ""
        echo "Extracting..."
        unzip -o *.zip
        
        echo ""
        echo -e "${GREEN}✓ Download complete!${NC}"
        echo ""
        echo "Files downloaded to: $(pwd)"
        ls -lh
        ;;
        
    3)
        echo ""
        echo "=================================="
        echo "  MANUAL DOWNLOAD INSTRUCTIONS"
        echo "=================================="
        echo ""
        echo "1. Go to your Kaggle notebook:"
        echo "   https://www.kaggle.com/code"
        echo ""
        echo "2. Open your training notebook"
        echo ""
        echo "3. Click 'Output' tab on the right"
        echo ""
        echo "4. Download these files:"
        echo "   - wlasl_50_best.keras (or wlasl_100_best.keras)"
        echo "   - wlasl_labels.txt"
        echo "   - training_50.png (optional)"
        echo ""
        echo "5. Move files to: $(pwd)"
        echo ""
        exit 0
        ;;
        
    4)
        echo "Exiting..."
        exit 0
        ;;
        
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

# Check what was downloaded
echo ""
echo "=================================="
echo "  DOWNLOADED FILES"
echo "=================================="
echo ""

# Find model files
model_files=$(find . -name "*.keras" -o -name "*.h5" 2>/dev/null)
label_files=$(find . -name "*label*.txt" -o -name "labels.txt" 2>/dev/null)

if [ -n "$model_files" ]; then
    echo -e "${GREEN}✓ Model files found:${NC}"
    echo "$model_files" | while read file; do
        size=$(du -h "$file" | cut -f1)
        echo "  - $(basename "$file") ($size)"
    done
else
    echo -e "${YELLOW}⚠ No model files found${NC}"
fi

echo ""

if [ -n "$label_files" ]; then
    echo -e "${GREEN}✓ Label files found:${NC}"
    echo "$label_files" | while read file; do
        lines=$(wc -l < "$file")
        echo "  - $(basename "$file") ($lines classes)"
    done
else
    echo -e "${YELLOW}⚠ No label files found${NC}"
fi

echo ""
echo "=================================="
echo "  NEXT STEPS"
echo "=================================="
echo ""

if [ -n "$model_files" ] && [ -n "$label_files" ]; then
    echo -e "${GREEN}✓ Ready to integrate!${NC}"
    echo ""
    echo "Run the integration script:"
    echo "  cd .."
    echo "  python integrate_kaggle_model.py"
    echo ""
else
    echo -e "${YELLOW}⚠ Missing files${NC}"
    echo ""
    echo "Make sure you have:"
    echo "  - Model file (.keras or .h5)"
    echo "  - Labels file (.txt)"
    echo ""
fi
