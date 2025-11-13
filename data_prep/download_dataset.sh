#!/bin/bash

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Script: download_dataset.sh
# Description: Downloads the WLASL dataset.
# Author: GitHub Copilot
# Date: 2025-11-08
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Stop on first error
set -e

# --- Configuration ---
WLASL_REPO="https://github.com/dxli94/WLASL.git"
TARGET_DIR_BASE="datasets"

# --- Functions ---

# Function to print colored messages
print_msg() {
    COLOR=$1
    MSG=$2
    NC='\033[0m' # No Color
    case $COLOR in
        "green")
            printf "${NC}[✅] ${MSG}\n"
            ;;
        "blue")
            printf "${NC}[ℹ️] ${MSG}\n"
            ;;
        "red")
            printf "${NC}[❌] ${MSG}\n"
            ;;
        *)
            printf "${NC}${MSG}\n"
            ;;
    esac
}

# Function to download the WLASL dataset
download_wlasl() {
    local target_dir="${TARGET_DIR_BASE}/WLASL"
    print_msg "blue" "Downloading WLASL dataset..."
    if [ -d "$target_dir" ]; then
        print_msg "blue" "WLASL directory already exists. Skipping download."
    else
        git clone "$WLASL_REPO" "$target_dir"
        print_msg "green" "WLASL dataset downloaded successfully to ${target_dir}."
        
        # Optional: Print size
        print_msg "blue" "Calculating WLASL dataset size..."
        du -sh "$target_dir"
    fi
}

# --- Main Execution ---

main() {
    print_msg "blue" "Starting dataset download process..."
    
    # Create the base directory if it doesn't exist
    mkdir -p "$TARGET_DIR_BASE"
    
    # Download datasets
    download_wlasl
    
    print_msg "green" "WLASL dataset downloaded successfully."
}

# Run main function
main
