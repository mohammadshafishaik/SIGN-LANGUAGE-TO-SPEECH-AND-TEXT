#!/bin/bash
# Cleanup script - Remove unnecessary files before downloading proper dataset
# This will free up ~5GB of space

echo "🧹 CLEANUP - Removing Unnecessary Files"
echo "========================================"
echo ""

# Navigate to project directory
cd /Users/shaikshafi/ML_PROJECT_LOCAL

echo "📊 Current disk usage:"
du -sh . 2>/dev/null
echo ""

# 1. Remove old datasets (insufficient data - only 10-40 samples/class)
echo "1️⃣  Removing old dataset directories..."
rm -rf "dataset" "dataset 2" "datasets" 2>/dev/null
echo "   ✅ Removed: dataset/, dataset 2/, datasets/"

# 2. Remove archive.zip and extracted features (insufficient data)
echo ""
echo "2️⃣  Removing archive.zip and extracted features..."
rm -f "archive.zip" "wlasl_100_processed.zip" 2>/dev/null
rm -rf "WLASL features npy" 2>/dev/null
echo "   ✅ Removed: archive.zip (479MB), wlasl_100_processed.zip, WLASL features npy/"

# 3. Remove old training scripts (obsolete)
echo ""
echo "3️⃣  Removing obsolete training scripts..."
rm -f train_wlasl_30_HIGH_ACCURACY.py \
      train_wlasl_CLEAN_90PERCENT.py \
      train_wlasl100_colab.py \
      download_wlasl_fast.py \
      download_wlasl_2000.py \
      download_wlasl100.py \
      download_proper_dataset.py 2>/dev/null
echo "   ✅ Removed old training scripts"

# 4. Remove old models (low accuracy - 39% and 21%)
echo ""
echo "4️⃣  Removing low-accuracy models..."
rm -f "wlasl_100_best (1).keras" \
      "wlasl_top30_best.keras" \
      checkpoints/wlasl_100_fast.keras \
      checkpoints/wlasl_100_improved_best.keras \
      checkpoints/wlasl_100_best.keras 2>/dev/null
echo "   ✅ Removed old models (keeping isl_best.keras - 85% accuracy)"

# 5. Remove old results and logs
echo ""
echo "5️⃣  Removing old results and logs..."
rm -f results_top30.json \
      results_wlasl_100.json \
      training_top30.png \
      training_wlasl_100.png \
      download.log 2>/dev/null
rm -rf logs/ results/ 2>/dev/null
echo "   ✅ Removed old results and logs"

# 6. Remove obsolete documentation files
echo ""
echo "6️⃣  Removing obsolete documentation..."
rm -f AUTSL_15GB_BROWSER_DOWNLOAD.md \
      ASL_WORDS_PLAN.md \
      COLAB_SIMPLE_GUIDE.md \
      DATASET_READY.md \
      DOWNLOAD_ISSUE_SOLUTIONS.md \
      DUAL_DOWNLOAD_STATUS.md \
      FINAL_FIX_SUMMARY.md \
      FIXES_APPLIED.md \
      IMPROVEMENTS_FOR_HIGH_ACCURACY.md \
      ISL_DOWNLOAD_STATUS.md \
      ISL_PROCESSING_STATUS.md \
      OPTION_A_ISL_STATUS.md \
      PLAN_90_PERCENT_ACCURACY.md \
      PREMIUM_FEATURES_FIXED.md \
      PROGRESS_SUMMARY.md \
      PROJECT_COMPLETE_STATUS.md \
      QUICK_START_ARCHIVE.md \
      SERVER_RESTARTED.md \
      STATUS.md \
      TRAINING_100_WORDS_PLAN.md \
      TRAINING_GUIDE_ARCHIVE.md \
      WLASL_2000_PLAN.md \
      WLASL_DOWNLOAD_STATUS.md \
      WHAT_HAPPENED.txt \
      fix_premium_features.txt 2>/dev/null
echo "   ✅ Removed obsolete docs (keeping README, WEB_APP_GUIDE, TROUBLESHOOTING)"

# 7. Remove old download/monitoring scripts
echo ""
echo "7️⃣  Removing old download scripts..."
rm -f download_best_wlasl.py \
      download_datasets.sh \
      download_wlasl100_official.sh \
      monitor_download.py \
      monitor_wlasl_download.py \
      move_to_local.sh \
      prepare_for_colab.sh \
      run_pipeline.sh \
      env_setup.sh 2>/dev/null
echo "   ✅ Removed old download/monitoring scripts"

# 8. Remove test files
echo ""
echo "8️⃣  Removing test files..."
rm -f test_gemini_voice.html \
      test_video.mp4 \
      debug_data.py \
      debug_webcam.py \
      validate_dataset.py \
      extract_training_images.py 2>/dev/null
echo "   ✅ Removed test files"

# 9. Remove old notebooks
echo ""
echo "9️⃣  Removing old notebooks..."
rm -f train_wlasl100_colab.ipynb 2>/dev/null
echo "   ✅ Removed old notebooks"

# 10. Remove old label files
echo ""
echo "🔟 Removing old label files..."
rm -f labels_wlasl_100.txt \
      checkpoints/wlasl_100_fast_results.json \
      checkpoints/wlasl_100_best_results.json \
      checkpoints/wlasl_100_results.json 2>/dev/null
echo "   ✅ Removed old label files"

# Summary
echo ""
echo "========================================"
echo "✅ CLEANUP COMPLETE!"
echo "========================================"
echo ""
echo "📊 New disk usage:"
du -sh . 2>/dev/null
echo ""

echo "📝 KEPT (Important files):"
echo "   ✅ checkpoints/isl_best.keras (85% accuracy - ISL model)"
echo "   ✅ checkpoints/isl_final.keras"
echo "   ✅ checkpoints/labels.txt"
echo "   ✅ inference/ (web app)"
echo "   ✅ models/ (training code)"
echo "   ✅ pose_extractor/ (MediaPipe)"
echo "   ✅ data_prep/ (preprocessing scripts)"
echo "   ✅ data_collector/ (data collection)"
echo "   ✅ venv/ (Python environment)"
echo "   ✅ project_paths.py (path management)"
echo "   ✅ requirements.txt"
echo "   ✅ README.md, WEB_APP_GUIDE.md, TROUBLESHOOTING.md"
echo "   ✅ train_wlasl_600_COLAB.py (current training script)"
echo "   ✅ datasets_local.nosync/ (will store new dataset here)"
echo ""

echo "🗑️  REMOVED (~5GB freed):"
echo "   ❌ Old datasets (dataset/, dataset 2/, datasets/)"
echo "   ❌ archive.zip (479MB - insufficient data)"
echo "   ❌ Old training scripts (obsolete)"
echo "   ❌ Low-accuracy models (39%, 21%)"
echo "   ❌ Old results and logs"
echo "   ❌ Obsolete documentation (25+ MD files)"
echo "   ❌ Old download/monitoring scripts"
echo "   ❌ Test files"
echo ""

echo "🎯 NEXT STEP: Download Full WLASL-100 Dataset"
echo "   Expected: 50K+ samples, 15-20GB"
echo "   Target accuracy: 90-95%"
echo ""
