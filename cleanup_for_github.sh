#!/bin/bash

# Remove unnecessary files for GitHub
# Keep only what's needed to run the app

echo "🧹 Cleaning up repository for GitHub..."
echo ""

# Remove extra documentation
rm -f FINAL_SOLUTION.md
rm -f GOOGLE_DRIVE_METHOD.md
rm -f MANUAL_DOWNLOAD_LINKS.md
rm -f OPTION_2_FULL_WLASL_GUIDE.md
rm -f STATUS_CLEANUP_COMPLETE.md
rm -f TROUBLESHOOTING.md
rm -f SIMPLE_SOLUTION.md
rm -f DOWNLOAD_KAGGLE_MODEL.md
rm -f .LOCAL_STORAGE

# Remove training scripts not needed for running
rm -f KAGGLE_TRAIN_WLASL.py
rm -f KAGGLE_TRAIN_WLASL_FIXED.py
rm -f COLAB_TRAIN_CLEAN.py
rm -f PROCESS_AND_TRAIN_VIDEOS.py
rm -f train_wlasl_600_COLAB.py
rm -f train_wlasl_600_COLAB_SIMPLE.py
rm -f download_wlasl_full_dataset.py
rm -f test_wlasl_model.py

# Remove extra data prep scripts
rm -f data_prep/aggressive_download.py
rm -f data_prep/download_dataset.sh
rm -f data_prep/download_wlasl_videos.py
rm -f data_prep/merge_datasets.py
rm -f data_prep/prepare_wlasl50_best.py
rm -f data_prep/preprocess_isl_images.py
rm -f data_prep/preprocess_local.py
rm -f data_prep/preprocess_local_dataset.py
rm -f data_prep/preprocess_signavatars.py
rm -f data_prep/preprocess_wlasl_enhanced.py
rm -f data_prep/process_wlasl100.py
rm -f data_prep/process_wlasl100_fixed.py
rm -f data_prep/process_wlasl100_json.py
rm -f data_prep/resplit_top_classes.py
rm -f data_prep/smart_download.py
rm -f data_prep/split_dataset.py

# Remove extra inference scripts
rm -f inference/compare_1_vs_g.py
rm -f inference/create_detailed_references.py
rm -f inference/create_reference_sheet.py
rm -f inference/diagnose_mediapipe_bug.py
rm -f inference/realtime_isl_debug.py
rm -f inference/test_feature_extraction.py
rm -f inference/test_number_recognition.py
rm -f inference/view_dataset_numbers.py
rm -f inference/web_interface.py
rm -f inference/web_simple.py

# Remove extra model training scripts
rm -f models/train_multidataset.py
rm -f models/train_robust.py
rm -f models/train_wlasl100_80percent.py
rm -f models/train_wlasl100_fast.py
rm -f models/train_wlasl100_improved.py
rm -f models/train_wlasl100_optimized.py

# Remove extra templates
rm -f inference/templates/index.html
rm -f inference/templates/index_simple.html
rm -f inference/templates/webapp.html
rm -f inference/templates/webapp_backup_*.html
rm -f templates/index.html

# Remove extra docs
rm -rf docs/

# Remove conversion scripts (not needed for basic usage)
rm -f convert_keras3_to_keras2.py

# Remove old cleanup script
rm -f cleanup_old_files.sh

echo "✅ Cleanup complete!"
echo ""
echo "Kept essential files:"
echo "  ✓ inference/webapp_simple.py (main app)"
echo "  ✓ inference/app.py (alternative app)"
echo "  ✓ inference/realtime_isl_FIXED.py"
echo "  ✓ models/train_isl.py"
echo "  ✓ models/train_wlasl.py"
echo "  ✓ data_collector/"
echo "  ✓ requirements.txt"
echo "  ✓ README files"
echo "  ✓ Setup scripts"
echo ""
