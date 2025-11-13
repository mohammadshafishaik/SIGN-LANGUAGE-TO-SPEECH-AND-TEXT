# 🚀 DOWNLOAD YOUR KAGGLE-TRAINED MODEL

## Complete Guide to Download and Integrate Your Trained Model

You trained a model on Kaggle with **50-100 classes** and achieved good accuracy. Now let's bring it to your Mac and integrate it with your ISL web app!

---

## 📋 STEP 1: Install Kaggle API

First, install the Kaggle CLI tool:

```bash
pip install kaggle
```

---

## 🔑 STEP 2: Setup Kaggle API Credentials

### Option A: Download from Kaggle Website (EASIEST)

1. Go to **https://www.kaggle.com/account**
2. Scroll to **API** section
3. Click **"Create New API Token"**
4. This downloads `kaggle.json` to your Downloads folder
5. Move it to the right location:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Option B: Manual Setup

Create `~/.kaggle/kaggle.json` with your credentials:

```json
{
  "username": "YOUR_KAGGLE_USERNAME",
  "key": "YOUR_API_KEY"
}
```

Then set permissions:
```bash
chmod 600 ~/.kaggle/kaggle.json
```

---

## 📥 STEP 3: Download Your Trained Model

### Method 1: Download from Kaggle Notebook Output

If you saved your model in a Kaggle notebook:

```bash
# List your notebooks
kaggle kernels list --mine

# Download output from specific notebook
kaggle kernels output YOUR_USERNAME/YOUR_NOTEBOOK_NAME -p ./kaggle_output
```

### Method 2: Download from Kaggle Dataset

If you uploaded your model as a dataset:

```bash
# List your datasets
kaggle datasets list --mine

# Download specific dataset
kaggle datasets download YOUR_USERNAME/YOUR_DATASET_NAME -p ./kaggle_output
```

### Method 3: Manual Download (If API doesn't work)

1. Go to your Kaggle notebook: **https://www.kaggle.com/code**
2. Open your training notebook
3. Click **"Output"** tab on the right
4. Download these files:
   - `wlasl_50_best.keras` (or `wlasl_100_best.keras`)
   - `wlasl_labels.txt` (or `labels_wlasl_50.txt`)
   - `training_50.png` (optional - training graphs)

---

## 🔧 STEP 4: Extract and Move Files

After downloading, extract and move to your project:

```bash
# If you downloaded a zip file
cd kaggle_output
unzip *.zip

# Move model files to checkpoints
mv wlasl_*_best.keras ~/ML_PROJECT_LOCAL/checkpoints/wlasl_model.keras
mv *labels*.txt ~/ML_PROJECT_LOCAL/checkpoints/wlasl_labels.txt

# Verify files
ls -lh ~/ML_PROJECT_LOCAL/checkpoints/
```

---

## 🎯 STEP 5: Integrate with Your Web App

The model is now ready! Run the integration script:

```bash
cd ~/ML_PROJECT_LOCAL
python integrate_kaggle_model.py
```

This will:
- ✅ Verify your downloaded model
- ✅ Check compatibility with your app
- ✅ Update the web app configuration
- ✅ Test the model with sample data
- ✅ Start the web interface

---

## 🌐 STEP 6: Test Your Model

Once integrated, test it:

```bash
# Start the web app
python inference/app.py
```

Then open: **http://localhost:8080**

You should see:
- ✅ Your trained model loaded
- ✅ 50-100 word recognition (instead of 25)
- ✅ Higher accuracy predictions
- ✅ Real-time webcam recognition

---

## 🔍 TROUBLESHOOTING

### Problem: "kaggle: command not found"

```bash
# Install kaggle
pip install kaggle

# Or with pip3
pip3 install kaggle
```

### Problem: "401 Unauthorized"

Your API credentials are wrong. Re-download `kaggle.json` from Kaggle.

### Problem: "Could not find notebook/dataset"

List your resources:
```bash
kaggle kernels list --mine
kaggle datasets list --mine
```

### Problem: Model file not compatible

Your model might be from a different TensorFlow version. Check:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

If version mismatch, reinstall TensorFlow:
```bash
pip install tensorflow==2.15.0
```

### Problem: Different feature dimensions

Your Kaggle model expects different input shape. Check the training script to see:
- How many frames? (30 frames)
- How many landmarks? (104 landmarks × 3 coords = 312D)

---

## 📊 WHAT YOU TRAINED ON KAGGLE

Based on your scripts, you trained:

**Model Architecture:**
- Transformer encoder (3 blocks)
- Bidirectional LSTM (2 layers)
- Dense layers with dropout
- Mixed precision training

**Dataset:**
- WLASL videos (50-100 classes)
- MediaPipe keypoint extraction
- 30 frames per video
- 104 landmarks (pose + face + hands)

**Expected Accuracy:**
- 50 classes: **85-90%**
- 100 classes: **80-85%**

---

## 🎉 NEXT STEPS

After integration:

1. **Test with webcam** - Try signing the words your model knows
2. **Check accuracy** - See if predictions match your signs
3. **Fine-tune** - If accuracy is low, retrain with more data
4. **Expand vocabulary** - Train on more words (200, 300, 600)

---

## 💡 QUICK COMMANDS SUMMARY

```bash
# 1. Install Kaggle
pip install kaggle

# 2. Setup credentials
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 3. Download model (replace with your notebook name)
kaggle kernels output YOUR_USERNAME/wlasl-training -p ./kaggle_output

# 4. Move files
cd kaggle_output
mv wlasl_*_best.keras ~/ML_PROJECT_LOCAL/checkpoints/wlasl_model.keras
mv *labels*.txt ~/ML_PROJECT_LOCAL/checkpoints/wlasl_labels.txt

# 5. Integrate
cd ~/ML_PROJECT_LOCAL
python integrate_kaggle_model.py

# 6. Run web app
python inference/app.py
```

---

## 📞 NEED HELP?

If you get stuck, run the diagnostic script:

```bash
python diagnose_kaggle_setup.py
```

This will check:
- ✅ Kaggle API installed
- ✅ Credentials configured
- ✅ Model files present
- ✅ TensorFlow version
- ✅ Web app configuration

---

**Ready to download your model? Let's do this! 🚀**
