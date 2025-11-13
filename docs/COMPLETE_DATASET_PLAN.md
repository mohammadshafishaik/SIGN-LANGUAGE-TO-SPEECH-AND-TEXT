# 🌍 Complete Sign Language Dataset Training Plan

## Goal
Train a **universal sign language recognition system** that can recognize:
- ✅ **ISL fingerspelling** (A-Z, 1-9) - DONE ✓
- 🎯 **ASL words** (HELLO, LIKE, WANT, etc.) - 2,000+ words
- 🎯 **Common phrases** (THANK YOU, GOOD MORNING, etc.)
- 🎯 **Multiple sign languages** (ASL, ISL, BSL, etc.)

---

## 📊 Available Datasets

### 1. **WLASL (Word-Level American Sign Language)** ⭐ PRIORITY
- **Size**: 2,000 words, 21,083 videos
- **Source**: https://github.com/dxli94/WLASL
- **Download**: Videos from YouTube (automated script)
- **Classes**: Common words (HELLO, THANK YOU, BOOK, etc.)
- **Status**: Script ready, need to download videos

### 2. **MS-ASL (Microsoft American Sign Language)**
- **Size**: 1,000 classes, 25,513 videos
- **Source**: https://www.microsoft.com/en-us/research/project/ms-asl/
- **Download**: Direct download (200GB)
- **Quality**: High quality, studio recorded
- **Status**: Needs download

### 3. **AUTSL (Turkish Sign Language)**
- **Size**: 226 signs, 38,336 videos
- **Source**: https://chalearnlap.cvc.uab.cat/dataset/40/description/
- **Download**: Registration required
- **Status**: Alternative dataset

### 4. **YouTube-ASL (Continuous)**
- **Size**: 1,000+ hours of continuous signing
- **Source**: Various YouTube channels
- **Download**: Can scrape with yt-dlp
- **Status**: Advanced - for later

---

## 🎯 **RECOMMENDED APPROACH: Start with WLASL**

### Why WLASL First?
1. ✅ **Most practical** - Real-world ASL words
2. ✅ **Well documented** - JSON metadata included
3. ✅ **Preprocessing script ready** - Already created
4. ✅ **2,000 words** - Covers most daily communication

---

## 📥 Step-by-Step: Download WLASL Dataset

### Option A: Download Top 100 Words (FASTEST - ~30 minutes)
```bash
# Download most common 100 ASL words
python data_collector/download_wlasl.py --num_words 100
```
**Words include**: HELLO, THANK YOU, PLEASE, WANT, LIKE, BOOK, etc.

### Option B: Download Top 300 Words (RECOMMENDED - ~2 hours)
```bash
# Download 300 most frequent words
python data_collector/download_wlasl.py --num_words 300
```

### Option C: Download ALL 2,000 Words (COMPLETE - ~12 hours)
```bash
# Download complete WLASL dataset
python data_collector/download_wlasl.py --num_words 2000
```

---

## 🔄 Complete Training Pipeline

### Phase 1: ISL (Indian Sign Language) ✅ COMPLETE
- [x] 35 classes (A-Z, 1-9)
- [x] 42,745 images
- [x] 99.02% test accuracy
- [x] Real-time webcam working

### Phase 2: WLASL (American Sign Language Words) 🎯 CURRENT
```bash
# 1. Download videos (choose one option above)
python data_collector/download_wlasl.py --num_words 100

# 2. Extract keypoints from videos
python data_prep/preprocess_wlasl.py

# 3. Train LSTM model (for temporal sequences)
python models/train_wlasl.py

# 4. Test real-time recognition
python inference/realtime_wlasl.py
```

### Phase 3: Combined System 🚀 FUTURE
```bash
# Multi-modal system:
# - ISL for fingerspelling
# - WLASL for words
# - TTS for speech output
python inference/realtime_combined.py
```

---

## 💾 Storage Requirements

| Dataset | Videos | Size | Processing Time |
|---------|--------|------|-----------------|
| WLASL (100 words) | ~1,000 | ~5 GB | 30 min |
| WLASL (300 words) | ~3,000 | ~15 GB | 2 hours |
| WLASL (2000 words) | ~21,000 | ~100 GB | 12 hours |
| MS-ASL | 25,000 | ~200 GB | 24 hours |

**Your Mac**: Check available space before downloading!

---

## 🎓 Expected Results

### After Training on WLASL:
- ✅ Recognize ASL words: HELLO, THANK YOU, PLEASE
- ✅ Handle temporal sequences (multiple frames)
- ✅ ~85-90% accuracy on test set
- ✅ Real-time word recognition in webcam

### Model Architecture:
- **LSTM/GRU** for temporal modeling
- Input: Sequence of keypoints (100 frames × 144 features)
- Output: 100/300/2000 word classes

---

## 🔥 **YOUR NEXT COMMAND**

Choose ONE of these:

### 🚀 **RECOMMENDED: Start Small (100 words)**
```bash
# Create download script first
python data_collector/download_wlasl.py --num_words 100
```
⏱️ Time: ~30 minutes | 💾 Space: ~5 GB

### 🎯 **BALANCED: 300 Words**
```bash
python data_collector/download_wlasl.py --num_words 300
```
⏱️ Time: ~2 hours | 💾 Space: ~15 GB

### 💪 **ALL-IN: Complete Dataset (2000 words)**
```bash
python data_collector/download_wlasl.py --num_words 2000
```
⏱️ Time: ~12 hours | 💾 Space: ~100 GB

---

## 📝 Notes
- Videos download from YouTube (requires `yt-dlp`)
- Some videos may be unavailable (deleted/private)
- Preprocessing extracts keypoints (reduces size ~90%)
- Training can start with partial dataset (100 words)
- Can add more words later incrementally

**Ready to download? Say which option you prefer!**
