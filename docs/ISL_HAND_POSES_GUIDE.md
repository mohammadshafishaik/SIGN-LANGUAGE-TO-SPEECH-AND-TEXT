# 🖐️ ISL (Indian Sign Language) Hand Poses Guide

## 📋 **What the Model Recognizes:**

### **35 Classes Total:**
- **26 Letters:** A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
- **9 Numbers:** 1, 2, 3, 4, 5, 6, 7, 8, 9

---

## 🔤 **LETTER HAND POSES (ISL Fingerspelling)**

### **Common ISL Letter Signs:**

**A** - Closed fist with thumb on the side
**B** - Flat hand, fingers together, thumb across palm
**C** - Hand curved like the letter C
**D** - Index finger up, other fingers touch thumb
**E** - All fingers curled down touching thumb
**F** - Index and thumb make circle, other three fingers up
**G** - Fist with index finger and thumb pointing sideways
**H** - Fist with index and middle finger extended horizontally
**I** - Pinky finger extended, other fingers closed
**J** - Pinky extended, trace a J motion
**K** - Index and middle finger up in V, thumb touches middle finger
**L** - Index finger up, thumb out at 90° angle
**M** - Three fingers over thumb
**N** - Two fingers over thumb
**O** - All fingertips touch, forming O shape
**P** - Like K but pointing down
**Q** - Index and thumb pointing down
**R** - Index and middle crossed
**S** - Fist with thumb across fingers
**T** - Thumb between index and middle
**U** - Index and middle finger up together
**V** - Index and middle finger up in V shape
**W** - Index, middle, ring finger up
**X** - Index finger crooked/bent
**Y** - Thumb and pinky extended (hang loose gesture)
**Z** - Draw Z shape with index finger

---

## 🔢 **NUMBER HAND POSES (ISL Numbers)**

### **Numbers 1-9:**

**1** - ☝️ Index finger extended, other fingers closed
**2** - ✌️ Index and middle finger extended (peace sign)
**3** - 🤟 Index, middle, and ring finger extended
**4** - 🖐️ Four fingers extended, thumb tucked in
**5** - ✋ All five fingers spread open
**6** - Three fingers up, pinky and thumb touch
**7** - Four fingers down, thumb and pinky touch (or ring/pinky up with thumb)
**8** - Index and thumb touch, other three fingers up
**9** - Index finger and thumb form circle (like OK sign)

---

## ❓ **Why You Might Not See Numbers:**

### **Possible Reasons:**

1. **Numbers are harder to distinguish** - Some ISL number signs are very similar to letters:
   - Number **1** vs Letter **D** or **I**
   - Number **2** vs Letter **V** or **U**
   - Number **3** vs Letter **W**
   - Number **5** vs Letter **B**
   - Number **9** vs Letter **F** or **O**

2. **Model might prefer letter predictions** - Since letters (26 classes) outnumber digits (9 classes), the model might have a bias toward predicting letters

3. **Hand positioning matters** - Numbers and letters may require specific orientations:
   - **Palm facing camera** vs **palm facing you**
   - **Fingers pointing up** vs **fingers pointing sideways**

4. **Training data variation** - The ISL dataset images might show numbers in specific poses that differ from natural signing

---

## 🧪 **How to Test Numbers:**

### **Try these clear number signs:**

1. **For "1"**: Point index finger straight up, make fist with other fingers
2. **For "2"**: Peace sign (V shape) with palm facing camera
3. **For "3"**: Extend index, middle, ring finger (like "W" but different angle)
4. **For "5"**: Open palm, all fingers spread wide

### **Tips for Better Recognition:**

✅ **Hold hand steady** for 1-2 seconds
✅ **Keep hand centered** in camera view
✅ **Ensure good lighting** on your hand
✅ **Try different angles** - rotate hand slightly
✅ **Watch the confidence score** - if it's low (<50%), try adjusting

---

## 🔍 **Checking the Model's Label Mapping:**

The model was trained on these exact classes. Let me verify what classes exist...

Run this to see all 35 classes:
```python
import json
with open('dataset/splits_isl/label_mappings.json', 'r') as f:
    labels = json.load(f)
print("Classes:", sorted(labels['label_to_idx'].keys()))
```

---

## 📊 **Model Performance by Class:**

From training results:
- **Perfect (100% F1)**: 1, 2, 3, 4, 5, 6, 7, 8, A, D, E, F, G, H, J, K, L, M, R, T, U, W, X, Y, Z
- **Near-perfect (99% F1)**: 9, B, I, N, S, V
- **Good (94-97% F1)**: C, O, P
- **Acceptable (86% F1)**: Q

**All 9 numbers (1-9) have perfect or near-perfect accuracy!**

So the model CAN recognize numbers - you just need to show the right poses!

---

## 💡 **Next Steps:**

1. **Check label_mappings.json** to confirm all 35 classes are loaded
2. **Try very distinct poses** - like number 5 (open hand) vs letter A (fist)
3. **Hold pose longer** - model predicts every 3 frames, give it time
4. **Check confidence threshold** - script only shows predictions >50% confidence

Would you like me to:
- Add visual debugging to show ALL predictions (not just top one)?
- Lower the confidence threshold?
- Create a testing mode that shows top 3 predictions?
