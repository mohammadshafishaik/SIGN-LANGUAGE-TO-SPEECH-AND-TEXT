"""
Fix for "1" being predicted as "G"
Shows side-by-side comparison and specific adjustments needed
"""

import cv2
import numpy as np
from pathlib import Path
import random

def show_1_vs_g_comparison():
    """Show examples of number 1 vs letter G to understand the difference"""
    
    dataset_root = Path("datasets/ISL/Indian")
    
    print("\n" + "="*70)
    print("NUMBER '1' vs LETTER 'G' - SIDE BY SIDE COMPARISON")
    print("="*70)
    print("\nKEY FINDINGS:")
    print("  - Feature distance: 3.74 (moderately similar)")
    print("  - Main difference: BODY POSE and FINGERTIP ANGLES")
    print("  - Letter G: Index+Thumb point SIDEWAYS (like pointing left)")
    print("  - Number 1: Index points STRAIGHT UP")
    print("\n" + "="*70)
    print("Press any key to see examples, 'q' to quit")
    print("="*70 + "\n")
    
    num_1_dir = dataset_root / "1"
    letter_g_dir = dataset_root / "G"
    
    if not num_1_dir.exists() or not letter_g_dir.exists():
        print("❌ Dataset directories not found")
        return
    
    # Get images
    num_1_images = list(num_1_dir.glob("*.jpg")) + list(num_1_dir.glob("*.png"))
    letter_g_images = list(letter_g_dir.glob("*.jpg")) + list(letter_g_dir.glob("*.png"))
    
    if not num_1_images or not letter_g_images:
        print("❌ No images found")
        return
    
    # Show 5 comparisons
    for i in range(5):
        # Pick random samples
        img_1_path = random.choice(num_1_images)
        img_g_path = random.choice(letter_g_images)
        
        # Load and resize
        img_1 = cv2.imread(str(img_1_path))
        img_g = cv2.imread(str(img_g_path))
        
        if img_1 is None or img_g is None:
            continue
        
        img_1 = cv2.resize(img_1, (400, 400))
        img_g = cv2.resize(img_g, (400, 400))
        
        # Add labels
        cv2.putText(img_1, "NUMBER '1'", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img_1, "Index UP", (10, 370),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(img_g, "LETTER 'G'", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img_g, "Index+Thumb SIDEWAYS", (10, 370),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Combine side by side
        combined = np.hstack([img_1, img_g])
        
        # Add title
        title_img = np.zeros((100, combined.shape[1], 3), dtype=np.uint8)
        title_text = f"Comparison {i+1}/5: Study the difference!"
        cv2.putText(title_img, title_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        instructions = "Key: '1' = vertical index | 'G' = sideways gesture"
        cv2.putText(title_img, instructions, (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        final_img = np.vstack([title_img, combined])
        
        # Show
        cv2.imshow('NUMBER 1 vs LETTER G - Find the Difference!', final_img)
        print(f"Showing comparison {i+1}/5...")
        print("  LEFT = NUMBER '1' (index up)")
        print("  RIGHT = LETTER 'G' (sideways gesture)")
        
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("SOLUTION TO GET NUMBER '1':")
    print("="*70)
    print("""
Based on the comparison, to get '1' instead of 'G':

1. ✓ Index finger must point STRAIGHT UP (not sideways)
2. ✓ Keep hand VERTICAL (not tilted/horizontal)
3. ✓ Thumb should be TUCKED to side of fist (not extended)
4. ✓ Other fingers TIGHTLY CLOSED in fist
5. ✓ Keep your BODY straight (not leaning)

Letter 'G' typically has:
✗ Hand oriented MORE HORIZONTAL
✗ Index finger + thumb both extended pointing SIDEWAYS
✗ Like making a "pointing left" gesture

Try This:
→ Make a fist
→ Point index finger STRAIGHT to the ceiling
→ Keep hand vertical (like raising hand in class)
→ Don't tilt hand sideways
""")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        show_1_vs_g_comparison()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
