"""
Create a visual reference sheet showing all ISL gestures (A-Z, 1-9)
This will be your complete guide for hand poses!
"""

import cv2
import numpy as np
from pathlib import Path
import random

def create_isl_reference_sheet():
    """Create a grid showing all 35 ISL signs"""
    
    dataset_root = Path("datasets/ISL/Indian")
    
    print("\n" + "="*70)
    print("CREATING ISL REFERENCE SHEET")
    print("="*70)
    print("Generating visual guide for all 35 signs...")
    print("="*70 + "\n")
    
    # All classes in order
    all_classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
                   'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                   'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                   'U', 'V', 'W', 'X', 'Y', 'Z']
    
    # Grid layout: 7 columns x 5 rows = 35 signs
    grid_cols = 7
    grid_rows = 5
    cell_size = 200  # Each image 200x200
    label_height = 40  # Space for label
    
    # Create blank canvas
    canvas_width = grid_cols * cell_size
    canvas_height = grid_rows * (cell_size + label_height)
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # White background
    
    print("Grid layout: 7 columns x 5 rows")
    print(f"Canvas size: {canvas_width} x {canvas_height}\n")
    
    # Process each class
    for idx, class_name in enumerate(all_classes):
        row = idx // grid_cols
        col = idx % grid_cols
        
        class_dir = dataset_root / class_name
        
        if not class_dir.exists():
            print(f"⚠️  Class {class_name} not found, skipping...")
            continue
        
        # Get a random sample image
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        if not images:
            print(f"⚠️  No images for class {class_name}")
            continue
        
        sample_img_path = random.choice(images)
        img = cv2.imread(str(sample_img_path))
        
        if img is None:
            continue
        
        # Resize to cell size
        img = cv2.resize(img, (cell_size, cell_size))
        
        # Calculate position
        y_start = row * (cell_size + label_height)
        x_start = col * cell_size
        
        # Place image on canvas
        canvas[y_start:y_start+cell_size, x_start:x_start+cell_size] = img
        
        # Add label below image
        label_y = y_start + cell_size + 30
        label_x = x_start + cell_size // 2
        
        # Determine if number or letter
        is_number = class_name in ['1','2','3','4','5','6','7','8','9']
        label_color = (0, 0, 255) if is_number else (0, 128, 0)  # Red for numbers, Green for letters
        label_prefix = "NUM" if is_number else "LTR"
        
        # Draw label
        label_text = f"{label_prefix}: {class_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Get text size for centering
        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        text_x = label_x - text_width // 2
        
        cv2.putText(canvas, label_text, (text_x, label_y), 
                   font, font_scale, label_color, thickness)
        
        print(f"✓ Added {class_name} at position ({row}, {col})")
    
    # Add title at the top
    title_height = 80
    final_canvas = np.ones((canvas_height + title_height, canvas_width, 3), dtype=np.uint8) * 255
    final_canvas[title_height:, :] = canvas
    
    # Title text
    title = "ISL (Indian Sign Language) Reference - All 35 Signs"
    subtitle = "Numbers 1-9 (RED) | Letters A-Z (GREEN)"
    
    cv2.putText(final_canvas, title, (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(final_canvas, subtitle, (20, 65), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
    
    # Save the reference sheet
    output_path = Path("docs/ISL_Reference_Sheet.jpg")
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), final_canvas)
    
    print("\n" + "="*70)
    print("✅ REFERENCE SHEET CREATED!")
    print("="*70)
    print(f"Saved to: {output_path}")
    print(f"Size: {final_canvas.shape[1]} x {final_canvas.shape[0]} pixels")
    print("\nShowing image (press any key to close)...")
    print("="*70 + "\n")
    
    # Display the reference sheet
    # Resize for display if too large
    display_width = 1400
    scale = display_width / final_canvas.shape[1]
    display_height = int(final_canvas.shape[0] * scale)
    display_canvas = cv2.resize(final_canvas, (display_width, display_height))
    
    cv2.imshow('ISL Complete Reference Sheet - All 35 Signs', display_canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("✓ Done! Reference sheet saved and displayed.")
    print(f"\nYou can find the image at: {output_path.absolute()}")
    print("\nUse this as your guide when showing signs to the webcam!")

if __name__ == "__main__":
    try:
        create_isl_reference_sheet()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
