"""
Create separate reference sheets for Numbers and Letters with multiple examples
"""

import cv2
import numpy as np
from pathlib import Path
import random

def create_detailed_reference(classes, title, output_name, label_color):
    """Create a detailed reference showing multiple examples per sign"""
    
    dataset_root = Path("datasets/ISL/Indian")
    
    samples_per_class = 3  # Show 3 examples per sign
    cell_size = 180
    label_height = 35
    spacing = 10
    
    # Calculate grid
    num_classes = len(classes)
    grid_cols = 5  # 5 columns
    grid_rows = (num_classes + grid_cols - 1) // grid_cols
    
    # Each row shows: 3 sample images + label
    row_height = cell_size + label_height + spacing
    col_width = (cell_size * samples_per_class) + spacing * 2
    
    canvas_width = grid_cols * col_width + spacing * (grid_cols + 1)
    canvas_height = grid_rows * row_height + spacing * (grid_rows + 1) + 100  # +100 for title
    
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 250
    
    print(f"\n{'='*70}")
    print(f"Creating {title}")
    print(f"{'='*70}")
    print(f"Grid: {grid_cols} cols x {grid_rows} rows")
    print(f"Canvas: {canvas_width} x {canvas_height}\n")
    
    for idx, class_name in enumerate(classes):
        row = idx // grid_cols
        col = idx % grid_cols
        
        class_dir = dataset_root / class_name
        
        if not class_dir.exists():
            continue
        
        # Get sample images
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        if len(images) < samples_per_class:
            samples = images
        else:
            samples = random.sample(images, samples_per_class)
        
        # Calculate position
        y_pos = spacing + (row * row_height) + 100  # +100 for title space
        x_pos = spacing + (col * col_width)
        
        # Place sample images horizontally
        for i, img_path in enumerate(samples):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img = cv2.resize(img, (cell_size, cell_size))
            
            x_img = x_pos + (i * cell_size)
            y_img = y_pos
            
            # Add border around image
            cv2.rectangle(canvas, (x_img-2, y_img-2), 
                         (x_img+cell_size+2, y_img+cell_size+2), 
                         (200, 200, 200), 2)
            
            canvas[y_img:y_img+cell_size, x_img:x_img+cell_size] = img
        
        # Add label below images (centered)
        label_y = y_pos + cell_size + 25
        label_x = x_pos + (samples_per_class * cell_size) // 2
        
        label_text = class_name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        
        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        text_x = label_x - text_width // 2
        
        # Background for label
        cv2.rectangle(canvas, (text_x-10, label_y-text_height-5),
                     (text_x+text_width+10, label_y+5),
                     (255, 255, 255), -1)
        
        cv2.putText(canvas, label_text, (text_x, label_y), 
                   font, font_scale, label_color, thickness)
        
        print(f"✓ {class_name}: Added {len(samples)} examples")
    
    # Add title
    cv2.rectangle(canvas, (0, 0), (canvas_width, 90), (255, 255, 255), -1)
    cv2.putText(canvas, title, (30, 55), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 4)
    
    # Save
    output_path = Path(f"docs/{output_name}")
    cv2.imwrite(str(output_path), canvas)
    
    print(f"\n✅ Saved: {output_path}")
    print(f"   Size: {canvas_width} x {canvas_height}\n")
    
    return canvas, output_path

def main():
    print("\n" + "="*70)
    print("CREATING ISL REFERENCE SHEETS")
    print("="*70)
    
    # Create Numbers reference
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    numbers_canvas, numbers_path = create_detailed_reference(
        numbers, 
        "ISL NUMBERS (1-9) - Reference Guide",
        "ISL_Numbers_Reference.jpg",
        (0, 0, 200)  # Red
    )
    
    # Create Letters reference (split into 2 sheets if needed)
    letters_part1 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
    letters_part2 = ['N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    letters1_canvas, letters1_path = create_detailed_reference(
        letters_part1,
        "ISL LETTERS (A-M) - Reference Guide",
        "ISL_Letters_A-M_Reference.jpg",
        (0, 128, 0)  # Green
    )
    
    letters2_canvas, letters2_path = create_detailed_reference(
        letters_part2,
        "ISL LETTERS (N-Z) - Reference Guide", 
        "ISL_Letters_N-Z_Reference.jpg",
        (0, 128, 0)  # Green
    )
    
    print("="*70)
    print("✅ ALL REFERENCE SHEETS CREATED!")
    print("="*70)
    print("\nDisplaying reference sheets (press any key to view next)...\n")
    
    # Display all sheets
    sheets = [
        (numbers_canvas, "Numbers 1-9", numbers_path),
        (letters1_canvas, "Letters A-M", letters1_path),
        (letters2_canvas, "Letters N-Z", letters2_path)
    ]
    
    for canvas, name, path in sheets:
        # Resize for display
        display_width = 1400
        scale = display_width / canvas.shape[1] if canvas.shape[1] > display_width else 1.0
        if scale < 1.0:
            display_height = int(canvas.shape[0] * scale)
            display = cv2.resize(canvas, (display_width, display_height))
        else:
            display = canvas
        
        print(f"Showing: {name}")
        print(f"  File: {path.absolute()}\n")
        
        cv2.imshow(f'ISL Reference - {name}', display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("✓ Complete! All reference sheets saved to docs/ folder")
    print("="*70)
    print("\nReference Files Created:")
    print(f"  1. Numbers:    {numbers_path}")
    print(f"  2. Letters A-M: {letters1_path}")
    print(f"  3. Letters N-Z: {letters2_path}")
    print("\nUse these as your guide when practicing ISL signs!")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
