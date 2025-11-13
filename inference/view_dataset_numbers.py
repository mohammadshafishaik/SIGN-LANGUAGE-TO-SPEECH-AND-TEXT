"""
View ISL Dataset Images - See exactly what hand poses the model learned
"""

import cv2
import numpy as np
from pathlib import Path
import random

def show_number_examples():
    """Show example images from the ISL dataset for each number"""
    
    dataset_root = Path("datasets/ISL/Indian")
    
    print("\n" + "="*70)
    print("ISL NUMBER DATASET IMAGE VIEWER")
    print("="*70)
    print("Press any key to see next number, 'q' to quit")
    print("="*70 + "\n")
    
    for num in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
        num_dir = dataset_root / num
        if not num_dir.exists():
            print(f"⚠️  Number {num} directory not found")
            continue
        
        # Get all images
        images = list(num_dir.glob("*.jpg")) + list(num_dir.glob("*.png"))
        if not images:
            print(f"⚠️  No images found for number {num}")
            continue
        
        # Show 6 random examples
        samples = random.sample(images, min(6, len(images)))
        
        # Create a grid of images
        grid_images = []
        for img_path in samples:
            img = cv2.imread(str(img_path))
            if img is not None:
                # Resize to standard size
                img = cv2.resize(img, (300, 300))
                grid_images.append(img)
        
        if not grid_images:
            continue
        
        # Arrange in 2x3 grid
        if len(grid_images) >= 6:
            row1 = np.hstack(grid_images[0:3])
            row2 = np.hstack(grid_images[3:6])
            grid = np.vstack([row1, row2])
        elif len(grid_images) >= 3:
            row1 = np.hstack(grid_images[0:3])
            grid = row1
        else:
            grid = np.hstack(grid_images)
        
        # Add title
        title_img = np.zeros((80, grid.shape[1], 3), dtype=np.uint8)
        title_text = f"NUMBER {num} - ISL Dataset Examples"
        cv2.putText(title_img, title_text, 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.5, (0, 255, 255), 3)
        
        instruction = "Study the hand pose! Press any key for next, 'q' to quit"
        cv2.putText(title_img, instruction,
                   (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1)
        
        final_img = np.vstack([title_img, grid])
        
        # Show the grid
        cv2.imshow('ISL Numbers - Match These Poses!', final_img)
        print(f"Showing NUMBER {num} examples...")
        
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("\n✓ Done! Now try matching these poses in the webcam.")

if __name__ == "__main__":
    show_number_examples()
