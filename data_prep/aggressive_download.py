"""
Aggressive downloader - get MANY videos for FEW signs.
Focus: 3 signs with 50+ videos each for robust training.
"""

import os
import json
import subprocess
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import time

def download_video_youtube(url, output_path):
    """Download from YouTube with yt-dlp."""
    try:
        if 'youtube.com' not in url and 'youtu.be' not in url:
            return False
        
        cmd = f"yt-dlp -f 'best[height<=480]' -o '{output_path}' '{url}' --quiet --no-warnings --no-check-certificate"
        result = subprocess.run(cmd, shell=True, capture_output=True, timeout=45)
        
        if result.returncode == 0 and os.path.exists(output_path):
            # Check file size
            if os.path.getsize(output_path) > 10000:  # > 10KB
                return True
        
        # Clean up failed download
        if os.path.exists(output_path):
            os.remove(output_path)
        return False
    except:
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def main():
    json_path = Path('datasets/WLASL/start_kit/WLASL_v0.3.json')
    output_dir = Path('dataset/raw/')
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # AGGRESSIVE STRATEGY: Get 50+ videos for top 3 signs
    target_signs = ['book', 'drink', 'computer']
    target_per_sign = 100  # Try to get 100, we'll likely get 30-50
    
    print(f"\n{'='*70}")
    print("AGGRESSIVE DOWNLOADER - MAXIMUM DATA STRATEGY")
    print(f"{'='*70}")
    print(f"Goal: {len(target_signs)} signs × {target_per_sign} attempts = high-quality dataset")
    print(f"Signs: {', '.join(target_signs)}")
    print(f"{'='*70}\n")
    
    downloaded_metadata = []
    
    for sign in target_signs:
        gloss_data = next((g for g in data if g['gloss'] == sign), None)
        if not gloss_data:
            continue
        
        instances = gloss_data['instances']
        print(f"\n📥 Downloading '{sign}' (attempting {target_per_sign} videos)...")
        print(f"   Total available: {len(instances)} videos")
        
        success_count = 0
        attempt_count = 0
        
        progress_bar = tqdm(total=min(target_per_sign, len(instances)), desc=sign)
        
        for instance in instances:
            if success_count >= target_per_sign:
                break
            
            attempt_count += 1
            video_id = instance['video_id']
            url = instance['url']
            filename = f"{sign}_{video_id}.mp4"
            output_path = output_dir / filename
            
            # Skip if exists
            if output_path.exists() and os.path.getsize(output_path) > 10000:
                downloaded_metadata.append({
                    'video_id': video_id,
                    'gloss': sign,
                    'filename': filename,
                    'url': url
                })
                success_count += 1
                progress_bar.update(1)
                continue
            
            # Download (only YouTube)
            if download_video_youtube(url, str(output_path)):
                downloaded_metadata.append({
                    'video_id': video_id,
                    'gloss': sign,
                    'filename': filename,
                    'url': url
                })
                success_count += 1
                progress_bar.update(1)
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        progress_bar.close()
        print(f"  ✅ {success_count} videos for '{sign}' (attempted {attempt_count})")
    
    # Save metadata
    df = pd.DataFrame(downloaded_metadata)
    df.to_csv(output_dir / 'wlasl_downloaded_metadata.csv', index=False)
    
    print(f"\n{'='*70}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"Total videos: {len(df)}")
    print(f"\nDistribution:")
    print(df['gloss'].value_counts().to_string())
    print(f"\n{'='*70}\n")
    
    print("✅ Next step: Run preprocessing to extract keypoints")
    print("   python data_prep/preprocess_wlasl_enhanced.py")

if __name__ == "__main__":
    main()
