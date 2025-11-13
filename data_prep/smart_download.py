"""
Smart WLASL downloader - focuses on getting many samples for FEW classes.
Strategy: Download 50+ videos for 5-10 high-quality signs.
"""

import os
import json
import subprocess
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def download_video(url, output_path):
    """Download with better error handling."""
    try:
        if 'youtube.com' in url or 'youtu.be' in url:
            cmd = f"yt-dlp -f 'best[height<=480]' -o '{output_path}' '{url}' --quiet --no-warnings"
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=30)
            return result.returncode == 0 and os.path.exists(output_path)
        else:
            # Skip non-YouTube URLs (they timeout frequently)
            return False
    except Exception:
        return False

def main():
    json_path = Path('datasets/WLASL/start_kit/WLASL_v0.3.json')
    output_dir = Path('dataset/raw/')
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Target: 5 signs with 40 videos each = 200 videos
    # Select signs with most YouTube videos
    target_signs = ['book', 'drink', 'computer', 'go', 'all']
    target_per_sign = 40
    
    print(f"\n{'='*70}")
    print("SMART DOWNLOADER - HIGH SAMPLE STRATEGY")
    print(f"{'='*70}")
    print(f"Target: {len(target_signs)} signs × {target_per_sign} videos = {len(target_signs)*target_per_sign} total")
    print(f"Signs: {', '.join(target_signs)}")
    print(f"{'='*70}\n")
    
    downloaded_metadata = []
    
    for sign in target_signs:
        gloss_data = next((g for g in data if g['gloss'] == sign), None)
        if not gloss_data:
            continue
        
        instances = gloss_data['instances'][:target_per_sign]
        print(f"\n📥 Downloading '{sign}' (target: {target_per_sign} videos)...")
        
        success_count = 0
        for instance in tqdm(instances, desc=sign):
            if success_count >= target_per_sign:
                break
            
            video_id = instance['video_id']
            url = instance['url']
            filename = f"{sign}_{video_id}.mp4"
            output_path = output_dir / filename
            
            # Skip if exists
            if output_path.exists():
                downloaded_metadata.append({
                    'video_id': video_id,
                    'gloss': sign,
                    'filename': filename,
                    'url': url
                })
                success_count += 1
                continue
            
            # Download
            if download_video(url, str(output_path)):
                downloaded_metadata.append({
                    'video_id': video_id,
                    'gloss': sign,
                    'filename': filename,
                    'url': url
                })
                success_count += 1
        
        print(f"  ✅ {success_count} videos for '{sign}'")
    
    # Save metadata
    df = pd.DataFrame(downloaded_metadata)
    df.to_csv(output_dir / 'wlasl_downloaded_metadata.csv', index=False)
    
    print(f"\n{'='*70}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"Total videos: {len(df)}")
    print(f"Distribution:")
    print(df['gloss'].value_counts().to_string())
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
