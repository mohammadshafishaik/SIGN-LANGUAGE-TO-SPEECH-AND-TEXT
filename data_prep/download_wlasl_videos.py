"""
Download WLASL videos for training.
Downloads multiple videos per sign to create a large, diverse dataset.

Usage:
    python data_prep/download_wlasl_videos.py --num_signs 20 --videos_per_sign 10
"""

import os
import json
import argparse
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def download_video(url, output_path):
    """Download a single video using yt-dlp or curl."""
    try:
        # Try yt-dlp first for YouTube videos
        if 'youtube.com' in url or 'youtu.be' in url:
            cmd = f"yt-dlp -f 'best[height<=480]' -o '{output_path}' '{url}'"
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=60)
            return result.returncode == 0
        else:
            # Use curl for direct video links
            cmd = f"curl -L -o '{output_path}' '{url}' --max-time 60"
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=60)
            return result.returncode == 0 and os.path.exists(output_path)
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main(args):
    print("="*60)
    print("WLASL VIDEO DOWNLOADER")
    print("="*60)
    
    # Paths
    json_path = Path(args.json_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load WLASL JSON
    print(f"\nLoading WLASL metadata from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total glosses in dataset: {len(data)}")
    
    # Select signs to download
    if args.specific_signs:
        selected_glosses = [g for g in data if g['gloss'] in args.specific_signs]
    else:
        # Select top N signs by number of instances
        glosses_sorted = sorted(data, key=lambda x: len(x['instances']), reverse=True)
        selected_glosses = glosses_sorted[:args.num_signs]
    
    print(f"\nSelected {len(selected_glosses)} signs:")
    for g in selected_glosses:
        print(f"  - {g['gloss']}: {len(g['instances'])} videos available")
    
    # Download videos
    downloaded_metadata = []
    total_downloaded = 0
    total_failed = 0
    
    for gloss_data in selected_glosses:
        gloss = gloss_data['gloss']
        instances = gloss_data['instances'][:args.videos_per_sign]
        
        print(f"\n📥 Downloading '{gloss}' ({len(instances)} videos)...")
        
        for idx, instance in enumerate(tqdm(instances, desc=gloss)):
            video_id = instance['video_id']
            url = instance['url']
            
            # Create filename
            filename = f"{gloss}_{video_id}.mp4"
            output_path = output_dir / filename
            
            # Skip if already downloaded
            if output_path.exists():
                downloaded_metadata.append({
                    'video_id': video_id,
                    'gloss': gloss,
                    'filename': filename,
                    'url': url,
                    'split': instance.get('split', 'train')
                })
                continue
            
            # Download
            success = download_video(url, str(output_path))
            
            if success and output_path.exists():
                downloaded_metadata.append({
                    'video_id': video_id,
                    'gloss': gloss,
                    'filename': filename,
                    'url': url,
                    'split': instance.get('split', 'train')
                })
                total_downloaded += 1
            else:
                total_failed += 1
                if output_path.exists():
                    output_path.unlink()  # Remove partial download
    
    # Save metadata
    df = pd.DataFrame(downloaded_metadata)
    metadata_path = output_dir / 'wlasl_downloaded_metadata.csv'
    df.to_csv(metadata_path, index=False)
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"✅ Successfully downloaded: {total_downloaded} videos")
    print(f"❌ Failed: {total_failed} videos")
    print(f"📊 Total unique signs: {df['gloss'].nunique()}")
    print(f"📁 Videos saved to: {output_dir}")
    print(f"📄 Metadata saved to: {metadata_path}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download WLASL videos for training")
    parser.add_argument('--json_path', type=str, 
                        default='datasets/WLASL/start_kit/WLASL_v0.3.json',
                        help='Path to WLASL JSON file')
    parser.add_argument('--output_dir', type=str,
                        default='dataset/raw/',
                        help='Output directory for videos')
    parser.add_argument('--num_signs', type=int, default=20,
                        help='Number of different signs to download')
    parser.add_argument('--videos_per_sign', type=int, default=10,
                        help='Number of videos to download per sign')
    parser.add_argument('--specific_signs', type=str, nargs='+',
                        help='Specific sign names to download (overrides num_signs)')
    
    args = parser.parse_args()
    main(args)
