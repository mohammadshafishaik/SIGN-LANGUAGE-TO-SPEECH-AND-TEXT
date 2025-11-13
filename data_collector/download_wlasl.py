#!/usr/bin/env python3
"""
WLASL Video Downloader
Downloads videos from WLASL dataset (Word-Level American Sign Language)

Usage:
    python data_collector/download_wlasl.py --num_words 100  # Download top 100 words
    python data_collector/download_wlasl.py --num_words 300  # Download 300 words
    python data_collector/download_wlasl.py --num_words 2000 # Download all words
"""

import os
import json
import argparse
import subprocess
from tqdm import tqdm
from collections import defaultdict

# Configuration
WLASL_JSON_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
OUTPUT_DIR = "datasets/WLASL/start_kit"
VIDEO_DIR = os.path.join(OUTPUT_DIR, "raw_videos")
METADATA_FILE = os.path.join(OUTPUT_DIR, "WLASL_v0.3.json")

def download_metadata():
    """Download WLASL metadata JSON file"""
    print("\n📥 Downloading WLASL metadata...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if os.path.exists(METADATA_FILE):
        print(f"✅ Metadata already exists: {METADATA_FILE}")
        return
    
    # Download using curl
    cmd = f"curl -L -o '{METADATA_FILE}' '{WLASL_JSON_URL}'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Metadata downloaded: {METADATA_FILE}")
    else:
        print(f"❌ Failed to download metadata: {result.stderr}")
        raise Exception("Metadata download failed")

def load_metadata():
    """Load WLASL metadata JSON"""
    with open(METADATA_FILE, 'r') as f:
        return json.load(f)

def select_top_words(metadata, num_words):
    """
    Select top N words by number of video instances
    Returns: List of (gloss, instances) tuples
    """
    # Count instances per word
    word_counts = []
    for entry in metadata:
        gloss = entry['gloss']
        instances = entry['instances']
        word_counts.append((gloss, len(instances), instances))
    
    # Sort by instance count (most videos first)
    word_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Take top N
    selected = word_counts[:num_words]
    
    print(f"\n📊 Selected top {num_words} words:")
    print(f"   Most common: {selected[0][0]} ({selected[0][1]} videos)")
    print(f"   Least common: {selected[-1][0]} ({selected[-1][1]} videos)")
    total_videos = sum(count for _, count, _ in selected)
    print(f"   Total videos to download: {total_videos}")
    
    return selected

def check_ytdlp():
    """Check if yt-dlp is installed"""
    try:
        result = subprocess.run(['yt-dlp', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ yt-dlp version: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("\n❌ yt-dlp not found!")
    print("\n📥 Installing yt-dlp...")
    print("   Run: pip install yt-dlp")
    
    # Try to install automatically
    install_cmd = "pip install yt-dlp"
    result = subprocess.run(install_cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ yt-dlp installed successfully!")
        return True
    else:
        print(f"❌ Installation failed: {result.stderr}")
        print("\nPlease install manually:")
        print("  pip install yt-dlp")
        return False

def download_video(video_id, url, output_path):
    """
    Download a single video using yt-dlp
    Returns: True if successful, False otherwise
    """
    if os.path.exists(output_path):
        return True  # Already downloaded
    
    # yt-dlp command with format selection
    cmd = [
        'yt-dlp',
        '-f', 'best[height<=480]',  # Limit to 480p to save space
        '-o', output_path,
        '--quiet',
        '--no-warnings',
        url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False

def download_videos(selected_words):
    """Download all videos for selected words"""
    os.makedirs(VIDEO_DIR, exist_ok=True)
    
    # Collect all download tasks
    tasks = []
    for gloss, count, instances in selected_words:
        for instance in instances:
            video_id = instance['video_id']
            url = f"https://www.youtube.com/watch?v={video_id}"
            output_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
            tasks.append((video_id, url, output_path, gloss))
    
    print(f"\n🎬 Downloading {len(tasks)} videos...")
    print(f"   Output directory: {VIDEO_DIR}")
    print(f"   This may take a while...\n")
    
    # Download statistics
    stats = {
        'success': 0,
        'already_exist': 0,
        'failed': 0,
        'unavailable': 0
    }
    
    failed_videos = []
    
    # Download each video with progress bar
    with tqdm(total=len(tasks), desc="Downloading") as pbar:
        for video_id, url, output_path, gloss in tasks:
            if os.path.exists(output_path):
                stats['already_exist'] += 1
                pbar.update(1)
                continue
            
            success = download_video(video_id, url, output_path)
            
            if success:
                stats['success'] += 1
            else:
                stats['failed'] += 1
                failed_videos.append((video_id, gloss))
            
            pbar.update(1)
    
    # Print summary
    print("\n" + "="*70)
    print("📊 DOWNLOAD SUMMARY")
    print("="*70)
    print(f"✅ Successfully downloaded: {stats['success']}")
    print(f"📁 Already existed: {stats['already_exist']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"📊 Total videos available: {stats['success'] + stats['already_exist']}")
    print("="*70)
    
    if failed_videos and len(failed_videos) <= 20:
        print("\n⚠️  Failed downloads (may be deleted/private):")
        for vid_id, gloss in failed_videos[:10]:
            print(f"   - {gloss}: {vid_id}")
        if len(failed_videos) > 10:
            print(f"   ... and {len(failed_videos) - 10} more")
    
    return stats

def create_filtered_metadata(selected_words, output_file):
    """Create a filtered metadata JSON with only downloaded words"""
    filtered_data = []
    
    for gloss, count, instances in selected_words:
        # Only include instances where video file exists
        valid_instances = []
        for instance in instances:
            video_id = instance['video_id']
            video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
            if os.path.exists(video_path):
                valid_instances.append(instance)
        
        if valid_instances:
            filtered_data.append({
                'gloss': gloss,
                'instances': valid_instances
            })
    
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"\n✅ Filtered metadata saved: {output_file}")
    print(f"   Words with videos: {len(filtered_data)}")

def main():
    parser = argparse.ArgumentParser(
        description='Download WLASL dataset videos from YouTube'
    )
    parser.add_argument(
        '--num_words',
        type=int,
        default=100,
        help='Number of top words to download (default: 100)'
    )
    args = parser.parse_args()
    
    print("="*70)
    print("🎬 WLASL Video Downloader")
    print("="*70)
    print(f"Target: Top {args.num_words} ASL words")
    print()
    
    # Step 1: Check yt-dlp
    if not check_ytdlp():
        return
    
    # Step 2: Download metadata
    download_metadata()
    
    # Step 3: Load and select words
    metadata = load_metadata()
    print(f"\n📚 Total words in WLASL: {len(metadata)}")
    
    selected_words = select_top_words(metadata, args.num_words)
    
    # Step 4: Download videos
    stats = download_videos(selected_words)
    
    # Step 5: Create filtered metadata
    filtered_file = os.path.join(OUTPUT_DIR, f"WLASL_top_{args.num_words}.json")
    create_filtered_metadata(selected_words, filtered_file)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ DOWNLOAD COMPLETE!")
    print("="*70)
    print(f"📁 Videos saved to: {VIDEO_DIR}")
    print(f"📄 Metadata saved to: {filtered_file}")
    print("\n🎯 NEXT STEPS:")
    print(f"   1. Preprocess videos: python data_prep/preprocess_wlasl.py")
    print(f"   2. Train model: python models/train_wlasl.py")
    print(f"   3. Test real-time: python inference/realtime_wlasl.py")
    print("="*70)

if __name__ == "__main__":
    main()
