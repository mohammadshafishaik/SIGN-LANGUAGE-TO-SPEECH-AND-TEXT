import cv2
import os
import pandas as pd
import argparse
import time
from datetime import datetime

class DataCollector:
    """
    A class to handle the collection of sign language video data.

    This class provides functionalities to:
    - Record video clips for specified sign language phrases.
    - Display a real-time countdown and recording progress.
    - Save the recorded videos to a designated directory.
    - Update a metadata file with information about each recording.
    """
    def __init__(self, output_dir, metadata_path):
        """
        Initializes the DataCollector.

        Args:
            output_dir (str): The directory where raw video files will be saved.
            metadata_path (str): The path to the CSV file for storing metadata.
        """
        self.output_dir = output_dir
        self.metadata_path = metadata_path
        os.makedirs(self.output_dir, exist_ok=True)
        self.init_metadata_file()

    def init_metadata_file(self):
        """Initializes the metadata CSV file if it doesn't exist."""
        if not os.path.exists(self.metadata_path):
            df = pd.DataFrame(columns=["video_id", "phrase"])
            df.to_csv(self.metadata_path, index=False)

    def record_phrase(self, phrase, duration=3, countdown=3, cam_index=0):
        """
        Records a video for a given phrase.

        Args:
            phrase (str): The sign language phrase to be recorded.
            duration (int): The duration of the recording in seconds.
            countdown (int): The countdown period before recording starts.
            cam_index (int): The index of the camera to use.
        """
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print(f"Error: Cannot open camera with index {cam_index}.")
            return

        # Video writer setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_id = f"{phrase.replace(' ', '_')}_{timestamp}"
        video_filename = f"{video_id}.mp4"
        video_path = os.path.join(self.output_dir, video_filename)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30 # Default fps if not available
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        # --- Countdown Phase ---
        for i in range(countdown, 0, -1):
            start_time = time.time()
            while time.time() - start_time < 1:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Display countdown text
                text = f"Get Ready: {i}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
                text_x = (width - text_size[0]) // 2
                text_y = (height + text_size[1]) // 2
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.imshow(f"Recording for: {phrase}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

        # --- Recording Phase ---
        start_recording_time = time.time()
        print(f"Recording '{phrase}' for {duration} seconds...")
        
        while time.time() - start_recording_time < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            
            # Display recording indicator
            elapsed_time = time.time() - start_recording_time
            progress = elapsed_time / duration
            cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1) # Red dot for recording
            # Progress bar
            cv2.rectangle(frame, (0, height - 10), (int(width * progress), height), (0, 255, 0), -1)

            cv2.imshow(f"Recording for: {phrase}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print(f"Finished recording for '{phrase}'.")
        
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Update metadata
        self.update_metadata(video_id, phrase)
        print(f"Saved video to {video_path} and updated metadata.")

    def update_metadata(self, video_id, phrase):
        """
        Appends a new record to the metadata CSV file.

        Args:
            video_id (str): The unique identifier for the video.
            phrase (str): The phrase associated with the video.
        """
        try:
            df = pd.read_csv(self.metadata_path)
        except FileNotFoundError:
            self.init_metadata_file()
            df = pd.read_csv(self.metadata_path)
            
        new_row = pd.DataFrame([{"video_id": video_id, "phrase": phrase}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(self.metadata_path, index=False)

def main():
    """Main function to run the data collection script."""
    parser = argparse.ArgumentParser(description="Record sign language phrases.")
    parser.add_argument("--phrase", type=str, required=True, help="The phrase to record (e.g., 'hello', 'thank_you').")
    parser.add_argument("--duration", type=int, default=3, help="Recording duration in seconds.")
    parser.add_argument("--countdown", type=int, default=3, help="Countdown before recording starts.")
    parser.add_argument("--output_dir", type=str, default="dataset/raw", help="Directory to save videos.")
    parser.add_argument("--metadata", type=str, default="dataset/metadata.csv", help="Path to metadata CSV file.")
    parser.add_argument("--cam_index", type=int, default=0, help="Index of the camera to use.")

    args = parser.parse_args()

    collector = DataCollector(args.output_dir, args.metadata)
    
    print("\n--- Sign Language Data Collector ---")
    print(f"Phrase to record: {args.phrase}")
    print(f"Press 'q' at any time to quit.")
    
    collector.record_phrase(args.phrase, args.duration, args.countdown, args.cam_index)
    print("\nCollection complete.")

if __name__ == "__main__":
    # Example usage from the root of the project:
    # python data_collector/collect.py --phrase "hello"
    # python data_collector/collect.py --phrase "thank you" --duration 5
    main()
