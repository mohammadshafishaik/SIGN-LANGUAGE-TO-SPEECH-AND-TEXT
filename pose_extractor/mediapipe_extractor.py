import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
from tqdm import tqdm

class MediaPipeExtractor:
    """
    Extracts 3D pose, hand, and face landmarks from video using MediaPipe Holistic,
    normalizes them, computes temporal features (velocity, acceleration), and
    saves the data.
    """
    def __init__(self, holistic=None):
        """
        Initializes the MediaPipeExtractor.
        Args:
            holistic: An existing MediaPipe Holistic instance. If None, a new one
                      is created with default parameters.
        """
        self.holistic = holistic if holistic else mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Total number of landmarks: 33 pose + 21 left hand + 21 right hand = 75
        self.num_landmarks = 33 + 21 + 21

    def close(self):
        """Releases the MediaPipe Holistic resources."""
        self.holistic.close()

    def extract_landmarks(self, frame):
        """
        Processes a single video frame to extract landmarks.
        Args:
            frame: A BGR image from OpenCV.
        Returns:
            A MediaPipe results object containing the detected landmarks.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True
        return results

    def get_landmarks_from_results(self, results):
        """
        Extracts and combines pose, hand, and face landmarks into a single NumPy array.
        If a landmark type is not detected, it is filled with NaNs.
        Args:
            results: MediaPipe results object.
        Returns:
            A NumPy array of shape (75, 3) with x, y, z coordinates.
        """
        landmarks = []
        landmark_map = {
            'pose_landmarks': (results.pose_landmarks, 33),
            'left_hand_landmarks': (results.left_hand_landmarks, 21),
            'right_hand_landmarks': (results.right_hand_landmarks, 21)
        }

        for key, (landmark_set, num_points) in landmark_map.items():
            if landmark_set:
                for lm in landmark_set.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([[np.nan] * 3] * num_points)
        
        return np.array(landmarks)

    def normalize_landmarks(self, landmarks):
        """
        Normalizes landmarks based on torso position and size.
        The origin is set to the center of the hips, and the scale is normalized
        by the distance between the shoulders.
        Args:
            landmarks: A NumPy array of shape (75, 3).
        Returns:
            A tuple containing:
            - normalized_landmarks: The transformed landmarks.
            - torso_center: The calculated center of the torso.
            - torso_size: The calculated size of the torso.
        """
        if landmarks.shape[0] == 0:
            return landmarks, None, None

        pose_landmarks = landmarks[:33]
        
        left_hip = pose_landmarks[23]
        right_hip = pose_landmarks[24]

        if np.isnan(left_hip).any() or np.isnan(right_hip).any():
            return landmarks, None, None  # Cannot normalize if hips are not visible

        torso_center = (left_hip + right_hip) / 2.0

        shoulder_l = pose_landmarks[11]
        shoulder_r = pose_landmarks[12]
        
        if np.isnan(shoulder_l).any() or np.isnan(shoulder_r).any():
            torso_size = 1.0  # Fallback if shoulders are not visible
        else:
            torso_size = np.linalg.norm(shoulder_l - shoulder_r)
            if torso_size < 1e-6:  # Avoid division by zero or near-zero
                torso_size = 1.0

        normalized_landmarks = (landmarks - torso_center) / torso_size
        
        # Replace any remaining NaNs with 0 after normalization
        normalized_landmarks[np.isnan(normalized_landmarks)] = 0
        
        return normalized_landmarks, torso_center, torso_size

    def process_video(self, video_path, output_path):
        """
        Processes an entire video file to extract, normalize, and compute
        temporal features for all frames.
        Args:
            video_path: Path to the input video file.
            output_path: Path to save the output .npz file.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            print(f"Warning: Video {video_path} has zero frames.")
            cap.release()
            return

        all_landmarks = []
        
        for _ in tqdm(range(frame_count), desc=f"Processing {os.path.basename(video_path)}"):
            ret, frame = cap.read()
            if not ret:
                break

            results = self.extract_landmarks(frame)
            landmarks = self.get_landmarks_from_results(results)
            normalized_landmarks, _, _ = self.normalize_landmarks(landmarks)
            all_landmarks.append(normalized_landmarks)

        cap.release()

        landmarks_arr = np.array(all_landmarks)
        
        # Compute velocity (difference between frames)
        # The first frame has no velocity, so we pad with zeros at the beginning.
        velocity = np.diff(landmarks_arr, axis=0)
        velocity = np.concatenate([np.zeros((1, self.num_landmarks, 3)), velocity], axis=0)

        # Compute acceleration (difference of velocity)
        # The first two frames have no or undefined acceleration.
        acceleration = np.diff(velocity, axis=0)
        acceleration = np.concatenate([np.zeros((1, self.num_landmarks, 3)), acceleration], axis=0)

        np.savez_compressed(
            output_path,
            landmarks=landmarks_arr,
            velocity=velocity,
            acceleration=acceleration
        )
        print(f"✅ Saved keypoints to {output_path}")

def main():
    """Main function to run the script from the command line."""
    parser = argparse.ArgumentParser(description="Extract 3D keypoints from video using MediaPipe Holistic.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing raw video files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .npz keypoint files.")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    extractor = MediaPipeExtractor()

    video_files = [f for f in os.listdir(args.video_dir) if f.endswith(('.mp4', '.mov', '.avi'))]
    if not video_files:
        print(f"No video files found in {args.video_dir}")
        return

    for video_file in video_files:
        video_path = os.path.join(args.video_dir, video_file)
        output_filename = os.path.splitext(video_file)[0] + ".npz"
        output_path = os.path.join(args.output_dir, output_filename)
        
        extractor.process_video(video_path, output_path)

    extractor.close()
    print("\nExtraction complete for all videos.")

if __name__ == "__main__":
    # Example usage from the root of the project:
    # python pose_extractor/mediapipe_extractor.py --video_dir dataset/raw --output_dir dataset/keypoints
    main()
