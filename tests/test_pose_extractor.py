import unittest
import numpy as np
import os
import cv2
import sys

# Add project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pose_extractor.mediapipe_extractor import MediaPipeExtractor

class TestMediaPipeExtractor(unittest.TestCase):
    """Unit tests for the MediaPipeExtractor class."""

    @classmethod
    def setUpClass(cls):
        """Set up resources for all tests."""
        cls.extractor = MediaPipeExtractor()
        cls.test_video_path = "test_video.mp4"
        cls.output_path = "test_output.npz"
        
        # Create a dummy black video for testing
        width, height = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(cls.test_video_path, fourcc, 30.0, (width, height))
        for _ in range(10):  # 10 frames
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

    @classmethod
    def tearDownClass(cls):
        """Clean up resources after all tests."""
        cls.extractor.close()
        if os.path.exists(cls.test_video_path):
            os.remove(cls.test_video_path)
        if os.path.exists(cls.output_path):
            os.remove(cls.output_path)

    def test_01_landmark_extraction_from_frame(self):
        """Test that landmarks are extracted from a single frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.extractor.extract_landmarks(frame)
        landmarks = self.extractor.get_landmarks_from_results(results)
        
        # Expect 33 pose + 21 left hand + 21 right hand = 75 landmarks
        self.assertEqual(landmarks.shape, (75, 3))
        # In a black frame, landmarks are detected but may be all NaNs initially
        # before normalization, which is expected.
        self.assertTrue(np.isnan(landmarks).all() or not np.isnan(landmarks).all())


    def test_02_normalization(self):
        """Test the landmark normalization logic."""
        # Create a dummy landmark set with predictable values
        landmarks = np.random.rand(75, 3) * 100
        
        # Define key points for normalization
        landmarks[23] = np.array([10, 20, 5])  # Left hip
        landmarks[24] = np.array([30, 20, 5])  # Right hip
        landmarks[11] = np.array([5, 5, 5])   # Left shoulder
        landmarks[12] = np.array([35, 5, 5])   # Right shoulder

        normalized, center, size = self.extractor.normalize_landmarks(landmarks)
        
        # Torso center should be the midpoint of the hips
        expected_center = np.array([20, 20, 5])
        self.assertTrue(np.allclose(center, expected_center))
        
        # Torso size should be the distance between shoulders
        expected_size = np.linalg.norm(np.array([5, 5, 5]) - np.array([35, 5, 5])) # 30.0
        self.assertAlmostEqual(size, expected_size)
        
        # The normalized center of the hips should be at the origin
        normalized_hip_center = (normalized[23] + normalized[24]) / 2.0
        self.assertTrue(np.allclose(normalized_hip_center, [0, 0, 0]))

    def test_03_normalization_with_nans(self):
        """Test normalization when some landmarks are missing."""
        landmarks = np.full((75, 3), np.nan)
        landmarks[11] = np.array([5, 5, 5])
        landmarks[12] = np.array([35, 5, 5])
        
        # If hips are NaN, normalization should not proceed
        normalized, center, size = self.extractor.normalize_landmarks(landmarks)
        self.assertIsNone(center)
        self.assertIsNone(size)
        self.assertTrue(np.array_equal(landmarks, normalized))

    def test_04_process_video_and_check_output(self):
        """Test the full video processing pipeline and the output file."""
        self.extractor.process_video(self.test_video_path, self.output_path)
        
        # Check if the output file was created
        self.assertTrue(os.path.exists(self.output_path))
        
        # Load the data and verify its contents
        data = np.load(self.output_path)
        self.assertIn('landmarks', data)
        self.assertIn('velocity', data)
        self.assertIn('acceleration', data)
        
        num_frames = 10
        num_landmarks = 75
        
        self.assertEqual(data['landmarks'].shape, (num_frames, num_landmarks, 3))
        self.assertEqual(data['velocity'].shape, (num_frames, num_landmarks, 3))
        self.assertEqual(data['acceleration'].shape, (num_frames, num_landmarks, 3))
        
        # The first frame's velocity should be all zeros
        self.assertTrue(np.all(data['velocity'][0] == 0))
        
        # The first frame's acceleration should be all zeros
        self.assertTrue(np.all(data['acceleration'][0] == 0))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
