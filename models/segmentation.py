import numpy as np

class SegmentationModule:
    """
    A module to detect the start and end of sign phrases in a continuous stream
    of keypoints using motion energy and model confidence.
    """
    def __init__(self, motion_threshold=0.5, min_sign_duration=10, max_silence_duration=15, confidence_threshold=0.7):
        """
        Initializes the segmentation module.

        Args:
            motion_threshold (float): The threshold for motion energy to consider a frame as active.
            min_sign_duration (int): The minimum number of consecutive active frames to be considered a sign.
            max_silence_duration (int): The maximum number of consecutive inactive frames before a sign is considered ended.
            confidence_threshold (float): The minimum model prediction confidence to confirm a sign.
        """
        self.motion_threshold = motion_threshold
        self.min_sign_duration = min_sign_duration
        self.max_silence_duration = max_silence_duration
        self.confidence_threshold = confidence_threshold
        
        self.is_signing = False
        self.sign_buffer = []
        self.silent_frames_count = 0
        self.active_frames_count = 0

    def calculate_motion_energy(self, velocity_data):
        """
        Calculates the motion energy for a single frame.
        This is a simple measure of the magnitude of movement of key joints.

        Args:
            velocity_data (np.ndarray): The velocity data for one frame, shape (num_landmarks, 3).

        Returns:
            float: The calculated motion energy.
        """
        # Focus on hands and face for motion energy calculation
        # Pose landmarks 0-10 (face), left hand 33-53, right hand 54-74
        hand_landmarks_velo = np.concatenate([
            velocity_data[33:54], 
            velocity_data[54:75]
        ])
        
        # Calculate the L2 norm (magnitude) of velocity vectors and sum them up
        motion_energy = np.linalg.norm(hand_landmarks_velo, axis=1).mean()
        return motion_energy

    def process_frame(self, frame_data, model_confidence=0.0):
        """
        Processes a new frame of keypoint data to update the signing state.

        Args:
            frame_data (dict): A dictionary containing 'landmarks', 'velocity', etc. for a single frame.
            model_confidence (float): The confidence score from the recognition model for the current window.

        Returns:
            tuple: A tuple containing (status, sign_sequence).
                   'status' can be 'START', 'SIGNING', 'END', or 'IDLE'.
                   'sign_sequence' is the buffer of frames if a sign has just ended.
        """
        motion_energy = self.calculate_motion_energy(frame_data['velocity'])
        
        is_active = motion_energy > self.motion_threshold

        if not self.is_signing:
            if is_active:
                # Potential start of a sign
                self.active_frames_count += 1
                self.sign_buffer.append(frame_data)
                if self.active_frames_count >= self.min_sign_duration:
                    self.is_signing = True
                    self.silent_frames_count = 0
                    return 'START', None
            else:
                # Reset if we see inactivity before a sign is confirmed
                self.active_frames_count = 0
                self.sign_buffer = []
                return 'IDLE', None
        
        # If we are currently in a signing state
        else:
            if is_active:
                self.sign_buffer.append(frame_data)
                self.silent_frames_count = 0 # Reset silence counter
                return 'SIGNING', None
            else:
                # Frame is not active, start counting silent frames
                self.silent_frames_count += 1
                if self.silent_frames_count >= self.max_silence_duration:
                    # End of sign detected
                    
                    # Check if the model was confident about the prediction
                    if model_confidence < self.confidence_threshold:
                        # If confidence is low, discard the buffer as noise
                        self.reset()
                        return 'IDLE', None

                    # Return the buffer of frames for processing
                    sequence_to_process = list(self.sign_buffer)
                    self.reset()
                    return 'END', sequence_to_process
                else:
                    # Still in the grace period of silence
                    self.sign_buffer.append(frame_data) # Keep collecting frames
                    return 'SIGNING', None
    
    def reset(self):
        """Resets the state of the segmentation module."""
        self.is_signing = False
        self.sign_buffer = []
        self.silent_frames_count = 0
        self.active_frames_count = 0
