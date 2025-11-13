import numpy as np
import argparse
import os

def load_keypoints(npz_path):
    """
    Loads keypoint data from a .npz file.

    Args:
        npz_path (str): The path to the .npz file.

    Returns:
        dict: A dictionary containing the landmark, velocity, and acceleration arrays.
    """
    try:
        return np.load(npz_path)
    except FileNotFoundError:
        print(f"Error: File not found at {npz_path}")
        return None

def generate_features(data, feature_type='3d_velo'):
    """
    Generates a specific feature set from the loaded keypoint data.

    Args:
        data (dict): A dictionary of keypoint data loaded from a .npz file.
        feature_type (str): The type of features to generate.
            Options: '2d', '3d', '3d_velo', '3d_velo_accel'.

    Returns:
        np.ndarray: A NumPy array of the generated features, flattened per frame.
    """
    landmarks = data['landmarks']
    
    if feature_type == '2d':
        # Use only x, y coordinates
        features = landmarks[:, :, :2]
    elif feature_type == '3d':
        # Use x, y, z coordinates
        features = landmarks
    elif feature_type == '3d_velo':
        # Concatenate 3D landmarks and velocity
        velocity = data['velocity']
        features = np.concatenate([landmarks, velocity], axis=2)
    elif feature_type == '3d_velo_accel':
        # Concatenate 3D landmarks, velocity, and acceleration
        velocity = data['velocity']
        acceleration = data['acceleration']
        features = np.concatenate([landmarks, velocity, acceleration], axis=2)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}. "
                         "Valid options are '2d', '3d', '3d_velo', '3d_velo_accel'.")

    # Flatten the features for each frame
    # (num_frames, num_landmarks, num_coords) -> (num_frames, num_landmarks * num_coords)
    num_frames = features.shape[0]
    features = features.reshape(num_frames, -1)
    
    return features

def pad_or_truncate_sequence(sequence, max_len):
    """
    Pads or truncates a sequence to a maximum length.

    Args:
        sequence (np.ndarray): The input sequence (num_frames, num_features).
        max_len (int): The target sequence length.

    Returns:
        np.ndarray: The processed sequence of shape (max_len, num_features).
    """
    if sequence.shape[0] > max_len:
        # Truncate the sequence
        return sequence[:max_len, :]
    elif sequence.shape[0] < max_len:
        # Pad the sequence with zeros
        padding_shape = (max_len - sequence.shape[0], sequence.shape[1])
        padding = np.zeros(padding_shape)
        return np.concatenate([sequence, padding], axis=0)
    else:
        return sequence

def main():
    """
    Main function to demonstrate feature generation from the command line.
    """
    parser = argparse.ArgumentParser(description="Generate feature sets from .npz keypoint files.")
    parser.add_argument("--npz_path", type=str, required=True, help="Path to the input .npz file.")
    parser.add_argument("--feature_type", type=str, default="3d_velo",
                        choices=['2d', '3d', '3d_velo', '3d_velo_accel'],
                        help="The type of features to generate.")
    parser.add_argument("--max_len", type=int, default=100,
                        help="The target sequence length for padding/truncation.")
    
    args = parser.parse_args()

    if not os.path.exists(args.npz_path):
        print(f"Error: The file {args.npz_path} does not exist.")
        return

    # 1. Load the data
    keypoint_data = load_keypoints(args.npz_path)
    if keypoint_data is None:
        return
        
    print(f"Original landmark shape: {keypoint_data['landmarks'].shape}")

    # 2. Generate the specified feature set
    features = generate_features(keypoint_data, args.feature_type)
    print(f"Generated features of type '{args.feature_type}' with shape: {features.shape}")

    # 3. Pad or truncate the sequence
    processed_sequence = pad_or_truncate_sequence(features, args.max_len)
    print(f"Processed sequence shape after padding/truncating to {args.max_len} frames: {processed_sequence.shape}")
    
    # Verify the output shape
    # For '3d_velo', we have (75 landmarks * (3 coords + 3 velo)) = 450 features per frame
    # For '2d', we have (75 landmarks * 2 coords) = 150 features per frame
    expected_features = 0
    if args.feature_type == '2d':
        expected_features = 75 * 2
    elif args.feature_type == '3d':
        expected_features = 75 * 3
    elif args.feature_type == '3d_velo':
        expected_features = 75 * (3 + 3)
    elif args.feature_type == '3d_velo_accel':
        expected_features = 75 * (3 + 3 + 3)
        
    print(f"Expected number of features per frame: {expected_features}")
    assert processed_sequence.shape == (args.max_len, expected_features), "Shape mismatch!"
    print("\n✅ Feature generation and processing successful.")


if __name__ == "__main__":
    # Example usage from the root of the project:
    # First, create a dummy file if you don't have one:
    # python -c "import numpy as np; np.savez_compressed('dummy.npz', landmarks=np.random.rand(90, 75, 3), velocity=np.random.rand(90, 75, 3), acceleration=np.random.rand(90, 75, 3))"
    #
    # Then run the utility script:
    # python pose_extractor/utils.py --npz_path dummy.npz --feature_type 3d_velo --max_len 100
    main()
