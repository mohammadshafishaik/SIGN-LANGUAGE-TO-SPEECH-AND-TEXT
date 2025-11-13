import os
import numpy as np
import pandas as pd
import argparse
import sys
import datetime
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization, 
                                     MultiHeadAttention, GlobalAveragePooling1D, Reshape, Add)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger

# --- Transformer Building Block ---
# Moved directly into this script to remove external dependencies.
def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    """A single transformer encoder block."""
    # Attention and Normalization
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(Add()([inputs, x]))
    
    # Feed Forward Part
    ff_out = Dense(ff_dim, activation="relu")(x)
    ff_out = Dropout(dropout)(ff_out)
    ff_out = Dense(inputs.shape[-1])(ff_out)
    
    # Add and Normalize
    return LayerNormalization(epsilon=1e-6)(Add()([x, ff_out]))

# --- Data Loading Functions ---
# Moved directly into this script.
def load_and_process_npz(path, label, num_classes):
    """Loads data, standardizes it, and one-hot encodes the label."""
    try:
        # Use path.numpy().decode('utf-8') because this runs in a tf.py_function
        filepath = path.numpy().decode('utf-8')
        with np.load(filepath) as data:
            keypoints = data['data']
        
        # Standardize the data: subtract mean, divide by standard deviation
        mean = np.mean(keypoints, axis=(0, 1), keepdims=True)
        std = np.std(keypoints, axis=(0, 1), keepdims=True)
        std[std == 0] = 1e-6 # Avoid division by zero for flat data
        keypoints = (keypoints - mean) / std
        
        # One-hot encode the label
        label_one_hot = tf.keras.utils.to_categorical(label, num_classes=num_classes)
        
        return keypoints.astype(np.float32), label_one_hot.astype(np.float32)
    except Exception as e:
        # Return empty arrays on error to be filtered out later
        print(f"Error loading or processing file {path.numpy().decode('utf-8')}: {e}", file=sys.stderr)
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

def create_dataset(data_dir, batch_size, num_classes, input_shape):
    """Creates a tf.data.Dataset from a directory of .npz files."""
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        print(f"FATAL: metadata.csv not found in {data_dir}. Cannot create dataset.", file=sys.stderr)
        return None

    df = pd.read_csv(metadata_path)
    if df.empty:
        print(f"Warning: metadata.csv in {data_dir} is empty. Cannot create dataset.", file=sys.stderr)
        return None

    file_paths = [os.path.join(data_dir, f"{vid}.npz") for vid in df['unified_video_id']]
    labels = df['label_encoded'].values

    # Filter out files that don't actually exist to prevent errors
    existing_files_data = [(path, label) for path, label in zip(file_paths, labels) if os.path.exists(path)]
    if not existing_files_data:
        print(f"FATAL: No existing .npz files found for the entries in {metadata_path}.", file=sys.stderr)
        return None
        
    file_paths, labels = zip(*existing_files_data)
    
    dataset = tf.data.Dataset.from_tensor_slices((list(file_paths), list(labels)))
    dataset = dataset.shuffle(len(file_paths))
    
    # Use tf.py_function to wrap the numpy-based loading function
    dataset = dataset.map(lambda path, label: tuple(tf.py_function(
        load_and_process_npz, [path, label, num_classes], [tf.float32, tf.float32])))
    
    # Filter out any samples that had loading errors
    dataset = dataset.filter(lambda data, label: tf.shape(data)[0] > 0)

    # Pad batches to a fixed size
    padded_shapes = (tf.TensorShape(input_shape), tf.TensorShape([num_classes]))
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# --- Model Definition ---
def build_transformer_classifier(input_shape, num_classes, head_size=256, num_heads=4, ff_dim=4, num_blocks=4, dropout=0.1):
    """Builds a Transformer Encoder-based model for sequence classification."""
    inputs = Input(shape=input_shape)
    
    # Reshape to (batch_size, sequence_length, features)
    x = Reshape((input_shape[0], input_shape[1] * input_shape[2]))(inputs)
    
    for _ in range(num_blocks):
        x = transformer_encoder_block(x, head_size, num_heads, ff_dim, dropout=dropout)

    x = GlobalAveragePooling1D(data_format="channels_last")(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model

# --- Training Phase Runner ---
def run_training_phase(model, phase_name, train_dataset, val_dataset, epochs, learning_rate, model_dir, patience):
    """Runs a single phase of training."""
    print(f"\n----- Starting Training Phase: {phase_name} -----")
    
    log_dir = os.path.join("logs", f"{phase_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = [
        TensorBoard(log_dir=log_dir),
        CSVLogger(os.path.join(log_dir, "training_log.csv")),
        ModelCheckpoint(
            filepath=os.path.join(model_dir, f"best_model_{phase_name}.h5"),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
    ]

    # Use the legacy optimizer on Apple Silicon to avoid performance issues
    # Add clipnorm=1.0 for gradient clipping to prevent exploding gradients
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, clipnorm=1.0)
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    
    print(f"----- Phase '{phase_name}' Complete -----")
    return model, history

# --- Main Orchestrator ---
def main(args):
    """Main function to orchestrate the training process."""
    print("--- Script Start ---")
    print(f"Arguments: {args}")
    os.makedirs(args.model_dir, exist_ok=True)

    # --- Determine Vocabulary and Model Shape ---
    print("--- Determining Vocabulary and Model Shape ---")
    train_metadata_path = 'dataset/keypoints_combined/train/metadata.csv'
    if not os.path.exists(train_metadata_path):
        print(f"FATAL: Cannot find training metadata at {train_metadata_path}. Run data prep scripts.", file=sys.stderr)
        return

    df = pd.read_csv(train_metadata_path)
    num_classes = df['label_encoded'].nunique()
    print(f"Number of classes: {num_classes}")
    
    try:
        sample_npz_path = os.path.join('dataset/keypoints_combined/train', f"{df['unified_video_id'].iloc[0]}.npz")
        with np.load(sample_npz_path) as data:
            input_shape = data['data'].shape
        print(f"Input shape from sample: {input_shape}")
    except (IndexError, FileNotFoundError):
        print("FATAL: Could not load a sample npz file to determine input shape.", file=sys.stderr)
        return

    # --- Build Model ---
    print("--- Building Model ---")
    model = build_transformer_classifier(input_shape=input_shape, num_classes=num_classes)
    model.summary()

    # --- Phase 1: Pre-training ---
    if 'pretrained_wlasl' in args.phases:
        train_dir = 'dataset/keypoints_combined/train'
        val_dir = 'dataset/keypoints_combined/val'

        print("--- Creating Datasets for Pre-training ---")
        train_dataset = create_dataset(train_dir, args.batch_size, num_classes, input_shape)
        val_dataset = create_dataset(val_dir, args.batch_size, num_classes, input_shape)

        if train_dataset and val_dataset:
            model, _ = run_training_phase(
                model=model,
                phase_name='pretrained_wlasl',
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                model_dir=args.model_dir,
                patience=args.patience
            )
            print("--- Pre-training Phase Complete ---")
        else:
            print("Skipping pre-training phase due to dataset creation failure.", file=sys.stderr)

    # --- Phase 2: Fine-tuning ---
    if 'finetune_local' in args.phases:
        # This example re-uses the same data; in a real scenario, you might have a different dataset split.
        train_dir = 'dataset/keypoints_combined/train'
        val_dir = 'dataset/keypoints_combined/val'

        print("--- Creating Datasets for Fine-tuning ---")
        train_dataset_ft = create_dataset(train_dir, args.batch_size, num_classes, input_shape)
        val_dataset_ft = create_dataset(val_dir, args.batch_size, num_classes, input_shape)

        if train_dataset_ft and val_dataset_ft:
            # Reduce learning rate for fine-tuning
            finetune_lr = args.learning_rate / 10
            print(f"Fine-tuning with reduced learning rate: {finetune_lr}")
            
            model, _ = run_training_phase(
                model=model,
                phase_name='finetune_local',
                train_dataset=train_dataset_ft,
                val_dataset=val_dataset_ft,
                epochs=args.epochs,
                learning_rate=finetune_lr,
                model_dir=args.model_dir,
                patience=args.patience
            )
            print("--- Fine-tuning Phase Complete ---")
        else:
            print("Skipping fine-tuning phase due to dataset creation failure.", file=sys.stderr)
    
    print("--- Script End ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-dataset training for sign language recognition.")
    parser.add_argument('--phases', type=str, nargs='+', default=['pretrained_wlasl', 'finetune_local'], help='Training phases to run.')
    parser.add_argument('--model_dir', type=str, default='models/saved_models', help='Directory to save models.')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate.')
    
    # Add project root to Python path to allow running from any directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    args = parser.parse_args()
    main(args)
