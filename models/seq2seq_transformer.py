import os
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, GRU, Dense, Dropout, TimeDistributed,
                                     Embedding, MultiHeadAttention, LayerNormalization,
                                     Add, Bidirectional)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pose_extractor.utils import load_keypoints, generate_features, pad_or_truncate_sequence

# --- Constants ---
MAX_SEQ_LENGTH = 100
FEATURE_TYPE = '3d_velo'
MODEL_PATH = 'models/saved_models/seq2seq_transformer.h5'
CTC_MODEL_PATH = 'models/saved_models/ctc_model.h5'

# --- Data Loading (re-used from baseline, can be refactored) ---
def load_data_for_seq2seq(keypoints_dir, metadata_path, feature_type, max_len):
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    metadata = pd.read_csv(metadata_path)
    all_features = []
    all_phrases = []

    for _, row in metadata.iterrows():
        video_id, phrase = row['video_id'], row['phrase']
        npz_path = os.path.join(keypoints_dir, f"{video_id}.npz")
        if not os.path.exists(npz_path):
            continue

        keypoint_data = load_keypoints(npz_path)
        features = generate_features(keypoint_data, feature_type)
        processed_sequence = pad_or_truncate_sequence(features, max_len)
        all_features.append(processed_sequence)
        all_phrases.append(phrase)

    # Tokenize text data
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, lower=True)
    tokenizer.fit_on_texts(all_phrases)
    
    # Add <start> and <end> tokens
    all_phrases_tokenized = tokenizer.texts_to_sequences(all_phrases)
    
    # For Seq2Seq, we need decoder input and output
    decoder_input_data = [([tokenizer.word_index['<start>']] + seq) for seq in all_phrases_tokenized]
    decoder_target_data = [(seq + [tokenizer.word_index['<end>']]) for seq in all_phrases_tokenized]

    # Pad sequences
    max_phrase_len = max(len(seq) for seq in decoder_target_data)
    decoder_input_data = tf.keras.preprocessing.sequence.pad_sequences(decoder_input_data, maxlen=max_phrase_len, padding='post')
    decoder_target_data = tf.keras.preprocessing.sequence.pad_sequences(decoder_target_data, maxlen=max_phrase_len, padding='post')

    X = np.array(all_features)
    y_decoder_in = np.array(decoder_input_data)
    y_decoder_out = tf.keras.utils.to_categorical(decoder_target_data, num_classes=len(tokenizer.word_index) + 1)

    return X, y_decoder_in, y_decoder_out, tokenizer, max_phrase_len

# --- Transformer Building Blocks ---
def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)
    
    # Feed Forward Part
    ff_out = Dense(ff_dim, activation="relu")(x)
    ff_out = Dropout(dropout)(ff_out)
    ff_out = Dense(inputs.shape[-1])(ff_out)
    
    # Add and Normalize
    return LayerNormalization(epsilon=1e-6)(x + ff_out)

def build_transformer_model(input_shape, num_classes, max_phrase_len, head_size=256, num_heads=4, ff_dim=4):
    """Builds a Transformer Encoder-only model for sequence classification."""
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Create multiple transformer blocks
    for _ in range(4):
        x = transformer_encoder_block(x, head_size, num_heads, ff_dim, dropout=0.1)

    # Pooling and Output
    x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    x = Dropout(0.1)(x)
    x = Dense(20, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model

# --- CTC Model ---
def build_ctc_model(input_shape, num_classes):
    """Builds a CTC-based model for sequence-to-sequence tasks."""
    inputs = Input(shape=input_shape, name="input_data")
    
    # Bidirectional GRUs
    x = Bidirectional(GRU(128, return_sequences=True))(inputs)
    x = Bidirectional(GRU(128, return_sequences=True))(x)
    
    # Output layer
    outputs = Dense(num_classes + 1, activation='softmax', name="ctc_output")(x) # +1 for blank token
    
    model = Model(inputs, outputs)
    
    # CTC loss function
    def ctc_loss(y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

    model.compile(optimizer='adam', loss=ctc_loss, metrics=['accuracy'])
    model.summary()
    return model

def main():
    parser = argparse.ArgumentParser(description="Train advanced sequence models.")
    parser.add_argument("--model_type", type=str, default="ctc", choices=['transformer', 'ctc'], help="Type of model to train.")
    # Add other args as needed: --train, --evaluate, etc.
    
    args = parser.parse_args()

    # Note: Data loading and training loops need to be implemented similarly to baseline_lstm.py
    # This file provides the model-building logic as a starting point.
    
    print(f"Building a '{args.model_type}' model...")
    
    # Dummy shapes for demonstration
    dummy_input_shape = (MAX_SEQ_LENGTH, 225) # (100 frames, 75 landmarks * 3 coords)
    dummy_num_classes = 15 # Example vocab size

    if args.model_type == 'transformer':
        # This is a sequence-to-one classifier, not seq2seq.
        # A full encoder-decoder transformer is more complex.
        build_transformer_model(dummy_input_shape, dummy_num_classes, max_phrase_len=20)
    elif args.model_type == 'ctc':
        build_ctc_model(dummy_input_shape, dummy_num_classes)
        
    print("\nModel architecture defined. Training and data loading pipelines would be next.")


if __name__ == "__main__":
    # This block is problematic when this file is imported by another script.
    # The argparse call will hijack the argument parsing of the importing script.
    # Commenting out the main call to prevent silent exits.
    # main()
    pass
