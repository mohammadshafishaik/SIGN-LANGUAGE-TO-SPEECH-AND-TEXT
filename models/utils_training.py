
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

class MultiSourceDataGenerator(tf.keras.utils.Sequence):
    """
    A Keras Sequence generator to load data from pre-processed .npz files,
    with optional filtering by data source.
    """
    def __init__(self, data_dir, batch_size, source_filter=None, shuffle=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        metadata_path = os.path.join(data_dir, 'metadata.csv')
        self.metadata = pd.read_csv(metadata_path)
        
        # Apply source filter if provided
        if source_filter:
            if isinstance(source_filter, str):
                source_filter = [source_filter]
            self.metadata = self.metadata[self.metadata['source'].isin(source_filter)]
            
        self.indices = self.metadata.index.tolist()
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        X, y = self.__data_generation(batch_indices)
        return X, y

    def on_epoch_end(self):
        """Updates indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_indices):
        """Generates data containing batch_size samples."""
        batch_metadata = self.metadata.loc[batch_indices]
        
        # Pre-allocate arrays
        # Assuming we know the shape from the merge script
        # Let's find one file to determine the shape dynamically
        sample_npz_path = os.path.join(self.data_dir, f"{self.metadata.iloc[0]['unified_video_id']}.npz")
        with np.load(sample_npz_path) as sample_npz:
            sample_data = sample_npz['data']
            feature_shape = sample_data.shape

        X = np.empty((len(batch_indices), *feature_shape))
        y = np.empty((len(batch_indices)), dtype=int)

        for i, (_, row) in enumerate(batch_metadata.iterrows()):
            npz_path = os.path.join(self.data_dir, f"{row['unified_video_id']}.npz")
            with np.load(npz_path) as npz_file:
                X[i,] = npz_file['data']
            y[i] = row['label_encoded']
            
        return X, y

def get_data_loader(data_dir, batch_size, source_filter=None, shuffle=True):
    """Factory function to create a data generator."""
    return MultiSourceDataGenerator(data_dir, batch_size, source_filter, shuffle)

def get_class_weights(data_dir, source_filter=None):
    """
    Computes class weights for handling imbalanced datasets.
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    metadata = pd.read_csv(metadata_path)
    
    if source_filter:
        if isinstance(source_filter, str):
            source_filter = [source_filter]
        metadata = metadata[metadata['source'].isin(source_filter)]
        
    if metadata.empty:
        return None

    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(metadata['label_encoded']),
        y=metadata['label_encoded']
    )
    return dict(enumerate(class_weights))

def freeze_layers(model, num_layers_to_freeze):
    """Freezes the first N layers of a model."""
    print(f"Freezing first {num_layers_to_freeze} layers.")
    for i, layer in enumerate(model.layers):
        if i < num_layers_to_freeze:
            layer.trainable = False
        else:
            layer.trainable = True

def unfreeze_layers(model):
    """Unfreezes all layers of a model."""
    print("Unfreezing all layers.")
    for layer in model.layers:
        layer.trainable = True

# TODO: Add functions for learning rate schedules (e.g., cosine decay)
# TODO: Add metric calculation functions (e.g., Word Error Rate - WER)
