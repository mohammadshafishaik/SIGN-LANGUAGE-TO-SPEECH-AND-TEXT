#!/usr/bin/env python3
"""
Train WLASL-100 Sign Language Recognition Model
Uses processed dataset with 12,730 training samples across 100 ASL words
"""

import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
CONFIG = {
    'data_dir': 'dataset/wlasl_100_processed',
    'checkpoint_dir': 'checkpoints',
    'logs_dir': 'logs',
    
    # Model hyperparameters
    'input_shape': (60, 171),  # (frames, features)
    'n_classes': 100,
    'lstm_units': [128, 64],
    'dense_units': 128,
    'dropout_rate': 0.3,
    'dense_dropout': 0.5,
    
    # Training hyperparameters
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'validation_split': 0.0,  # We have separate val set
    
    # Early stopping
    'patience': 10,
    'min_delta': 0.001,
}

class WLASL100Trainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
        self.metadata = None
        
        # Create directories
        Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        Path(config['logs_dir']).mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load preprocessed dataset"""
        print("=" * 70)
        print("📂 LOADING DATASET")
        print("=" * 70)
        
        data_dir = Path(self.config['data_dir'])
        
        # Load metadata
        with open(data_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        print(f"📋 Dataset Info:")
        print(f"   Classes: {self.metadata['n_classes']}")
        print(f"   Features: {self.metadata['n_features']}")
        print(f"   Frames: {self.metadata['n_frames']}")
        print(f"   Training samples: {self.metadata['n_train']:,}")
        print(f"   Validation samples: {self.metadata['n_val']:,}")
        print(f"   Test samples: {self.metadata['n_test']:,}")
        
        # Load training data
        print("\n📦 Loading training data...")
        train_data = np.load(data_dir / 'train_data.npz')
        self.X_train = train_data['X'].astype(np.float32)
        self.y_train = train_data['y']
        
        # Handle NaN and Inf values (replace with 0)
        self.X_train = np.nan_to_num(self.X_train, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Load validation data
        print("📦 Loading validation data...")
        val_data = np.load(data_dir / 'val_data.npz')
        self.X_val = val_data['X'].astype(np.float32)
        self.y_val = val_data['y']
        self.X_val = np.nan_to_num(self.X_val, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Load test data
        print("📦 Loading test data...")
        test_data = np.load(data_dir / 'test_data.npz')
        self.X_test = test_data['X'].astype(np.float32)
        self.y_test = test_data['y']
        self.X_test = np.nan_to_num(self.X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert labels to categorical
        self.y_train_cat = keras.utils.to_categorical(self.y_train, self.config['n_classes'])
        self.y_val_cat = keras.utils.to_categorical(self.y_val, self.config['n_classes'])
        self.y_test_cat = keras.utils.to_categorical(self.y_test, self.config['n_classes'])
        
        print(f"\n✅ Data loaded successfully!")
        print(f"   X_train: {self.X_train.shape}")
        print(f"   y_train: {self.y_train.shape}")
        print(f"   X_val: {self.X_val.shape}")
        print(f"   X_test: {self.X_test.shape}")
        
        # Data statistics
        print(f"\n📊 Data Statistics:")
        print(f"   Value range: [{self.X_train.min():.3f}, {self.X_train.max():.3f}]")
        print(f"   Mean: {self.X_train.mean():.3f}")
        print(f"   Std: {self.X_train.std():.3f}")
        
    def build_model(self):
        """Build LSTM model for sign language recognition"""
        print("\n" + "=" * 70)
        print("🏗️  BUILDING MODEL")
        print("=" * 70)
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.config['input_shape']),
            
            # First LSTM layer
            layers.Bidirectional(
                layers.LSTM(
                    self.config['lstm_units'][0],
                    return_sequences=True,
                    dropout=self.config['dropout_rate']
                )
            ),
            layers.BatchNormalization(),
            
            # Second LSTM layer
            layers.Bidirectional(
                layers.LSTM(
                    self.config['lstm_units'][1],
                    return_sequences=False,
                    dropout=self.config['dropout_rate']
                )
            ),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(
                self.config['dense_units'],
                activation='relu'
            ),
            layers.Dropout(self.config['dense_dropout']),
            layers.BatchNormalization(),
            
            # Output layer
            layers.Dense(
                self.config['n_classes'],
                activation='softmax'
            )
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
        )
        
        self.model = model
        
        print("\n📋 Model Summary:")
        model.summary()
        
        # Calculate parameters
        total_params = model.count_params()
        print(f"\n📊 Total Parameters: {total_params:,}")
        
        return model
    
    def train(self):
        """Train the model"""
        print("\n" + "=" * 70)
        print("🚀 TRAINING MODEL")
        print("=" * 70)
        
        # Callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            # Model checkpoint - save best model
            keras.callbacks.ModelCheckpoint(
                filepath=f"{self.config['checkpoint_dir']}/wlasl_100_best.keras",
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['patience'],
                min_delta=self.config['min_delta'],
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            
            # TensorBoard
            keras.callbacks.TensorBoard(
                log_dir=f"{self.config['logs_dir']}/wlasl100_{timestamp}",
                histogram_freq=1
            ),
            
            # CSV Logger
            keras.callbacks.CSVLogger(
                f"{self.config['logs_dir']}/training_{timestamp}.csv"
            )
        ]
        
        # Train model
        print(f"\n🎯 Training Configuration:")
        print(f"   Batch size: {self.config['batch_size']}")
        print(f"   Epochs: {self.config['epochs']}")
        print(f"   Learning rate: {self.config['learning_rate']}")
        print(f"   Early stopping patience: {self.config['patience']}")
        print(f"\n⏱️  Starting training... (this may take 1-2 hours)")
        print("-" * 70)
        
        self.history = self.model.fit(
            self.X_train, self.y_train_cat,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=(self.X_val, self.y_val_cat),
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ Training completed!")
        
    def evaluate(self):
        """Evaluate model on test set"""
        print("\n" + "=" * 70)
        print("📊 EVALUATING MODEL")
        print("=" * 70)
        
        # Load best model
        print("\n📦 Loading best model...")
        best_model_path = f"{self.config['checkpoint_dir']}/wlasl_100_best.keras"
        self.model = keras.models.load_model(best_model_path)
        print(f"   ✅ Loaded: {best_model_path}")
        
        # Evaluate on test set
        print("\n🧪 Testing on test set...")
        test_results = self.model.evaluate(
            self.X_test, self.y_test_cat,
            batch_size=self.config['batch_size'],
            verbose=1
        )
        
        print("\n" + "=" * 70)
        print("📈 FINAL RESULTS")
        print("=" * 70)
        print(f"Test Loss: {test_results[0]:.4f}")
        print(f"Test Accuracy: {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
        print(f"Test Top-5 Accuracy: {test_results[2]:.4f} ({test_results[2]*100:.2f}%)")
        
        # Save results
        results = {
            'test_loss': float(test_results[0]),
            'test_accuracy': float(test_results[1]),
            'test_top5_accuracy': float(test_results[2]),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f"{self.config['checkpoint_dir']}/wlasl_100_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {self.config['checkpoint_dir']}/wlasl_100_results.json")
        
        return test_results
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("⚠️  No training history available")
            return
        
        print("\n📊 Generating training plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"{self.config['logs_dir']}/training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved plot: {plot_path}")
        
        plt.close()
    
    def run(self):
        """Run complete training pipeline"""
        print("\n" + "=" * 70)
        print("🎬 WLASL-100 TRAINING PIPELINE")
        print("=" * 70)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load data
        self.load_data()
        
        # Build model
        self.build_model()
        
        # Train model
        self.train()
        
        # Evaluate model
        self.evaluate()
        
        # Plot training history
        self.plot_training_history()
        
        print("\n" + "=" * 70)
        print("✅ TRAINING PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n📁 Outputs:")
        print(f"   Model: {self.config['checkpoint_dir']}/wlasl_100_best.keras")
        print(f"   Results: {self.config['checkpoint_dir']}/wlasl_100_results.json")
        print(f"   Logs: {self.config['logs_dir']}/")
        print(f"\n🚀 Next step: Update webapp_simple.py to use the new model!")

def main():
    """Main training function"""
    # GPU configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"🎮 GPU Available: {len(gpus)} device(s)")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("💻 Running on CPU")
    
    # Create trainer and run
    trainer = WLASL100Trainer(CONFIG)
    trainer.run()

if __name__ == "__main__":
    main()
