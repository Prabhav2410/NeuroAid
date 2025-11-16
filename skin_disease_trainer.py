# skin_disease_trainer.py
"""
Skin Disease Classification Model Training - OPTIMIZED FOR SPEED
Compatible with both CPU (venv) and GPU (tf_wsl with CUDA)
Supports TensorFlow 2.x with lower versions for GPU compatibility

OPTIMIZED VERSION: Ready in 1.5-2 hours with 20 epochs!
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import json
from datetime import datetime

# ============================================================
# CONFIGURATION - OPTIMIZED FOR SPEED
# ============================================================

class Config:
    # Paths
    TRAIN_DIR = "skin_disease_dataset/train"
    TEST_DIR = "skin_disease_dataset/test"
    MODEL_SAVE_PATH = "skin_disease_model.h5"
    CLASS_NAMES_PATH = "skin_disease_classes.json"
    HISTORY_PATH = "training_history.json"
    
    # Image parameters - OPTIMIZED: Smaller images = faster training
    IMG_HEIGHT = 160  # Reduced from 224 for speed
    IMG_WIDTH = 160   # Reduced from 224 for speed
    IMG_CHANNELS = 3
    
    # Training parameters - OPTIMIZED FOR SPEED
    BATCH_SIZE = 64   # Increased from 32 for faster epochs
    EPOCHS = 20       # Reduced from 50 - sufficient for good results
    LEARNING_RATE = 0.001
    
    # Model architecture
    USE_PRETRAINED = True  # Use transfer learning
    FREEZE_BASE = True     # Freeze base model initially
    
    # GPU configuration
    USE_GPU = True
    MIXED_PRECISION = False  # Enable for faster training on modern GPUs

# ============================================================
# GPU SETUP - OPTIMIZED
# ============================================================

def setup_gpu():
    """Configure GPU settings for optimal performance"""
    print("=" * 60)
    print("GPU CONFIGURATION")
    print("=" * 60)
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to prevent TF from allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"✅ Found {len(gpus)} GPU(s)")
            print(f"GPU Details: {gpus}")
            
            # SPEED BOOST: Enable XLA JIT compilation
            tf.config.optimizer.set_jit(True)
            print("✅ XLA JIT compilation enabled for faster training")
            
            # Enable mixed precision for faster training (optional)
            if Config.MIXED_PRECISION:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("✅ Mixed precision enabled")
            
        except RuntimeError as e:
            print(f"⚠️ GPU setup error: {e}")
    else:
        print("⚠️ No GPU found. Training will use CPU")
        print("For GPU support, ensure CUDA and cuDNN are properly installed")
    
    # Print TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    print("=" * 60 + "\n")

# ============================================================
# DATA LOADING AND PREPROCESSING - OPTIMIZED
# ============================================================

def create_data_generators():
    """Create training and validation data generators with augmentation"""
    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    
    # Training data augmentation - OPTIMIZED: Simpler augmentation = faster processing
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,        # Reduced from 20
        width_shift_range=0.15,   # Reduced from 0.2
        height_shift_range=0.15,  # Reduced from 0.2
        zoom_range=0.15,          # Reduced from 0.2
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 80% train, 20% validation
    )
    
    # Test data - only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        Config.TRAIN_DIR,
        target_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Load validation data
    validation_generator = train_datagen.flow_from_directory(
        Config.TRAIN_DIR,
        target_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Load test data
    test_generator = test_datagen.flow_from_directory(
        Config.TEST_DIR,
        target_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Get class names
    class_names = list(train_generator.class_indices.keys())
    num_classes = len(class_names)
    
    print(f"✅ Training samples: {train_generator.samples}")
    print(f"✅ Validation samples: {validation_generator.samples}")
    print(f"✅ Test samples: {test_generator.samples}")
    print(f"✅ Number of classes: {num_classes}")
    print(f"✅ Classes: {class_names}")
    print(f"✅ Batch size: {Config.BATCH_SIZE} (optimized for speed)")
    print(f"✅ Steps per epoch: ~{train_generator.samples // Config.BATCH_SIZE}")
    print("=" * 60 + "\n")
    
    return train_generator, validation_generator, test_generator, class_names, num_classes

# ============================================================
# MODEL ARCHITECTURE - OPTIMIZED
# ============================================================

def create_model(num_classes):
    """Create CNN model with transfer learning"""
    print("=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    
    if Config.USE_PRETRAINED:
        # OPTIMIZED: Use MobileNetV2 (fastest and most efficient)
        # Falls back to EfficientNetB0 if MobileNetV2 fails
        try:
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.IMG_CHANNELS),
                alpha=1.0  # Full model - can reduce to 0.75 for even faster training
            )
            print("✅ Using MobileNetV2 as base model (optimized for speed)")
        except:
            try:
                base_model = EfficientNetB0(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.IMG_CHANNELS)
                )
                print("✅ Using EfficientNetB0 as base model (fallback)")
            except:
                raise Exception("❌ Could not load pre-trained model. Check TensorFlow installation.")
        
        # Freeze base model layers initially
        if Config.FREEZE_BASE:
            base_model.trainable = False
            print("✅ Base model layers frozen")
        
        # Build model - OPTIMIZED: Simpler head for faster training
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
    else:
        # Build custom CNN from scratch
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.IMG_CHANNELS)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Classifier
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        print("✅ Using custom CNN architecture")
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    print("\nModel Summary:")
    model.summary()
    print(f"\n⚡ Total trainable parameters: {model.count_params():,}")
    print("=" * 60 + "\n")
    
    return model

# ============================================================
# TRAINING - OPTIMIZED
# ============================================================

def train_model(model, train_gen, val_gen, epochs=Config.EPOCHS):
    """Train the model with callbacks"""
    print("=" * 60)
    print("TRAINING - OPTIMIZED FOR SPEED")
    print("=" * 60)
    
    # Callbacks - OPTIMIZED: More aggressive early stopping
    callbacks = [
        # Save best model
        ModelCheckpoint(
            Config.MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping - OPTIMIZED: Reduced patience for faster completion
        EarlyStopping(
            monitor='val_loss',
            patience=4,  # Reduced from 10 - stops if no improvement for 4 epochs
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau - OPTIMIZED
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,  # Reduced from 5
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print(f"🚀 Starting training for maximum {epochs} epochs")
    print(f"⏱️  Estimated completion time: 1.5-2 hours")
    print(f"⚡ Early stopping enabled - likely stops around epoch 12-16")
    print(f"📊 Image size: {Config.IMG_HEIGHT}x{Config.IMG_WIDTH} (optimized)")
    print(f"📦 Batch size: {Config.BATCH_SIZE} (optimized)")
    print()
    
    # Train model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✅ Training completed!")
    print("=" * 60 + "\n")
    
    return history

# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, test_gen):
    """Evaluate model on test set"""
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    # Evaluate
    results = model.evaluate(test_gen, verbose=1)
    
    print(f"\n✅ Test Loss: {results[0]:.4f}")
    print(f"✅ Test Accuracy: {results[1]:.4f}")
    print(f"✅ Test Top-3 Accuracy: {results[2]:.4f}")
    print("=" * 60 + "\n")
    
    return results

# ============================================================
# VISUALIZATION
# ============================================================

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("✅ Training history plot saved as 'training_history.png'")
    plt.close()

# ============================================================
# SAVE ARTIFACTS
# ============================================================

def save_artifacts(class_names, history):
    """Save class names and training history"""
    # Save class names
    with open(Config.CLASS_NAMES_PATH, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"✅ Class names saved to {Config.CLASS_NAMES_PATH}")
    
    # Save training history
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'top_3_accuracy': [float(x) for x in history.history['top_3_accuracy']],
        'val_top_3_accuracy': [float(x) for x in history.history['val_top_3_accuracy']],
    }
    
    with open(Config.HISTORY_PATH, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"✅ Training history saved to {Config.HISTORY_PATH}")

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main training pipeline"""
    print("\n" + "=" * 60)
    print("SKIN DISEASE CLASSIFICATION - OPTIMIZED TRAINING")
    print("=" * 60)
    print("⚡ SPEED OPTIMIZATIONS ENABLED")
    print(f"⏱️  Target completion time: 1.5-2 hours")
    print(f"📊 Max epochs: {Config.EPOCHS}")
    print(f"📦 Batch size: {Config.BATCH_SIZE}")
    print(f"🖼️  Image size: {Config.IMG_HEIGHT}x{Config.IMG_WIDTH}")
    print("=" * 60 + "\n")
    
    # Setup GPU
    setup_gpu()
    
    # Check if directories exist
    if not os.path.exists(Config.TRAIN_DIR):
        raise FileNotFoundError(f"Training directory not found: {Config.TRAIN_DIR}")
    if not os.path.exists(Config.TEST_DIR):
        raise FileNotFoundError(f"Test directory not found: {Config.TEST_DIR}")
    
    # Load data
    train_gen, val_gen, test_gen, class_names, num_classes = create_data_generators()
    
    # Create model
    model = create_model(num_classes)
    
    # Train model
    history = train_model(model, train_gen, val_gen)
    
    # Evaluate model
    evaluate_model(model, test_gen)
    
    # Plot training history
    plot_training_history(history)
    
    # Save artifacts
    save_artifacts(class_names, history)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE! MODEL READY FOR DEMO")
    print("=" * 60)
    print(f"📁 Model saved to: {Config.MODEL_SAVE_PATH}")
    print(f"📁 Class names saved to: {Config.CLASS_NAMES_PATH}")
    print(f"📁 Training history saved to: {Config.HISTORY_PATH}")
    print(f"📁 Training plot saved to: training_history.png")
    print("\n🎯 Test your model:")
    print("   python skin_disease_prediction.py <image_path>")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()