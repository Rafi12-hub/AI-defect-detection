"""
anomaly_model.py
================
Builds and trains a CNN Autoencoder on GOOD (non-defective) images.
"""

import os
import yaml
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential

BASE = Path(__file__).parent.parent
SETTINGS_FILE = BASE / "config" / "settings.yaml"

with open(SETTINGS_FILE, "r") as f:
    settings = yaml.safe_load(f)

img_size = settings['training']['yolo']['img_size']
epochs = settings['training']['autoencoder']['epochs']
batch_size = settings['training']['autoencoder']['batch_size']
learning_rate = settings['training']['autoencoder']['learning_rate']

def build_autoencoder():
    aug_settings = settings['augmentations']['autoencoder']
    
    # Data Augmentation Layer (Sequential)
    data_augmentation = Sequential([
        layers.RandomFlip(aug_settings['flip']),
        layers.RandomRotation(aug_settings.get('rotation_factor', 0.2)),
        layers.RandomContrast(aug_settings.get('contrast_factor', 0.2))
    ], name="data_augmentation")

    input_img = layers.Input(shape=(img_size, img_size, 3))
    
    x = data_augmentation(input_img)
    x = layers.Rescaling(1./255)(x)

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    
    return autoencoder

def train_autoencoder():
    print(f"  AI Defect Detection — Autoencoder Training")
    
    good_dir = BASE / "data" / "good"
    if not good_dir.exists():
        print(f"❌ GOOD image directory not found at {good_dir}")
        print("   Run 'python reorganize_dataset.py' first.")
        return
    
    model = build_autoencoder()
    
    EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    good_paths = [str(f) for f in good_dir.iterdir() if f.suffix.lower() in EXTS]
    
    if not good_paths:
        print(f"❌ No images found in {good_dir}")
        return
    
    print(f"  Found {len(good_paths)} GOOD images for training")
    
    # Load and preprocess images
    def load_images(paths):
        images = []
        for p in paths:
            img = tf.keras.utils.load_img(p, target_size=(img_size, img_size))
            img = tf.keras.utils.img_to_array(img) / 255.0
            images.append(img)
        return np.array(images)
    
    train_images = load_images(good_paths)
    print(f"  Loaded {train_images.shape[0]} images, shape {train_images.shape[1:]}")
    
    # Split 80/20 train/val
    split = int(0.8 * len(train_images))
    train_data = train_images[:split]
    val_data = train_images[split:]
    
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}")
    
    # Callbacks
    model_path = BASE / "models" / "autoencoder.h5"
    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
    ]
    
    history = model.fit(
        train_data, train_data,
        validation_data=(val_data, val_data),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    
    model.save(str(model_path))
    print(f"✅ Trained autoencoder saved to {model_path}")
    
    # Save training history
    import json
    hist_path = BASE / "models" / "autoencoder_history.json"
    with open(hist_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
    print(f"✅ Training history saved to {hist_path}")

if __name__ == "__main__":
    train_autoencoder()
