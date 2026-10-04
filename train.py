"""
ASL - Training Script
=============================
Train the landmark-based ASL model.
"""

import argparse
import os
import sys
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import FeatureExtractor


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Folder containing one sub-folder of images per letter (A..Z).
    # Override with --dataset or the ASL_DATASET_PATH environment variable.
    DATASET_PATH = os.environ.get("ASL_DATASET_PATH", "data/asl_alphabet_train")
    MODEL_PATH = str(PROJECT_ROOT / "models" / "asl_landmark_model.h5")
    MEDIAPIPE_MODEL = str(PROJECT_ROOT / "src" / "hand_landmarker.task")
    
    LETTERS = [chr(65 + i) for i in range(26)]
    NUM_CLASSES = 26
    
    BATCH_SIZE = 128
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    VALIDATION_SPLIT = 0.15
    
    HIDDEN_LAYERS = [256, 128, 64]
    DROPOUT = 0.3
    L2_REG = 1e-4


# =============================================================================
# LANDMARK EXTRACTION
# =============================================================================

def extract_landmarks(
    dataset_path: str,
    letters: List[str],
    samples_per_letter: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract landmarks from dataset images."""
    
    print("\n" + "=" * 50)
    print("  EXTRACTING LANDMARKS")
    print("=" * 50)
    
    base_options = mp_tasks.BaseOptions(model_asset_path=Config.MEDIAPIPE_MODEL)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)
    extractor = FeatureExtractor()
    
    all_features = []
    all_labels = []
    
    for idx, letter in enumerate(letters):
        letter_path = os.path.join(dataset_path, letter)
        
        if not os.path.exists(letter_path):
            print(f"  WARNING: {letter} not found")
            continue
        
        images = [f for f in os.listdir(letter_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(images) > samples_per_letter:
            np.random.seed(42)
            images = list(np.random.choice(images, samples_per_letter, replace=False))
        
        count = 0
        for img_file in images:
            img = cv2.imread(os.path.join(letter_path, img_file))
            if img is None:
                continue
            
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)
            
            if result.hand_landmarks:
                features = extractor.extract(result.hand_landmarks[0])
                all_features.append(features)
                all_labels.append(idx)
                count += 1
        
        print(f"  {letter}: {count}/{len(images)}")
    
    detector.close()
    
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    
    n_features = X.shape[1] if X.ndim == 2 else 0
    print(f"\nTotal: {len(X)} samples, {n_features} features")
    
    return X, y


# =============================================================================
# MODEL
# =============================================================================

def create_model(input_dim: int, num_classes: int) -> Model:
    """Create MLP model."""
    
    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    
    for i, units in enumerate(Config.HIDDEN_LAYERS):
        x = layers.Dense(units, kernel_regularizer=regularizers.l2(Config.L2_REG))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.Dropout(Config.DROPOUT)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)


# =============================================================================
# TRAINING
# =============================================================================

def train():
    """Train the model."""
    
    print("\n" + "=" * 50)
    print("  ASL EXPERT - TRAINING")
    print("=" * 50)
    
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
    
    # Extract landmarks
    X, y = extract_landmarks(Config.DATASET_PATH, Config.LETTERS)
    
    # Validate extracted data
    if len(X) == 0 or len(y) == 0:
        print("\n  ERROR: No landmarks extracted from dataset!")
        print("  Please check:")
        print(f"    1. Dataset path exists: {Config.DATASET_PATH}")
        print("    2. Dataset contains valid images")
        print("    3. MediaPipe can detect hands in images")
        return
    
    if len(np.unique(y)) < 2:
        print(f"\n  ERROR: Need at least 2 classes, found {len(np.unique(y))}")
        print("  Cannot perform stratified split with only one class.")
        return
    
    # Split
    y_cat = keras.utils.to_categorical(y, Config.NUM_CLASSES)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=Config.VALIDATION_SPLIT, random_state=42, stratify=y
    )
    
    # Augment
    noise = 0.02
    X_train_aug = np.concatenate([
        X_train,
        X_train + np.random.normal(0, noise, X_train.shape),
        X_train + np.random.normal(0, noise, X_train.shape),
    ])
    y_train_aug = np.concatenate([y_train, y_train, y_train])
    
    # Class weights
    class_weights = dict(zip(
        range(Config.NUM_CLASSES),
        compute_class_weight('balanced', classes=np.unique(y), y=y)
    ))
    
    print(f"\n  Training: {len(X_train_aug)}, Validation: {len(X_val)}")
    
    # Create model
    model = create_model(X_train.shape[1], Config.NUM_CLASSES)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(Config.MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    ]
    
    # Train
    print("\n  Training...")
    start = time.time()
    
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    print(f"\n  Best accuracy: {max(history.history['val_accuracy']):.2%}")
    print(f"  Training time: {time.time() - start:.1f}s")
    print(f"  Model saved: {Config.MODEL_PATH}")
    
    # Evaluate
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    y_true = np.argmax(y_val, axis=1)
    print("\n" + classification_report(y_true, y_pred, target_names=Config.LETTERS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the landmark-based ASL model.")
    parser.add_argument(
        "--dataset",
        default=Config.DATASET_PATH,
        help="Folder with one sub-folder of images per letter (A..Z). "
             "Defaults to $ASL_DATASET_PATH or data/asl_alphabet_train.",
    )
    Config.DATASET_PATH = parser.parse_args().dataset
    train()
