# Before running this file make sure to 
# pip install tensorflow tf2onnx onnx "numpy<2.0"

import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import subprocess
import sys

def main():
    print("🚀 Starting Fashion MNIST Training Pipeline...")
    
    # ==========================================
    # 1. LOAD & PREPROCESS DATA
    # ==========================================
    print("\n[1/4] Loading Data...")
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # Reshape for CNN (Batch, 28, 28, 1)
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)

    # ==========================================
    # 2. BUILD CNN MODEL
    # ==========================================
    print("\n[2/4] Building CNN Model...")
    model = models.Sequential([
        # Data Augmentation (Flip/Rotation) to make model robust
        layers.RandomFlip("horizontal", input_shape=(28, 28, 1)),
        layers.RandomRotation(0.05),
        
        # Conv Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Conv Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Conv Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.Flatten(),

        # Dense Block
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(10) # Output layer (10 classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # ==========================================
    # 3. TRAIN
    # ==========================================
    print("\n[3/4] Training (This may take a moment)...")
    # We use EarlyStopping to finish fast if accuracy stops improving
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )

    model.fit(train_images, train_labels, 
              epochs=10, 
              validation_data=(test_images, test_labels),
              callbacks=[early_stopping],
              verbose=1)

    # ==========================================
    # 4. EXPORT & CONVERT
    # ==========================================
    print("\n[4/4] Exporting to ONNX...")
    
    # A. Export to standard TensorFlow SavedModel folder
    # We use 'model.export' (Keras 3 standard) to get a clean artifact
    export_dir = "final_model_export"
    model.export(export_dir)
    print(f"   - Saved intermediate TensorFlow model to: {export_dir}")

    # B. Run tf2onnx via subprocess (simulates running command in terminal)
    # This takes the folder we just made and converts it
    output_file = "fashion_model.onnx"
    
    command = [
        sys.executable, "-m", "tf2onnx.convert", 
        "--saved-model", export_dir, 
        "--output", output_file,
        "--opset", "13"
    ]
    
    try:
        subprocess.check_call(command)
        print(f"\n✅ SUCCESS! Model saved as '{output_file}'")
        print("   You can now move this file to your website folder.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: Conversion failed. Details: {e}")

if __name__ == "__main__":
    main()