import os
import numpy as np
import tensorflow as tf
import cv2
from dotenv import load_dotenv
from model import build_autoencoder
from tensorflow.keras.callbacks import ModelCheckpoint

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Paths from environment variables
DATA_DIR = os.getenv('DATA_DIR')
MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'outputs')  # Default fallback ke 'outputs'


# Function to load and preprocess images
def load_images(folder, low_res_size=(128, 128), high_res_size=(512, 512)):
    low_res, high_res = [], []
    for filename in os.listdir(folder):
        if filename.endswith(".bmp"):
            img = cv2.imread(os.path.join(folder, filename))
            img_high = cv2.resize(img, high_res_size)
            img_low = cv2.resize(img, low_res_size)

            # Normalize
            img_high = img_high / 255.0
            img_low = img_low / 255.0

            low_res.append(img_low)
            high_res.append(img_high)

    return np.array(low_res), np.array(high_res)

# Function to generate high-resolution image from low-resolution input
def generate_high_res(model, low_res_image, output_path):
    # Predict high-resolution image
    high_res_pred = model.predict(np.expand_dims(low_res_image, axis=0))[0]
    
    # Denormalize
    high_res_pred = high_res_pred * 255.0
    
    # Convert to uint8 for saving
    high_res_pred = np.clip(high_res_pred, 0, 255).astype(np.uint8)
    
    # Save the output image
    cv2.imwrite(output_path, high_res_pred)

# Load training data
X_train, Y_train = load_images(DATA_DIR)

# Build and compile the autoencoder model
model = build_autoencoder()
model.compile(optimizer='adam', loss='mse')

# Set up model checkpointing
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, verbose=1)

# Train the model
model.fit(X_train, Y_train, epochs=250, batch_size=8, validation_split=0.2, callbacks=[checkpoint])

# Save model after training
model.save(MODEL_SAVE_PATH)

# Generate high-resolution images from the low-resolution inputs after training
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for i, low_res_image in enumerate(X_train):
    output_path = os.path.join(OUTPUT_DIR, f'output_image_{i}.bmp')
    generate_high_res(model, low_res_image, output_path)

print("Output images have been saved.")
