# Extract files from zipped folder

import zipfile
import os

dataset_file_path = 'dataset.zip'
unidentified_file_path = "unidentified.zip"

with zipfile.ZipFile(dataset_file_path, 'r') as zip_ref:
    zip_ref.extractall('dataset')
with zipfile.ZipFile(unidentified_file_path, 'r') as zip_ref:
    zip_ref.extractall('unidentified')

# List the extracted files to verify
extracted_files = os.listdir('dataset') + os.listdir('unidentified')
print(extracted_files)



# Load images. Preprocessing not necessary since it's all 32x32 px at 256 colors.

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define directories
train_dir = 'dataset/train'
validation_dir = 'dataset/validation'
test_dir = 'dataset/test'

# Image data generator for preprocessing
datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),
    batch_size=16,
    class_mode='binary'
)

validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(32, 32),
    batch_size=16,
    class_mode='binary'
)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(32, 32),
    batch_size=16,
    class_mode='binary'
)



# Build and compile the model

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model

model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)



# Predict cases using the new model!

import numpy as np
from PIL import Image

def predict_image(model, image_path):
    image = Image.open(image_path)
    image = image.resize((32, 32))  # Resize the image to match the input shape
    image = np.array(image) / 255.0  # Normalize pixel values

    # Ensure the image has 3 channels
    if image.ndim == 2:  # If grayscale, convert to RGB
        image = np.stack((image,) * 3, axis=-1)
    elif image.shape[-1] != 3:  # If not RGB or grayscale, raise an error
        raise ValueError(f"Image has unexpected number of channels: {image.shape[-1]}")

    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    return "Duck" if prediction[0][0] > 0.5 else "Rabbit"

# Example usage
for file in os.listdir('unidentified'):
    print(file, predict_image(model, 'unidentified/' + file))