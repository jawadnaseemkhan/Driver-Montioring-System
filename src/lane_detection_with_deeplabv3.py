import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load pre-trained DeepLabV3
model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (512, 512))  # Resize to 512x512 for DeepLab
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_lanes(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    mask = np.argmax(prediction[0], axis=-1)  # Get segmentation mask
    return mask

# Loop through multiple images
image_dir = "/lhome/jawakha/Desktop/Project/data/TUSimple/test_set/clips"
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith(".jpg"):
            image_path = os.path.join(root, file)
            mask = predict_lanes(image_path)
            plt.imshow(mask)
            plt.show()
