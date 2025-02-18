import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import streamlit as st

@st.cache_resource
def load_trained_model():
    """Loads the saved Baseline CNN model"""
    return load_model("models/Baseline_CNN.keras")

def preprocess_image(image):
    """Resize and normalize the image."""
    img = image.resize((130, 130))  # Ensure it matches the input size of the model
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def make_prediction(model, image):
    """Predicts malaria presence in the image."""
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    return prediction[0][0]  # Return probability score
