import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Google Drive File ID of your model
FILE_ID = "1IjUckCLBs7nAp3LSrYpNb36UEeQk4kl"
MODEL_NAME = "forest_fire_from_drive.h5"  # local name to save model

# URL to download the model
download_url = f"https://drive.google.com/uc?id={FILE_ID}&export=download"

# Download model if not present
if not os.path.exists(MODEL_NAME):
    with st.spinner("Downloading model..."):
        gdown.download(download_url, MODEL_NAME, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_NAME)

# Streamlit UI
st.title("🌲🔥 Forest Fire Detection")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image (resize, normalize, etc) — adjust size if your model expects a different input
    image_resized = image.resize((128, 128))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    # Assuming binary classification, threshold at 0.5
    if prediction[0][0] > 0.5:
        st.error("⚠️ Fire Detected!")
    else:
        st.success("✅ No Fire Detected")
