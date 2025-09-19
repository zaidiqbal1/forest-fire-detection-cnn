import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Google Drive File ID of your model
FILE_ID = "1IjUckCLBs7nAp3LSrYpNb36UEeQkq4kl"   # your file id
MODEL_NAME = "forest_fire_final.h5"            # updated model name

# Direct download URL
download_url = f"https://drive.google.com/uc?id={FILE_ID}&confirm=t"

# Download model if not exists
if not os.path.exists(MODEL_NAME):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(download_url, MODEL_NAME, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_NAME)

# Streamlit UI
st.title("🌲🔥 Forest Fire Detection")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize to match training input size
    image_resized = image.resize((128, 128))  # change (128,128) if your model used another size
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    # Assuming binary classification: Fire vs No Fire
    if prediction[0][0] > 0.5:
        st.error("⚠️ Fire Detected!")
    else:
        st.success("✅ No Fire Detected")
