import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Use the new Google Drive File ID based on the link you gave
FILE_ID = "1QIkDmqTN8sRDDHMpkvek1mpHIr8M87ww"
MODEL_NAME = "forest_fire_final.h5"

# Download model if not already in the working directory
if not os.path.exists(MODEL_NAME):
    with st.spinner("Downloading model from Google Drive..."):
        # Let gdown handle with file ID directly
        gdown.download(id=FILE_ID, output=MODEL_NAME, quiet=False)

# Load the model
model = tf.keras.models.load_model(MODEL_NAME)

# Streamlit UI
st.title("🌲🔥 Forest Fire Detection")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess same way as training
    image_resized = image.resize((128, 128))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    # Adjust threshold if needed
    if prediction[0][0] > 0.5:
        st.error("⚠️ Fire Detected!")
    else:
        st.success("✅ No Fire Detected")
