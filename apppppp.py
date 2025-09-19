import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# -------------------------------
# 1. Download model if not present
# -------------------------------
FILE_ID = "1QIkDmqTN8sRDDHMpkvek1mpHIr8M87ww"   # your Google Drive file ID
MODEL_NAME = "forest_fire_final.h5"

if not os.path.exists(MODEL_NAME):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(id=FILE_ID, output=MODEL_NAME, quiet=False)

# -------------------------------
# 2. Load the trained model
# -------------------------------
model = tf.keras.models.load_model(MODEL_NAME)

# -------------------------------
# 3. Streamlit App UI
# -------------------------------
st.title("🌲🔥 Forest Fire Detection using CNN")
st.write("Upload an image, and the model will predict whether there is a **Fire** or **No Fire**.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image (same size used in training: 128x128)
    image_resized = image.resize((128, 128))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -------------------------------
    # 4. Make Prediction
    # -------------------------------
    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction, axis=1)[0]  # 0 = No Fire, 1 = Fire
    confidence = np.max(prediction)

    # Debug info
    st.write("Raw prediction probabilities:", prediction)

    # Display result
    if pred_class == 1:
        st.error(f"⚠️ Fire Detected! (Confidence: {confidence:.2f})")
    else:
        st.success(f"✅ No Fire Detected (Confidence: {confidence:.2f})")
