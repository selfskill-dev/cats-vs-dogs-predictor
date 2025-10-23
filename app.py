import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(page_title="Cat vs Dog Classifier 🐱🐶", layout="centered")

# Header
st.title("🐶🐱 Cat vs Dog Image Classifier")
st.markdown("### Upload an image, and the AI will predict whether it's a **Cat** or a **Dog**!")

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('cats_vs_dogs_prediction.h5 ')
    return model

model = load_model()

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image of a Cat or Dog", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image preview
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Uploaded Image", use_container_width=True)

    # Add Predict button
    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image... Please wait ⏳"):
            # Preprocess image
            img = img.resize((150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Predict
            prediction = model.predict(img_array)[0][0]
            confidence = round(prediction * 100, 2) if prediction > 0.5 else round((1 - prediction) * 100, 2)
            label = "🐶 Dog" if prediction > 0.5 else "🐱 Cat"

            # Display result
            st.success(f"### ✅ Prediction: {label}")
            st.write(f"**Confidence:** {confidence}%")
else:
    st.info("👆 Please upload an image to start prediction.")

