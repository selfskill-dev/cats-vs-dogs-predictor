import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# App configuration
st.set_page_config(page_title="🐱🐶 Cat vs Dog Classifier", layout="centered")

st.title("🐶🐱 Cat vs Dog Image Classifier")
st.markdown("### Upload an image, and the AI will tell you if it's a **Cat** or a **Dog**!")

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("cats_vs_dogs_model.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {e}")
        st.stop()

model = load_model()

uploaded_file = st.file_uploader("📤 Upload a cat or dog image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing... Please wait ⏳"):
            try:
                # Resize and preprocess
                img = image.resize((160, 160))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

                # Prediction
                preds = model.predict(img_array)
                pred = preds[0][0]

                label = "🐶 Dog" if pred > 0.5 else "🐱 Cat"
                confidence = round(pred * 100, 2) if pred > 0.5 else round((1 - pred) * 100, 2)

                st.success(f"### ✅ Prediction: {label}")
                st.write(f"**Confidence:** {confidence}%")
            except Exception as e:
                st.error(f"⚠️ Prediction error: {e}")
else:
    st.info("👆 Please upload an image to start prediction.")
