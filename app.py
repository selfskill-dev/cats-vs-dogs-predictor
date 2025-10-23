import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("cats_vs_dogs_model.h5")

st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload an image, and the model will tell you whether it's a cat or a dog.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image
    img = img.resize((160, 160))  # same size used during training
    img_array = np.array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Make prediction
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.success("🐶 It's a Dog!")
    else:
        st.success("🐱 It's a Cat!")
