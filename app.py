import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🧠 MNIST Digit Classifier")

# Load model
model = tf.keras.models.load_model("mnist_model.h5")

uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L").resize((28, 28))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    prediction = model.predict(img_array)
    st.write(f"### Predicted Digit: {np.argmax(prediction)}")
