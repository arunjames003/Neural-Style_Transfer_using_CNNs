import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

# Function to load and preprocess image
def load_img(image_file):
    img = Image.open(image_file)
    img = img.resize((256, 256))  # Resize for consistency
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)  # Ensure float32 type
    return img

# Function to deprocess image
def deprocess_img(processed_img):
    x = np.squeeze(processed_img, axis=0)
    x = (x * 255).astype(np.uint8)
    return x

# Load the model
@st.cache_resource
def load_model():
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Streamlit app
st.title("Neural Style Transfer using CNNs")
st.markdown('Built by **Arun James** ')

st.write("""
Neural style transfer merges the content of one image with the artistic style of another using convolutional neural networks (CNNs). 
""")

st.write('TensorFlow Hub Model Link-  https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

st.divider()

st.write("Upload the content and style images for style transfer.")

# Upload the content image
content_image = st.file_uploader("Choose a Content Image", type=["jpg", "jpeg", "png"])
if content_image is not None:
    # Display the content image
    st.image(content_image, caption='Content Image', width=300)

# Upload the style image
style_image = st.file_uploader("Choose a Style Image", type=["jpg", "jpeg", "png"])
if style_image is not None:
    # Display the style image
    st.image(style_image, caption='Style Image', width=300)

st.divider()

if content_image and style_image:
    is_clicked = st.button('Generate Image', type='primary')
    if is_clicked:
        st.write("Running style transfer... This may take a few seconds.")
        
        content_img = load_img(content_image)
        style_img = load_img(style_image)
        
        model = load_model()
        stylized_img = model(tf.constant(content_img), tf.constant(style_img))[0]
        
        output_image = deprocess_img(stylized_img)
        
        st.image(output_image, caption='Output Image', width=300)
        st.write("Style transfer completed.")
