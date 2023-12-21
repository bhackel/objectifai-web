import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your pre-trained TensorFlow model
# Replace this with loading your own model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model')  # Replace with your model loading code
    return model

# Function to process the image using your model
def process_image(image, model):
    # Preprocess the image for your model (resize, normalize, etc.)
    # Replace this with your image preprocessing code
    image = np.array(image)
    # Normalize
    image = image / 255.0
    # Reshape the array to (1, 128, 128, 3)
    image = np.expand_dims(image, axis=0)
    # get rating as a decimal from 0 to 1, convert to 0 to 10
    rating = model(image)
    rating = float(rating.numpy()[0])
    rating = round(rating*10, 2)

    return rating

# Streamlit app code
def main():
    st.title('Image ML App')
    st.write('Upload an image')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        model = load_model()  # Load the model

        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Process the image and get the output value
        output_value = process_image(image, model)

        # Display the output value
        st.write('Output Value:', round(output_value, 1))

if __name__ == '__main__':
    main()
