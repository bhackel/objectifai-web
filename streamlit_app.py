import streamlit as st
from tensorflow.keras.models import load_model as tf_load_model
import numpy as np
from PIL import Image
import face_recognition
from skimage import transform


# Function to rotate and align the given image (source: carykh)
def face_aligner(image):
    IMAGE_EXTENSIONS = ['.png', '.jpg']
    OUTPUT_EXTENSION = '.png'

    DESIRED_X = 64
    DESIRED_Y = 42
    DESIRED_SIZE = 48

    FINAL_IMAGE_WIDTH = 128
    FINAL_IMAGE_HEIGHT = 128

    def get_avg(face, landmark):
        cum = np.zeros(2)
        for point in face[landmark]:
            cum[0] += point[0]
            cum[1] += point[1]
        return cum / len(face[landmark])


    def get_norm(a):
        return (a - np.mean(a)) / np.std(a)

    image_numpy = image
    face_landmarks = face_recognition.face_landmarks(image_numpy)
    colorAmount = 0
    imageSaved = False
    if len(image_numpy.shape) == 3:
        nR = get_norm(image_numpy[:, :, 0])
        nG = get_norm(image_numpy[:, :, 1])
        nB = get_norm(image_numpy[:, :, 2])
        colorAmount = np.mean(np.square(nR - nG)) + np.mean(np.square(nR - nB)) + np.mean(np.square(nG - nB))
    # We need there to only be one face in the image, AND we need it to be a colored image.
    if len(face_landmarks) == 1 and colorAmount >= 0.04:
        leftEyePosition = get_avg(face_landmarks[0], 'left_eye')
        rightEyePosition = get_avg(face_landmarks[0], 'right_eye')
        nosePosition = get_avg(face_landmarks[0], 'nose_tip')
        mouthPosition = get_avg(face_landmarks[0], 'bottom_lip')

        centralPosition = (leftEyePosition + rightEyePosition) / 2

        faceWidth = np.linalg.norm(leftEyePosition - rightEyePosition)
        faceHeight = np.linalg.norm(centralPosition - mouthPosition)
        if faceHeight * 0.7 <= faceWidth <= faceHeight * 1.5:
            faceSize = (faceWidth + faceHeight) / 2

            toScaleFactor = faceSize / DESIRED_SIZE
            toXShift = (centralPosition[0])
            toYShift = (centralPosition[1])
            toRotateFactor = np.arctan2(rightEyePosition[1] - leftEyePosition[1],
                                        rightEyePosition[0] - leftEyePosition[0])

            rotateT = transform.SimilarityTransform(scale=toScaleFactor, rotation=toRotateFactor,
                                                    translation=(toXShift, toYShift))
            moveT = transform.SimilarityTransform(scale=1, rotation=0, translation=(-DESIRED_X, -DESIRED_Y))

            outputArr = transform.warp(image=image_numpy, inverse_map=(moveT + rotateT))[0:FINAL_IMAGE_HEIGHT,
                        0:FINAL_IMAGE_WIDTH]

            outputArr = (outputArr*255).astype(np.uint8)

            imageSaved = True
    if imageSaved:
        print("Aligned image successfully!")
        return outputArr
    else:
        print("Face image alignment failed. Either the image is grayscale, has no face, or the ratio of eye distance to "
              "mouth distance isn't close enough to 1.")
        return None


# Load your pre-trained TensorFlow model
# Replace this with loading your own model
@st.cache_data
def load_model():
    model = tf_load_model('model')  # Replace with your model loading code
    return model

# Function to process the image using your model
def process_image(image, model):
    # Align the face into the center
    image = np.array(image)
    image = face_aligner(image)
    # Normalize pixel values
    image = image / 255.0
    # Reshape the array to (1, 128, 128, 3) for compatibility
    image = np.expand_dims(image, axis=0)
    # get rating as a decimal from 0 to 1, convert to 0 to 10
    rating = model(image)
    rating = float(rating.numpy()[0])
    rating = round(rating*10, 2)

    return rating

# Streamlit app code
def main():
    st.title('Objectif.ai')

    uploaded_file = st.file_uploader("Choose an image or take a picture", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load the model
        model = load_model()

        # Process the image and get the output value
        image = Image.open(uploaded_file)
        output_value = process_image(image, model)

        # Display the output rating
        st.markdown(f"<h1 style='text-align: center;'>Objective Rating: {round(output_value, 1)}</h1>", unsafe_allow_html=True)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)




if __name__ == '__main__':
    main()
