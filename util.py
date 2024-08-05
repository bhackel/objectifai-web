import numpy as np
from skimage import transform
import face_recognition

# Function to rotate and align the given image (source: carykh)
def face_aligner(image):
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
    if not (len(face_landmarks) == 1 and colorAmount >= 0.04):
        print(f"Alignment failed: grayscale: {colorAmount >= 0.04}, or face count: {len(face_landmarks)}")
        return None

    leftEyePosition = get_avg(face_landmarks[0], 'left_eye')
    rightEyePosition = get_avg(face_landmarks[0], 'right_eye')
    nosePosition = get_avg(face_landmarks[0], 'nose_tip')
    mouthPosition = get_avg(face_landmarks[0], 'bottom_lip')

    centralPosition = (leftEyePosition + rightEyePosition) / 2

    faceWidth = np.linalg.norm(leftEyePosition - rightEyePosition)
    faceHeight = np.linalg.norm(centralPosition - mouthPosition)

    # Check face dimensions
    if not (faceHeight * 0.7 <= faceWidth <= faceHeight * 1.5):
        print("Alignment failed: face dimensions")
        return None

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

    return outputArr