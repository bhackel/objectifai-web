from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from io import BytesIO
from util import face_aligner
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded', 400
        file = request.files['file']
        if file.filename == '':
            return 'No file selected', 400
        if file:
            image = Image.open(file)
            image = np.array(image)
            aligned_image = face_aligner(image)
            if aligned_image is None:
                return 'Face not detected', 400
            aligned_image = Image.fromarray(aligned_image)
            aligned_image = aligned_image.resize((128, 128))
            # Ensure the image is in RGB format
            if aligned_image.mode != 'RGB':
                aligned_image = aligned_image.convert('RGB')
            aligned_image = np.array(aligned_image)
            aligned_image = aligned_image / 255.0
            # Add batch dimension
            aligned_image = np.expand_dims(aligned_image, axis=0)
            # Load the model
            model = load_model('model_converted.keras')
            # Predict
            prediction = model.predict(aligned_image)
            return str(prediction[0][0])
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5100)
