from flask import Flask, request, render_template, url_for
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import os
import uuid
import random
from util import face_aligner

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded. Please upload an image.")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected. Please select an image.")
        if file:
            try:
                # Generate a unique filename
                filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
                filepath = os.path.join(UPLOAD_FOLDER, filename)

                # Save the file
                file.save(filepath)

                image = Image.open(filepath)
                image = np.array(image)
                aligned_image = face_aligner(image)
                if aligned_image is None:
                    os.remove(filepath)  # Remove file if face not detected
                    raise ValueError("Face not detected in the image. Please try again with a clear image of your face.")
                aligned_image = Image.fromarray(aligned_image)
                aligned_image = aligned_image.resize((128, 128))
                if aligned_image.mode != 'RGB':
                    aligned_image = aligned_image.convert('RGB')
                aligned_image = np.array(aligned_image)
                aligned_image = aligned_image / 255.0
                aligned_image = np.expand_dims(aligned_image, axis=0)
                model = load_model('model_converted.keras')
                prediction = model.predict(aligned_image)
                rating = prediction[0][0]

                # Convert the rating to a 0-10 scale
                rating = round(rating * 10, 2)

                # Determine the Harry Potter House based on the rating
                if rating >= 8.5:
                    house = "Ravenclaw"
                elif rating >= 5:
                    house = "Gryffindor"
                else:
                    house = "Hufflepuff" if random.choice([0, 1]) else "Slytherin"

                image_url = url_for('static', filename='uploads/' + filename)
                response = render_template('result.html', rating=rating, house=house, image_url=image_url)

                # Delete the image after rendering the response
                os.remove(filepath)

                return response
            except Exception as e:
                return render_template('index.html', error=str(e))
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5100)
