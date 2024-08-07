from flask import Flask, request, render_template, url_for
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
import os
import uuid
import random
import magic  # Library for MIME type checking
from util import face_aligner
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define the maximum width and height for the uploaded images
MAX_WIDTH = 400
MAX_HEIGHT = 400

# Define the maximum file age (in minutes) before deletion
MAX_FILE_AGE_MINUTES = 5

# Define the maximum file size in bytes (e.g., 5 MB)
MAX_FILE_SIZE = 5 * 1024 * 1024

# Allowed MIME types for image files
ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png', 'image/gif', 'image/bmp'}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded. Please upload an image.")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected. Please select an image.")
        if file:
            # Check the file size
            file.seek(0, os.SEEK_END)  # Move to the end of the file
            file_length = file.tell()  # Get the file size
            file.seek(0)  # Reset file pointer to the beginning

            if file_length > MAX_FILE_SIZE:
                return render_template('index.html', error="File size exceeds the limit of 5 MB.")

            # Check the MIME type of the file
            mime = magic.Magic(mime=True)
            mime_type = mime.from_buffer(file.read(1024))
            file.seek(0)  # Reset file pointer to the beginning

            if mime_type not in ALLOWED_MIME_TYPES:
                return render_template('index.html', error="Invalid file type. Please upload a valid image.")

            # Generate a unique filename
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            # Open the image
            image = Image.open(file)

            # Apply rotation tag if it exists
            image = ImageOps.exif_transpose(image)

            # Resize image while maintaining aspect ratio
            image.thumbnail((MAX_WIDTH, MAX_HEIGHT))

            # Save the resized image
            image.save(filepath)

            # Convert image to RGB (if not already in RGB mode)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Convert image to NumPy array and remove alpha channel if present
            image_np = np.array(image)[:, :, :3]

            # Align the face into the center
            aligned_image = face_aligner(image_np)
            if aligned_image is None:
                os.remove(filepath)  # Remove file if face not detected
                return render_template('index.html', error="Face not detected, please try again")

            # Resize aligned image for model input
            aligned_image = Image.fromarray(aligned_image).resize((128, 128))

            # Normalize and prepare image for model
            aligned_image = np.array(aligned_image) / 255.0
            aligned_image = np.expand_dims(aligned_image, axis=0)

            # Load model and make prediction
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
            # os.remove(filepath)

            return response
    return render_template('index.html')

def delete_old_files():
    """Deletes files older than MAX_FILE_AGE_MINUTES."""
    now = datetime.now()
    cutoff = now - timedelta(minutes=MAX_FILE_AGE_MINUTES)
    for filename in os.listdir(UPLOAD_FOLDER):
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(filepath):
            file_creation_time = datetime.fromtimestamp(os.path.getctime(filepath))
            if file_creation_time < cutoff:
                os.remove(filepath)
                print(f"Deleted old file: {filename}")

# Set up the background scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(func=delete_old_files, trigger="interval", minutes=1)
scheduler.start()

# Shut down the scheduler when exiting the app
import atexit
atexit.register(lambda: scheduler.shutdown())

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5100)
