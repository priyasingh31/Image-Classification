from flask import Flask, render_template, request, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__, static_url_path='/static')

# Load the pre-trained model
model = tf.keras.models.load_model('image_classifier.h5')

# Define class names
class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    # Save the uploaded image
    uploads_folder = 'static/uploads'
    if not os.path.exists(uploads_folder):
        os.makedirs(uploads_folder)

    file_path = os.path.join(uploads_folder, file.filename)
    file.save(file_path)

    # Load and preprocess the image
    img = image.load_img(file_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array / 255.0, axis=0)

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Render the result page
    return render_template('result.html', file=file, image_path=url_for('static', filename='uploads/' + file.filename), prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)