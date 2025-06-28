from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)  # ✅ Corrected from _name_
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'model/fresh_rotten_model.h5'
IMG_SIZE = 224

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = load_model(MODEL_PATH)

def predict_image(img_path):
    """Preprocess image and predict using the model."""
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]

    # Adjust labels based on binary classification
    if prediction < 0.5:
        label = "Fresh / Healthy"
        confidence = (1 - prediction) * 100
    else:
        label = "Rotten"
        confidence = prediction * 100

    return label, round(confidence, 2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "Empty filename", 400

    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Predict the result
    label, confidence = predict_image(file_path)
    return render_template('result.html', label=label, confidence=confidence, image=file.filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':  # ✅ Corrected from _main_
    app.run(debug=True)
