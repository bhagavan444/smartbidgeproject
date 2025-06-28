from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'model/fresh_rotten_model.h5'
IMG_SIZE = 224

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = load_model(MODEL_PATH)
model.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 3)))  # warm-up

def predict_image(img_path):
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    print(f"🧠 Raw prediction score: {prediction:.4f}")

    if prediction < 0.5:
        label = "Fresh"
        confidence = (1 - prediction) * 100
    else:
        label = "Rotten"
        confidence = prediction * 100

    print(f"🔍 Predicted Label: {label}, Confidence: {confidence:.2f}%")
    return label, round(confidence, 2)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "⚠️ No image uploaded", 400

        file = request.files['image']
        if file.filename == '':
            return "⚠️ Empty file name", 400

        filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        label, confidence = predict_image(file_path)
        return render_template('result.html', label=label, confidence=confidence, image=filename)
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)