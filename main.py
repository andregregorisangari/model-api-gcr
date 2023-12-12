import os
import numpy as np
import gdown
from PIL import Image
from flask import Flask, jsonify, request
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as tf_image


url = "https://drive.google.com/uc?id=1QAsgzqY-62pWoAh6x4B7vDLOSZ1H0tXG"
output = 'model-kopra.h5'
gdown.download(url, output)

load_dotenv()
app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MODEL_KOPRA'] = 'model-kopra.h5'
model_copra = load_model(app.config['MODEL_KOPRA'], compile=False)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def predict_grade(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((150, 150))  # Sesuaikan dengan ukuran input model Anda
    x = tf_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    result = model_copra.predict(x)[0]

    # Ambil kelas dengan probabilitas tertinggi sebagai prediksi
    predicted_class = np.argmax(result)

    # Mapping kelas numerik ke label Grade A, B, atau C
    class_mapping = {0: 'A', 1: 'B', 2: 'C'}
    predicted_label = class_mapping[predicted_class]

    # Ambil probabilitas prediksi
    confidence = float(result[np.argmax(result)].item())

    return predicted_label, confidence

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': {
            'code': 200,
            'message': 'Hello World!'
        }
    }), 200

@app.route('/predict', methods=['POST'])
def predict_grade_endpoint():
    if request.method == 'POST':
        req_image = request.files['image']
        if req_image and allowed_file(req_image.filename):
            filename = secure_filename(req_image.filename)
            req_image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Prediksi Grade menggunakan fungsi predict_grade
            predicted_grade, confidence = predict_grade(image_path)

            return jsonify({
                'status': {
                    'code': 200,
                    'message': 'Success predicting',
                    'data': {'class': predicted_grade, 'confidence': confidence}
                }
            }), 200
        else:
            return jsonify({
                'status': {
                    'code': 400,
                    'message': 'Invalid file format. Please upload a JPG, JPEG, or PNG image.'
                }
            }), 400
    else:
        return jsonify({
            'status': {
                'code': 405,
                'message': 'Method not allowed'
            }
        }), 405

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))