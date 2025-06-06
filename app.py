import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Determine if we're running on Railway
IS_RAILWAY = os.environ.get('RAILWAY', 'false').lower() == 'true'

# Set model path based on environment
if IS_RAILWAY:
    MODEL_PATH = '/data/vgg16_model.keras'
else:
    MODEL_PATH = 'vgg16_model.keras'

# Load the model
try:
    model = load_model(MODEL_PATH, compile=False)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

def load_image_from_url(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image_url' not in data:
            return jsonify({"error": "No image URL provided"}), 400

        image_url = data['image_url']
        logger.info(f"Processing image from URL: {image_url}")

        # Load and preprocess the image
        img_array = load_image_from_url(image_url)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])

        # Map class index to label (update these based on your model's classes)
        class_labels = ['class1', 'class2', 'class3']  # Replace with your actual class labels
        predicted_label = class_labels[predicted_class]

        return jsonify({
            "prediction": predicted_label,
            "confidence": confidence,
            "success": True
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)