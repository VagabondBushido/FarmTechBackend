from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Get the absolute path to the model directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vgg16_model.keras')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.weights.h5')

logger.info(f"Looking for model at: {MODEL_PATH}")

# Load model
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH, compile=False)
        logger.info("Model loaded successfully")
    else:
        logger.error(f"Model file not found at {MODEL_PATH}")
        model = None
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

def load_image(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        return img_array
    except requests.exceptions.Timeout:
        logger.error("Timeout while fetching image")
        raise Exception("Timeout while fetching image")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching image: {str(e)}")
        raise Exception(f"Error fetching image: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        logger.error("Model not loaded")
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        if not data or 'image_url' not in data:
            logger.error("No image URL provided")
            return jsonify({"error": "No image URL provided"}), 400

        # Load and preprocess image
        img_array = load_image(data['image_url'])
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])

        logger.info(f"Prediction successful: class {predicted_class} with confidence {confidence}")
        return jsonify({
            "prediction": int(predicted_class),
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
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)