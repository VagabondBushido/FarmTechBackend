from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import io
from flask_cors import CORS
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import logging
import json
import os
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
CORS(app)

# Get port from environment variable or default to 10000
port = int(os.environ.get('PORT', 10000))

def fix_layer_config(layer_config):
    """Fix layer configuration by handling dtype and other compatibility issues"""
    if isinstance(layer_config, dict):
        # Fix InputLayer batch_shape
        if layer_config.get('class_name') == 'InputLayer' and 'config' in layer_config:
            if 'batch_shape' in layer_config['config']:
                shape = layer_config['config']['batch_shape']
                if shape[0] is None:  # Remove batch dimension
                    layer_config['config']['input_shape'] = shape[1:]
                else:
                    layer_config['config']['input_shape'] = shape
                del layer_config['config']['batch_shape']
        
        # Fix dtype configuration
        if 'config' in layer_config and 'dtype' in layer_config['config']:
            dtype_config = layer_config['config']['dtype']
            if isinstance(dtype_config, dict):
                # Convert new format to string
                layer_config['config']['dtype'] = dtype_config.get('config', {}).get('name', 'float32')
        
        # Fix shape configurations in inbound_nodes
        if 'inbound_nodes' in layer_config:
            for node in layer_config['inbound_nodes']:
                if isinstance(node, dict) and 'args' in node:
                    for arg in node['args']:
                        if isinstance(arg, dict) and 'config' in arg:
                            if 'shape' in arg['config']:
                                # Convert shape to list if it's a string
                                if isinstance(arg['config']['shape'], str):
                                    try:
                                        # Parse shape string to list
                                        shape_str = arg['config']['shape'].strip('[]').split(',')
                                        arg['config']['shape'] = [int(s.strip()) if s.strip() != 'None' else None for s in shape_str]
                                    except:
                                        # If parsing fails, use default shape
                                        arg['config']['shape'] = [None, 180, 180, 3]
        
        # Process nested configs
        for key, value in layer_config.items():
            if isinstance(value, dict):
                layer_config[key] = fix_layer_config(value)
            elif isinstance(value, list):
                layer_config[key] = [fix_layer_config(item) if isinstance(item, dict) else item for item in value]
    
    return layer_config

def load_model():
    try:
        # Create model with VGG16 base
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(180, 180, 3))
        
        # Create the full model
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(38, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Load custom weights
        model_path = os.path.join(os.path.dirname(__file__), 'vgg16_model.keras', 'model.weights.h5')
        try:
            model.load_weights(model_path, by_name=True, skip_mismatch=True)
            logger.info(f"Successfully loaded custom layer weights from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load custom weights: {str(e)}")
            logger.info("Continuing with ImageNet weights only")
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Load the trained model
try:
    model = load_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def preprocess_image(image_url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(image_url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch image: {response.status_code}")
    img = Image.open(io.BytesIO(response.content))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((180, 180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received request")
        data = request.json
        print("Request data:", data)
        image_url = data.get("image_url")
        print("Image URL:", image_url)
        if not image_url:
            return jsonify({"error": "No image URL provided"}), 400
        input_image = preprocess_image(image_url)
        print("Image preprocessed")
        predictions = model.predict(input_image)
        print("Prediction done")
        predicted_class_index = int(np.argmax(predictions, axis=1)[0])
        predicted_class = class_names[predicted_class_index]
        predicted_probability = float(predictions[0][predicted_class_index])
        result = {
            "predicted_class_index": predicted_class_index,
            "predicted_class": predicted_class,
            "confidence": predicted_probability
        }
        print("Returning result:", result)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/test-urls', methods=['POST'])
def test_urls():
    """Test API with multiple URLs to verify different results"""
    try:
        data = request.json
        urls = data.get("urls", [])
        
        if not urls:
            return jsonify({"error": "No URLs provided"}), 400
            
        results = []
        for url in urls:
            # Process image
            input_image = preprocess_image(url)
            
            # Predict
            predictions = model.predict(input_image)
            class_index = np.argmax(predictions[0])
            
            # Get results with both indexing systems
            zero_based_crop, zero_based_disease = CLASS_LABELS.get(class_index, ("Unknown", "Unknown"))
            one_based_crop, one_based_disease = CLASS_LABELS_1_BASED.get(class_index + 1, ("Unknown", "Unknown"))
            
            results.append({
                "url": url,
                "zero_based_result": {
                    "class": int(class_index),
                    "crop": zero_based_crop,
                    "disease": zero_based_disease,
                    "confidence": float(predictions[0][class_index])
                },
                "one_based_result": {
                    "class": int(class_index) + 1,
                    "crop": one_based_crop,
                    "disease": one_based_disease,
                    "confidence": float(predictions[0][class_index])
                }
            })
            
        return jsonify({"results": results})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None,
        "class_labels": {
            "zero_based": "0-37",
            "one_based": "1-38",
            "tomato_healthy_zero_based": 36,
            "tomato_healthy_one_based": 37
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)