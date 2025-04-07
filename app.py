from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import io
import logging
import os
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load the trained model
MODEL_PATH = "model/model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info(f"Model loaded successfully: {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# Class labels with crop names and diseases - EXPANDED to include ALL 38 classes
CLASS_LABELS = {
    0: ("Apple", "Apple Scab"),
    1: ("Apple", "Black Rot"),
    2: ("Apple", "Cedar Apple Rust"),
    3: ("Apple", "Healthy"),
    4: ("Blueberry", "Healthy"),
    5: ("Cherry", "Powdery Mildew"),
    6: ("Cherry", "Healthy"),
    7: ("Corn", "Cercospora Leaf Spot"),
    8: ("Corn", "Common Rust"),
    9: ("Corn", "Northern Leaf Blight"),
    10: ("Corn", "Healthy"),
    11: ("Grape", "Black Rot"),
    12: ("Grape", "Esca (Black Measles)"),
    13: ("Grape", "Leaf Blight"),
    14: ("Grape", "Healthy"),
    15: ("Orange", "Huanglongbing (Citrus Greening)"),
    16: ("Peach", "Bacterial Spot"),
    17: ("Peach", "Healthy"),
    18: ("Pepper", "Bacterial Spot"),
    19: ("Pepper", "Healthy"),
    20: ("Potato", "Early Blight"),
    21: ("Potato", "Late Blight"),
    22: ("Potato", "Healthy"),
    23: ("Raspberry", "Healthy"),
    24: ("Soybean", "Healthy"),
    25: ("Squash", "Powdery Mildew"),
    26: ("Strawberry", "Leaf Scorch"),
    27: ("Strawberry", "Healthy"),
    28: ("Tomato", "Bacterial Spot"),
    29: ("Tomato", "Early Blight"),
    30: ("Tomato", "Leaf Mold"),
    31: ("Tomato", "Septoria Leaf Spot"),
    32: ("Tomato", "Spider Mites"),
    33: ("Tomato", "Target Spot"),
    34: ("Tomato", "Yellow Leaf Curl Virus"),
    35: ("Tomato", "Mosaic Virus"),
    36: ("Tomato", "Healthy"),
    37: ("Background", "No Plant")  # Add missing class 37
}

# Create a reversed version for indexing check
CLASS_LABELS_1_BASED = {
    1: ("Apple", "Apple Scab"),
    2: ("Apple", "Black Rot"),
    3: ("Apple", "Cedar Apple Rust"),
    4: ("Apple", "Healthy"),
    5: ("Blueberry", "Healthy"),
    6: ("Cherry", "Powdery Mildew"),
    7: ("Cherry", "Healthy"),
    8: ("Corn", "Cercospora Leaf Spot"),
    9: ("Corn", "Common Rust"),
    10: ("Corn", "Northern Leaf Blight"),
    11: ("Corn", "Healthy"),
    12: ("Grape", "Black Rot"),
    13: ("Grape", "Esca (Black Measles)"),
    14: ("Grape", "Leaf Blight"),
    15: ("Grape", "Healthy"),
    16: ("Orange", "Huanglongbing (Citrus Greening)"),
    17: ("Peach", "Bacterial Spot"),
    18: ("Peach", "Healthy"),
    19: ("Pepper", "Bacterial Spot"),
    20: ("Pepper", "Healthy"),
    21: ("Potato", "Early Blight"),
    22: ("Potato", "Late Blight"),
    23: ("Potato", "Healthy"),
    24: ("Raspberry", "Healthy"),
    25: ("Soybean", "Healthy"),
    26: ("Squash", "Powdery Mildew"),
    27: ("Strawberry", "Leaf Scorch"),
    28: ("Strawberry", "Healthy"),
    29: ("Tomato", "Bacterial Spot"),
    30: ("Tomato", "Early Blight"),
    31: ("Tomato", "Leaf Mold"),
    32: ("Tomato", "Septoria Leaf Spot"),
    33: ("Tomato", "Spider Mites"),
    34: ("Tomato", "Target Spot"),
    35: ("Tomato", "Yellow Leaf Curl Virus"),
    36: ("Tomato", "Mosaic Virus"),
    37: ("Tomato", "Healthy"),
    38: ("Background", "No Plant")
}

def preprocess_image(image_url, save_debug=True):
    """
    Preprocess image to match the model's exact preprocessing
    """
    try:
        # Download image from URL
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))
        
        # Save original image for debugging
        if save_debug:
            if not os.path.exists('debug'):
                os.makedirs('debug')
            image.save(f"debug/original_image.jpg")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to match model input shape
        image = image.resize((224, 224))

        
        # Save resized image for debugging
        if save_debug:
            image.save(f"debug/resized_image.jpg")
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # IMPORTANT: Use tf.keras.applications.vgg16.preprocess_input for VGG16
        # This matches the preprocessing in your model's layers
        from tensorflow.keras.applications.vgg16 import preprocess_input
        processed_image = preprocess_input(image_array.copy())
        
        # Add batch dimension
        batched = np.expand_dims(processed_image, axis=0)
        
        return batched
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_url = data.get("image_url")
        
        if not image_url:
            return jsonify({"error": "No image URL provided"}), 400
        
        logger.info(f"Received prediction request for image: {image_url}")
        
        # Preprocess image using VGG16's preprocessing
        input_image = preprocess_image(image_url)
        
        # Make prediction
        logger.info("Making prediction...")
        predictions = model.predict(input_image)
        
        # Log raw predictions for debugging
        logger.info(f"Raw prediction shape: {predictions.shape}")
        logger.info(f"Full prediction array: {predictions[0]}")
        
        # Get top class and confidence
        class_index = np.argmax(predictions[0])
        logger.info(f"Raw predicted class index: {class_index}")
        
        # Check both 0-based and 1-based indexing results
        zero_based_crop, zero_based_disease = CLASS_LABELS.get(class_index, ("Unknown", "Unknown"))
        one_based_crop, one_based_disease = CLASS_LABELS_1_BASED.get(class_index + 1, ("Unknown", "Unknown"))
        
        logger.info(f"0-based indexing result: Crop={zero_based_crop}, Disease={zero_based_disease}")
        logger.info(f"1-based indexing result: Crop={one_based_crop}, Disease={one_based_disease}")
        
        # Check if tomato healthy has high probability
        tomato_healthy_0_based_prob = float(predictions[0][36]) if predictions.shape[1] > 36 else 0
        tomato_healthy_1_based_prob = float(predictions[0][36]) if predictions.shape[1] > 36 else 0
        logger.info(f"Tomato Healthy (class 36) probability: {tomato_healthy_0_based_prob}")
        logger.info(f"Tomato Healthy (class 37 adjusted) probability: {tomato_healthy_1_based_prob}")
        
        # Get top 5 indices for debugging
        top_indices = np.argsort(predictions[0])[-5:][::-1]
        logger.info(f"Top 5 prediction indices: {top_indices}")
        logger.info(f"Top 5 prediction values: {[predictions[0][i] for i in top_indices]}")
        
        # Return results for both indexing methods
        confidence = float(predictions[0][class_index])
        
        # Get top 3 predictions with both indexing systems
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_zero_based = [
            {
                "class_index": int(i),
                "crop": CLASS_LABELS.get(i, ("Unknown", "Unknown"))[0],
                "disease": CLASS_LABELS.get(i, ("Unknown", "Unknown"))[1],
                "probability": float(predictions[0][i])
            }
            for i in top_3_indices
        ]
        
        top_3_one_based = [
            {
                "class_index": int(i+1),  # Adjusted to 1-based
                "crop": CLASS_LABELS_1_BASED.get(i+1, ("Unknown", "Unknown"))[0],
                "disease": CLASS_LABELS_1_BASED.get(i+1, ("Unknown", "Unknown"))[1],
                "probability": float(predictions[0][i])
            }
            for i in top_3_indices
        ]
        
        result = {
            "original_class_index": int(class_index),
            "zero_based_result": {
                "class": int(class_index),
                "crop": zero_based_crop,
                "disease": zero_based_disease,
                "confidence": confidence,
                "top_3_predictions": top_3_zero_based
            },
            "one_based_result": {
                "class": int(class_index) + 1,  # Adjusted to 1-based
                "crop": one_based_crop,
                "disease": one_based_disease,
                "confidence": confidence,
                "top_3_predictions": top_3_one_based
            }
        }
        
        logger.info(f"Prediction results: {result}")
        
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
            input_image = preprocess_image(url, save_debug=False)
            
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
    app.run(debug=True)