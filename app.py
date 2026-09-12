from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# Load trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_mobilenetv2.keras")

model = tf.keras.models.load_model(MODEL_PATH)

print("Trained model loaded successfully!")

# Class names must match training order
class_names = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Target_Spot",
    "Tomato___healthy"
]

# Treatment information
treatments = {
    "Tomato___Bacterial_spot":
        "Remove infected leaves and avoid overhead watering. Improve air circulation and use an appropriate copper-based treatment if recommended.",

    "Tomato___Early_blight":
        "Remove infected leaves, improve air circulation, avoid overhead watering, and use an appropriate fungicide if necessary.",

    "Tomato___Late_blight":
        "Remove affected plant material immediately, improve air circulation, avoid wet foliage, and use an appropriate fungicide according to local recommendations.",

    "Tomato___Leaf_Mold":
        "Remove infected leaves, reduce humidity, improve ventilation, and avoid overhead irrigation.",

    "Tomato___Septoria_leaf_spot":
        "Remove infected leaves, keep foliage dry, improve air circulation, and use an appropriate fungicide if necessary.",

    "Tomato___Target_Spot":
        "Remove affected leaves, improve air circulation, avoid overhead watering, and use an appropriate fungicide when required.",

    "Tomato___healthy":
        "The plant appears healthy. Continue proper watering, sunlight, nutrition, and regular monitoring."
}


@app.route("/")
def home():
    return jsonify({
        "status": "success",
        "message": "Plant Disease Prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Check whether image was uploaded
        if "image" not in request.files:
            return jsonify({
                "error": "No image uploaded"
            }), 400

        file = request.files["image"]

        # Read image
        image = Image.open(file).convert("RGB")

        # Resize to MobileNetV2 input size
        image = image.resize((224, 224))

        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # MobileNetV2 preprocessing
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            image_array
        )

        # Prediction
        predictions = model.predict(image_array, verbose=0)

        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index]) * 100

        predicted_class = class_names[predicted_index]

        # Determine condition
        if predicted_class == "Tomato___healthy":
            condition = "Healthy"
            disease_name = "No Disease Detected"
        else:
            condition = "Diseased"
            disease_name = predicted_class.replace("Tomato___", "").replace("_", " ")

        # Treatment
        treatment = treatments[predicted_class]

        # Return result
        return jsonify({
            "plant_name": "Tomato",
            "condition": condition,
            "disease_name": disease_name,
            "confidence": round(confidence, 2),
            "treatment": treatment
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )