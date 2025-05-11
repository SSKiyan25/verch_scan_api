import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from ultralytics import YOLO
from PIL import Image
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Load your custom ONNX model with Ultralytics
try:
    logger.info("Loading model from path: best.onnx")
    model = YOLO("best.onnx", task="detect")  # Explicitly set task to detect
    logger.info(f"Model loaded successfully. Model type: {type(model)}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(traceback.format_exc())
    raise

@app.route('/', methods=['GET'])
def index():
    """Root route for health checks"""
    return jsonify({"status": "ok", "message": "Verch Scan API is running", "model_loaded": model is not None}), 200

@app.route('/detect', methods=['POST'])
def detect_items():
    try:
        if 'image' not in request.files:
            logger.warning("No image provided in request")
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files['image']
        logger.info(f"Received image: {image_file.filename}, size: {len(image_file.read())} bytes")
        image_file.seek(0)  # Reset file pointer after reading

        try:
            img = Image.open(image_file)
            logger.info(f"Image opened successfully. Size: {img.size}, Mode: {img.mode}")
        except Exception as img_error:
            logger.error(f"Error opening image: {str(img_error)}")
            return jsonify({"error": f"Invalid image format: {str(img_error)}"}), 400

        # Use Ultralytics for preprocessing/inference
        logger.info("Running inference...")
        results = model(img)  # Directly pass PIL image
        logger.info(f"Inference complete. Results: {len(results)} items")

        # Process results using Ultralytics API
        detections = []
        for i, result in enumerate(results):
            logger.info(f"Processing result {i}: {len(result.boxes)} boxes")
            for j, box in enumerate(result.boxes):
                try:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    logger.info(f"Detection {j}: class={class_name}, conf={confidence:.2f}, box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                    
                    detections.append({
                        "class": class_name,
                        "confidence": confidence,
                        "box": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2)
                        }
                    })
                except Exception as box_error:
                    logger.error(f"Error processing detection box {j}: {str(box_error)}")

        logger.info(f"Returning {len(detections)} detections")
        return jsonify({"detections": detections})

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run() 