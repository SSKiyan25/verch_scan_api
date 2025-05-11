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
            
            # Get original image dimensions
            img_width, img_height = img.size
            logger.info(f"Original image dimensions: {img_width}x{img_height}")
            
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
                    
                    logger.info(f"Raw Detection {j}: class={class_name}, conf={confidence:.2f}, box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                    
                    # Check if coordinates are valid - fix potential zero height boxes
                    if y1 == y2:
                        # Adjust to make a small box if height is zero
                        if y1 == 0:
                            y2 = min(10, img_height/10)  # Use 10px or 10% of image height
                        else:
                            y1 = max(0, y2 - min(10, img_height/10))
                    
                    if x1 == x2:
                        # Adjust to make a small box if width is zero
                        if x1 == 0:
                            x2 = min(10, img_width/10)
                        else:
                            x1 = max(0, x2 - min(10, img_width/10))
                    
                    # Calculate center and dimensions for the Flutter app
                    width = x2 - x1
                    height = y2 - y1
                    center_x = x1 + width/2
                    center_y = y1 + height/2
                    
                    # Only include valid detections (non-zero dimensions)
                    if width > 0 and height > 0:
                        logger.info(f"Adjusted Detection {j}: class={class_name}, conf={confidence:.2f}, box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                        
                        detections.append({
                            "class": class_name,
                            "confidence": confidence,
                            "box": {
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2),
                                "width": float(width),
                                "height": float(height),
                                "center_x": float(center_x),
                                "center_y": float(center_y)
                            }
                        })
                    else:
                        logger.warning(f"Skipping detection with invalid dimensions: width={width}, height={height}")
                        
                except Exception as box_error:
                    logger.error(f"Error processing detection box {j}: {str(box_error)}")
                    logger.error(traceback.format_exc())

        logger.info(f"Returning {len(detections)} detections")
        return jsonify({
            "detections": detections,
            "image_size": {
                "width": img_width,
                "height": img_height
            }
        })

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run() 