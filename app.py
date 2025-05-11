import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from ultralytics import YOLO
from PIL import Image

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Load your custom ONNX model with Ultralytics
model = YOLO("best.onnx")  # Ultralytics handles ONNX models natively

@app.route('/', methods=['GET'])
def index():
    """Root route for health checks"""
    return jsonify({"status": "ok", "message": "Verch Scan API is running"}), 200

@app.route('/detect', methods=['POST'])
def detect_items():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files['image']
        img = Image.open(image_file)
        
        # Use Ultralytics for preprocessing/inference
        results = model(img)  # Directly pass PIL image
        
        # Process results using Ultralytics API
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                # Use the model's built-in names
                class_name = model.names[class_id]
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detections.append({
                    "class": class_name,
                    "confidence": float(box.conf[0]),
                    "box": {  # Changed from 'bbox' to 'box' to match client code
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2)
                    }
                })

        return jsonify({"detections": detections})

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    # Let Gunicorn handle the port binding
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)