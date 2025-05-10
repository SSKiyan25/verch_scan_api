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

@app.route('/detect', methods=['POST'])
def detect_items():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files['image']
        
        # Use Ultralytics for preprocessing/inference
        results = model(Image.open(image_file))  # Directly pass PIL image
        
        # Process results using Ultralytics API
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                # Use the model's built-in names
                class_name = model.names[class_id]
                
                detections.append({
                    "class": class_name,
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                })

        return jsonify({"detections": detections})

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    # For Render deployment - get port from environment variable
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Verch Scan API server on port {port}")
    print("Press Ctrl+C to stop the server")
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )