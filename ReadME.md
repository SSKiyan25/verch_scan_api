# verch_scan_api

A lightweight Flask API that uses YOLOv8 ONNX models to detect and classify verch items (CS3 lanyards, shirts, and polo shirts) in images. The API accepts image uploads via HTTP POST requests and returns detection results including class names, confidence scores, and bounding box coordinates.

## Features

- REST API endpoint for object detection
- Utilizes Ultralytics YOLO models for inference
- Cross-origin resource sharing (CORS) support
- Error handling with detailed feedback
- Environment variable configuration

## Usage

Send a POST request to the `/detect` endpoint with an image file to receive detection results in JSON format.
