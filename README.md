# Advanced Object Detection with YOLOv8

## Overview
This project demonstrates an advanced object detection pipeline using YOLOv8 (Ultralytics). In addition to running inference on single images, the project now supports:
- **Image Mode:** Detect objects in a single image.
- **Video Mode:** Process a video file frame-by-frame, display detections in real‑time, and save an annotated video.
- **Webcam Mode:** Perform real‑time detection on live webcam input.
- **Interactive Dashboard:** Launch a Gradio interface for image/video upload and interactive inference.

These enhancements showcase advanced computer vision techniques and provide flexibility for different deployment scenarios.

## Features
- **Pretrained YOLOv8 Model:** Uses a YOLOv8 model (default: `yolov8n.pt`) pretrained on COCO.
- **Multi-Mode Input:** Supports image, video, and webcam input.
- **Video Processing:** Annotates and saves processed video.
- **Real-Time Webcam Detection:** Uses OpenCV to capture and process live video.
- **Gradio Interface:** Optional interactive web interface for inference.
- **Docker-Ready:** Comes with a Dockerfile for containerized deployment.

## Project Structure
```
advanced-object-detection-yolov8/
├── README.md               # Project overview, features, installation, and usage.
├── requirements.txt        # Required Python packages.
├── Dockerfile              # Container configuration.
├── detect.py               # Main detection script supporting image, video, webcam, Gradio, and REST API modes.
├── tracking.py             # Simple object tracking (centroid tracking) utility.
├── api.py                  # FastAPI server for detection via REST endpoints.
└── notebooks/
    └── Object_Detection.ipynb  # Notebook for interactive experimentation and visualization.
```

## Installation

**Clone the Repository:**
   ```
   git clone https://github.com/mda84/advanced-object-detection-yolov8.git
   cd advanced-object-detection-yolov8
   ```

Create and Activate a Virtual Environment:
   ```
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

Install Dependencies:
   ```
pip install -r requirements.txt
   ```

Usage
Command-Line Script
The script supports multiple modes. For example:

Image Mode:
   ```
python detect.py --mode image --input path/to/your/image.jpg
   ```

Video Mode:
   ```
python detect.py --mode video --input path/to/your/video.mp4 --output path/to/output_video.mp4
   ```

Webcam Mode:
   ```
python detect.py --mode webcam
   ```

Gradio Mode (Interactive Dashboard):
   ```
python detect.py --mode gradio
   ```

## Interactive Notebook
Open the notebook in the notebooks/ folder for interactive experimentation:
   ```
jupyter notebook notebooks/Object_Detection.ipynb
   ```

## Docker Deployment
To build and run the Docker container:
   ```
docker build -t advanced-object-detection-yolov8 .
docker run -it advanced-object-detection-yolov8
   ```

## License
This project is licensed under the MIT License.

## Contact
For questions, collaboration, or contributions, please contact dorkhah9@gmail.com