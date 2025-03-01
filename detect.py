import os
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import gradio as gr
from tracking import CentroidTracker

def detect_image(model, image_path):
    results = model(image_path)
    for res in results:
        print("Detected objects:", res.boxes)
    annotated_img = results[0].plot()
    plt.figure(figsize=(10,6))
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.title("Detection Results")
    plt.axis('off')
    plt.show()
    results.save()  # Saves annotated image in default directory
    return annotated_img

def detect_video(model, input_video, output_video):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    tracker = CentroidTracker(max_disappeared=30)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Save frame temporarily
        temp_img = "temp_frame.jpg"
        cv2.imwrite(temp_img, frame)
        results = model(temp_img)
        # Example: Extract bounding boxes as [x1, y1, x2, y2] list.
        boxes = []
        if results[0].boxes is not None:
            for box in results[0].boxes.xyxy.cpu().numpy():
                boxes.append(box.astype(int).tolist())
        objects = tracker.update(boxes)
        # Optionally, annotate tracking IDs on the frame.
        for (objectID, centroid) in objects.items():
            text = f"ID {objectID}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        annotated_frame = results[0].plot()
        # Blend detection results with tracking annotations.
        blended = cv2.addWeighted(annotated_frame, 0.7, frame, 0.3, 0)
        out.write(blended)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    cap.release()
    out.release()
    os.remove(temp_img)
    print(f"Video saved to {output_video}")

def detect_webcam(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        temp_img = "temp_frame.jpg"
        cv2.imwrite(temp_img, frame)
        results = model(temp_img)
        annotated_frame = results[0].plot()
        cv2.imshow("Webcam Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists(temp_img):
        os.remove(temp_img)

def gradio_interface(model):
    import cv2
    import gradio as gr

    def predict(image):
        # image is now provided as a file path (if using type="filepath")
        results = model(image)
        annotated_img = results[0].plot()
        # Convert BGR to RGB for display.
        return cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    iface = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="filepath", label="Upload Image"),
        outputs=gr.Image(label="Detection Result"),
        title="Advanced YOLOv8 Object Detection",
        description="Upload an image for object detection."
    )
    iface.launch(server_name="0.0.0.0", server_port=7860)

def main():
    parser = argparse.ArgumentParser(description="Advanced Object Detection with YOLOv8")
    parser.add_argument("--mode", type=str, default="image", choices=["image", "video", "webcam", "gradio", "api"],
                        help="Mode: image, video, webcam, gradio, or api (REST endpoint)")
    parser.add_argument("--input", type=str, help="Input file path (for image or video modes)")
    parser.add_argument("--output", type=str, help="Output file path (for video mode)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLOv8 model checkpoint")
    args = parser.parse_args()

    model = YOLO(args.model)

    if args.mode == "image":
        if not args.input:
            print("Error: --input is required for image mode.")
            return
        detect_image(model, args.input)
    elif args.mode == "video":
        if not args.input or not args.output:
            print("Error: --input and --output are required for video mode.")
            return
        detect_video(model, args.input, args.output)
    elif args.mode == "webcam":
        detect_webcam(model)
    elif args.mode == "gradio":
        gradio_interface(model)
    elif args.mode == "api":
        # For API mode, call the REST API server (see api.py)
        print("For API mode, run the 'api.py' server using Uvicorn.")
    else:
        print("Invalid mode.")

if __name__ == "__main__":
    main()
