from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI(title="YOLOv8 Object Detection API")

# Load the model (adjust model path if necessary)
model = YOLO("yolov8n.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    contents = await file.read()
    # Convert the file contents to a numpy array and decode the image.
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    results = model(image)
    # Extract bounding boxes and classes.
    output = []
    for box in results[0].boxes.xyxy.cpu().numpy():
        output.append(box.tolist())
    return JSONResponse(content={"detections": output})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
