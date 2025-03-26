import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from io import BytesIO
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import websockets
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://edge21-crt9y.ondigitalocean.app",
        "https://hammerhead-app-4s7v6.ondigitalocean.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

yolo_model = YOLO("app/data/best1.pt")


# Function to  available cameras
def get_available_cameras():
    """Returns a list of available camera indices."""
    available_cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras


def resize_stretch(image, target_size=(640, 640)):
    """Resize image to target size with stretching (no aspect ratio preservation)."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


card_labels = ["10", "2", "3", "4", "5", "6", "7", "8", "9", "A", "J", "K", "Q"]


@app.get("/")
def health_check():
    return {"health_check": "OK"}


@app.websocket("/video-detect")
async def video_detect(websocket: WebSocket):
    """
    Handles real-time card detection using server-side camera.
    Captures frames from camera, processes them with YOLO, and sends results to client.
    """
    await websocket.accept()
    print("WebSocket connection established, initializing camera...")

    cap = None
    try:
        # Get available cameras
        cameras = get_available_cameras()
        if not cameras:
            await websocket.send_json({"error": "No cameras found on server"})
            return

        print(f"Available Cameras: {cameras}")
        camera_index = cameras[-1]

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            await websocket.send_json({"error": "Could not open camera"})
            return

        print(f"Camera {camera_index} initialized successfully")
        await websocket.send_json({"info": f"Camera {camera_index} initialized"})

        time.sleep(1)

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                if data.lower() == "stop":
                    await websocket.send_json({"info": "Stopping camera as requested"})
                    break
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                print(f"Client message error: {str(e)}")
                break

            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                await websocket.send_json({"error": "Failed to capture frame"})
                await asyncio.sleep(0.5)
                continue

            results = yolo_model(frame)

            detected_objects = []
            for r in results:
                for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    x1, y1, x2, y2 = map(int, box[:4])
                    class_id = int(cls.item())
                    confidence = float(conf.item())

                    if class_id < 0 or class_id >= len(card_labels):
                        continue

                    detected_objects.append(
                        {
                            "label": card_labels[class_id],
                            "bbox": [x1, y1, x2, y2],
                            "confidence": confidence,
                        }
                    )

            print(f"Sending {len(detected_objects)} detections to client")
            await websocket.send_json({"detections": detected_objects})

            await asyncio.sleep(0.05)  # ~20 FPS

    except websockets.exceptions.ConnectionClosedOK:
        print("Client disconnected normally")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Client disconnected with error: {str(e)}")
    except Exception as e:
        print(f"Error in camera websocket: {str(e)}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        print("Closing camera and WebSocket connection")
        if cap and cap.isOpened():
            cap.release()
        await websocket.close()
