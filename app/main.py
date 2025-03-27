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
    allow_origins=["*"
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
    Receives image frames from the client over WebSocket,
    runs YOLO detection on each frame, and sends back detection results.
    """
    await websocket.accept()
    print("✅ WebSocket connection accepted")

    try:
        while True:
            print("⏳ Waiting for image frame...")
            data = await websocket.receive_bytes()
            print(f"📸 Received {len(data)} bytes")

            # Decode to OpenCV frame
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                print("❌ Frame decode failed")
                await websocket.send_json({"error": "Invalid image data"})
                continue
            # Run YOLO detection
            try:
                results = yolo_model(frame)
            except Exception as e:
                print(f"❌ YOLO inference error: {e}")
                await websocket.send_json({"error": f"YOLO error: {str(e)}"})
                continue

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

            print(f"🎯 Detected {len(detected_objects)} object(s)")
            await websocket.send_json({"detections": detected_objects})

    except websockets.exceptions.ConnectionClosed:
        print("🔌 WebSocket disconnected")
    except Exception as e:
        print(f"❌ Error in video_detect: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        await websocket.close()
        print("✅ WebSocket closed")
