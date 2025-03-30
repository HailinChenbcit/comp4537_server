import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import asyncio
import websockets
from collections import deque
import time

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
yolo_model = YOLO("app/data/best1.pt") 


card_labels = ["10", "2", "3", "4", "5", "6", "7", "8", "9", "A", "J", "K", "Q"]

# Global frame queue with only the latest frame retained
frame_queue = deque(maxlen=1)

@app.get("/")
def health_check():
    return {"health_check": "OK"}

@app.websocket("/video-detect")
async def video_detect(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket connected")

    # Frame receiving coroutine
    async def receive_frames():
        while True:
            try:
                print("📥 Waiting for new frame...")
                data = await websocket.receive_bytes()
                frame_queue.append(data)
                print(f"📦 Frame received ({len(data)} bytes)")
            except Exception as e:
                break

    # Start receiver in background
    asyncio.create_task(receive_frames())

    try:
        while True:
            if not frame_queue:
                await asyncio.sleep(0.01)
                continue

            data = frame_queue.pop()
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({"error": "Invalid image data"})
                continue

            # YOLO inference
            results = yolo_model(frame, conf=0.3)
            detected_objects = []
            for r in results:
                for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    x1, y1, x2, y2 = map(int, box[:4])
                    class_id = int(cls.item())
                    confidence = float(conf.item())

                    if 0 <= class_id < len(card_labels):
                        detected_objects.append({
                            "label": card_labels[class_id],
                            "bbox": [x1, y1, x2, y2],
                            "confidence": confidence,
                        })

            await websocket.send_json({"detections": detected_objects})

    except websockets.exceptions.ConnectionClosed:
        print("WebSocket disconnected")
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        await websocket.close()
