"""
Real-Time Object Detection & Tracking System

Features:
- YOLOv8 Detection
- ByteTrack Tracking
- Object Counting
- FPS Monitoring
- Webcam & Video Support
"""

import cv2
import time
from collections import Counter
from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5

MODE = "webcam"

VIDEO_PATH = "sample.mp4"

model = YOLO(MODEL_NAME)

if MODE == "webcam":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open source.")
    exit()

WINDOW_NAME = "Real-Time Object Detection & Tracking"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)


prev_time = time.time()

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model.track(
        frame,
        persist=True,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    annotated_frame = results[0].plot()

    counts = Counter()

    if results[0].boxes is not None:

        class_ids = results[0].boxes.cls.tolist()

        for cls_id in class_ids:
            class_name = model.names[int(cls_id)]
            counts[class_name] += 1
 
    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    y_offset = 30

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    y_offset += 35

    total_objects = sum(counts.values())

    cv2.putText(
        annotated_frame,
        f"Objects: {total_objects}",
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    y_offset += 40

    for obj_name, count in counts.items():

        cv2.putText(
            annotated_frame,
            f"{obj_name}: {count}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        y_offset += 30

    cv2.imshow(WINDOW_NAME, annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()