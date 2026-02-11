# ===================== IMPORTS =====================
import cv2
import requests
import numpy as np

# ===================== CONFIG =====================
API_URL = "http://localhost:8000/detect-frame"
CONF_THRESHOLD = 0.25

# ===================== OPEN WEBCAM =====================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Unable to access webcam")
    exit()

print("🎥 Webcam started. Press 'Q' to quit.")

# ===================== MAIN LOOP =====================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Encode frame to JPG
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        continue

    # Send frame to API
    try:
        response = requests.post(
            API_URL,
            files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
            timeout=3
        )
    except requests.exceptions.RequestException:
        print("⚠️ API not reachable")
        break

    if response.status_code != 200:
        print("⚠️ Detection failed")
        continue

    data = response.json()
    detections = data.get("detections", [])

    # Draw detections
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        conf = det["confidence"]
        class_id = det["class_id"]

        if conf < CONF_THRESHOLD:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Class {class_id} | {conf:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Traffic Sign Detection (API)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ===================== CLEANUP =====================
cap.release()
cv2.destroyAllWindows()
print("👋 Webcam stopped")
