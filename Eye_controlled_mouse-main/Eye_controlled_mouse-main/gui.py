import cv2
import mediapipe as mp
import pyautogui
import time
from datetime import datetime
from pymongo import MongoClient
import matplotlib.pyplot as plt
import pandas as pd
import threading


MONGO_URI = "mongodb+srv://mishrashardendu07:mongomishra@cluster0.c1mfitg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client.eyeMouse
collection = db.activityLogs


cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()


prev_x, prev_y = 0, 0
smooth_factor = 0.2
deadzone = 5


def detect_posture(nose_y, frame_h):
    relative_y = nose_y / frame_h
    if relative_y > 0.55:
        return "Leaning Forward"
    elif relative_y < 0.35:
        return "Leaning Backward"
    else:
        return "Neutral"

def alert_user(posture):
    if posture == "Leaning Forward":
        print("\u26a0\ufe0f Sit upright! You are leaning forward.")
    elif posture == "Leaning Backward":
        print("\u26a0\ufe0f Sit upright! You are leaning backward.")


def generate_heatmap():
    while True:
        time.sleep(30)  # Generate every 30 seconds
        logs = list(collection.find().sort("timestamp", -1).limit(200))
        if logs:
            df = pd.DataFrame(logs)
            plt.figure(figsize=(8, 4))
            plt.hexbin(df['cursor_x'], df['cursor_y'], gridsize=40, cmap='plasma')
            plt.title("Gaze Heatmap (last 200 points)")
            plt.xlabel("Screen X")
            plt.ylabel("Screen Y")
            plt.colorbar(label='Density')
            plt.tight_layout()
            plt.savefig("heatmap.png")
            plt.close()

# Start heatmap thread
heatmap_thread = threading.Thread(target=generate_heatmap, daemon=True)
heatmap_thread.start()

# === Main Loop ===
while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    clicked = False
    screen_x, screen_y = 0, 0
    nose_y_pixel = 0
    posture = "Unknown"

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # Eye Gaze Tracking
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            if id == 1:
                screen_x = screen_w / frame_w * x
                screen_y = screen_h / frame_h * y

                smooth_x = prev_x + (screen_x - prev_x) * smooth_factor
                smooth_y = prev_y + (screen_y - prev_y) * smooth_factor

                if abs(smooth_x - prev_x) > deadzone or abs(smooth_y - prev_y) > deadzone:
                    pyautogui.moveTo(smooth_x, smooth_y)

                prev_x, prev_y = smooth_x, smooth_y

        # Blink Click Detection
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        if (left[0].y - left[1].y) < 0.02:
            pyautogui.doubleClick()
            clicked = True
            pyautogui.sleep(1)

        # Posture Detection
        nose = landmarks[1]
        nose_y_pixel = int(nose.y * frame_h)
        posture = detect_posture(nose_y_pixel, frame_h)

        # Show posture alert
        alert_user(posture)

        # Show posture text
        cv2.putText(frame, f'Posture: {posture}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # === Log to MongoDB ===
    if abs(smooth_x - prev_x) > deadzone or abs(smooth_y - prev_y) > deadzone:
        log_data = {
            "timestamp": datetime.utcnow(),
            "cursor_x": int(screen_x),
            "cursor_y": int(screen_y),
            "blink_clicked": clicked,
            "nose_y": nose_y_pixel,
            "posture": posture
        }
        collection.insert_one(log_data)

    # === Show Frame ===
    cv2.imshow("Eye-Controlled Mouse with MongoDB Logging", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# === Cleanup ===
cam.release()
cv2.destroyAllWindows()
