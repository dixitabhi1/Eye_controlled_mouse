# Eye_controlled_mouse


This project implements a real-time eye-tracking mouse controller using OpenCV, MediaPipe, and PyAutoGUI. It allows users to control the cursor with eye movement and click using blinks, offering a hands-free way to interact with a computer — especially useful for people with mobility impairments.



## 🚀 Features
- **Gaze-based cursor movement** using webcam feed with OpenCV and MediaPipe for facial landmark detection  
- **Blink-to-click activation** via PyAutoGUI  
- **No additional hardware required**—uses just a standard webcam

---

## 🛠️ Tech Stack
- `opencv-python` – for camera capture and image processing  
- `mediapipe` – for real-time face and eye landmark detection  
- `pyautogui` – to map eye movements and blink actions to mouse controls
