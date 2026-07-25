import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import math
import urllib.request
import os

# --- Configuration & Initialization ---
cam_width, cam_height = 640, 480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

# Auto-download the Hand Landmarker model file if not present in the run directory
model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
model_path = "hand_landmarker.task"

if not os.path.exists(model_path):
    print("Downloading hand_landmarker.task (approx. 5.6 MB)...")
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print("Download complete.")
    except Exception as e:
        raise RuntimeError(f"Failed to download model file: {e}")

# Initialize MediaPipe Tasks HandLandmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
)
detector = HandLandmarker.create_from_options(options)

screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = False

smoothening = 5
prev_x, prev_y = 0, 0
prev_scroll_y = 0
frame_reduction = 100

# Debounce states for click events
left_clicked = False
right_clicked = False

# For FPS overlay calculation
prev_time = time.time()

# Hand skeleton connections for manual visualization
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index Finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle Finger
    (9, 10), (10, 11), (11, 12),
    # Ring Finger
    (13, 14), (14, 15), (15, 16),
    # Pinky Finger
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm joints inside
    (5, 9), (9, 13), (13, 17)
]

def get_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

print("Virtual Mouse Started. Press 'q' to quit.")

try:
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        
        # Establish a visual indicator for tracking zone
        cv2.rectangle(
            img,
            (frame_reduction, frame_reduction),
            (cam_width - frame_reduction, cam_height - frame_reduction),
            (180, 0, 180),
            2,
        )
        cv2.putText(
            img,
            "TRACKING ZONE",
            (frame_reduction + 5, frame_reduction - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 0, 180),
            1,
            cv2.LINE_AA,
        )

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = detector.detect(mp_image)

        # UI variables
        mode_text = "NO HAND DETECTED"
        mode_color = (0, 0, 255)  # BGR Red

        if results.hand_landmarks:
            mode_text = "IDLE (CALIBRATING)"
            mode_color = (255, 255, 0)  # BGR Cyan

            for hand_landmarks in results.hand_landmarks:
                h, w, _ = img.shape
                
                # Key fingertip locations
                index_tip = (
                    int(hand_landmarks[8].x * w),
                    int(hand_landmarks[8].y * h),
                )
                middle_tip = (
                    int(hand_landmarks[12].x * w),
                    int(hand_landmarks[12].y * h),
                )
                thumb_tip = (
                    int(hand_landmarks[4].x * w),
                    int(hand_landmarks[4].y * h),
                )

                # Key baseline landmarks for scale invariance
                wrist = (
                    int(hand_landmarks[0].x * w),
                    int(hand_landmarks[0].y * h),
                )
                middle_mcp = (
                    int(hand_landmarks[9].x * w),
                    int(hand_landmarks[9].y * h),
                )

                # Reference hand scale (distance from wrist to middle MCP)
                hand_scale = get_distance(wrist, middle_mcp)
                if hand_scale == 0:
                    hand_scale = 1  # Avoid division by zero

                # Calculate landmarks for finger up detections
                pip_6_y = int(hand_landmarks[6].y * h)
                pip_10_y = int(hand_landmarks[10].y * h)

                # Relative fingertip checks
                index_up = index_tip[1] < pip_6_y
                middle_up = middle_tip[1] < pip_10_y

                # Calculate relative distances for clicking (scale-invariant)
                left_click_dist = get_distance(index_tip, thumb_tip) / hand_scale
                right_click_dist = get_distance(middle_tip, thumb_tip) / hand_scale

                # Draw skeleton connections manually
                for start_idx, end_idx in HAND_CONNECTIONS:
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]
                    start_pt = (int(start.x * w), int(start.y * h))
                    end_pt = (int(end.x * w), int(end.y * h))
                    cv2.line(img, start_pt, end_pt, (180, 180, 180), 1, cv2.LINE_AA)

                # Draw joints
                for lm in hand_landmarks:
                    pt = (int(lm.x * w), int(lm.y * h))
                    cv2.circle(img, pt, 4, (0, 255, 255), cv2.FILLED)

                # 1. Cursor Movement Mode (Index finger up, middle finger down, not clicking)
                if index_up and not middle_up and left_click_dist >= 0.20:
                    mode_text = "MOVE CURSOR"
                    mode_color = (255, 100, 0)  # BGR Light Blue/Orange

                    # Interpolate index coordinates from workspace window to screen size
                    x3 = np.interp(
                        index_tip[0],
                        (frame_reduction, cam_width - frame_reduction),
                        (0, screen_width),
                    )
                    y3 = np.interp(
                        index_tip[1],
                        (frame_reduction, cam_height - frame_reduction),
                        (0, screen_height),
                    )

                    # Linear smooth step interpolation
                    curr_x = prev_x + (x3 - prev_x) / smoothening
                    curr_y = prev_y + (y3 - prev_y) / smoothening
                    curr_x = max(0, min(screen_width - 1, curr_x))
                    curr_y = max(0, min(screen_height - 1, curr_y))

                    pyautogui.moveTo(curr_x, curr_y)
                    cv2.circle(img, index_tip, 15, (255, 0, 0), cv2.FILLED)
                    prev_x, prev_y = curr_x, curr_y

                # 2. Left Click (Pinch Index & Thumb)
                if left_click_dist < 0.20:
                    mode_text = "LEFT CLICK"
                    mode_color = (0, 255, 0)  # BGR Green
                    cv2.circle(img, index_tip, 15, mode_color, cv2.FILLED)
                    cv2.line(img, index_tip, thumb_tip, mode_color, 2)
                    
                    if not left_clicked:
                        pyautogui.click()
                        left_clicked = True  # Non-blocking lock
                else:
                    left_clicked = False  # Unlock on finger separation

                # 3. Right Click (Pinch Middle & Thumb)
                if right_click_dist < 0.20:
                    mode_text = "RIGHT CLICK"
                    mode_color = (0, 0, 255)  # BGR Red
                    cv2.circle(img, middle_tip, 15, mode_color, cv2.FILLED)
                    cv2.line(img, middle_tip, thumb_tip, mode_color, 2)

                    if not right_clicked:
                        pyautogui.rightClick()
                        right_clicked = True  # Non-blocking lock
                else:
                    right_clicked = False  # Unlock on finger separation

                # 4. Scrolling Mode (Index & Middle fingers both straight up)
                if index_up and middle_up and left_click_dist >= 0.20 and right_click_dist >= 0.20:
                    mode_text = "SCROLL MODE"
                    mode_color = (0, 242, 255)  # BGR Yellow-Green
                    
                    current_scroll_y = (index_tip[1] + middle_tip[1]) / 2
                    if prev_scroll_y == 0:
                        prev_scroll_y = current_scroll_y
                    else:
                        scroll_delta = current_scroll_y - prev_scroll_y
                        # Accumulate scroll distance until it breaks a threshold to handle slow scrolls
                        if scroll_delta < -15:
                            pyautogui.scroll(120)  # Scroll Up
                            prev_scroll_y = current_scroll_y
                        elif scroll_delta > 15:
                            pyautogui.scroll(-120)  # Scroll Down
                            prev_scroll_y = current_scroll_y
                else:
                    prev_scroll_y = 0

        # Draw semi-transparent status HUD panel
        hud_overlay = img.copy()
        cv2.rectangle(hud_overlay, (10, 10), (cam_width - 10, 50), (45, 45, 45), cv2.FILLED)
        img = cv2.addWeighted(hud_overlay, 0.6, img, 0.4, 0)

        # Render HUD texts
        cv2.putText(
            img,
            f"MODE: {mode_text}",
            (25, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            mode_color,
            2,
            cv2.LINE_AA,
        )
        
        # Calculate and display FPS
        curr_time = time.time()
        time_diff = curr_time - prev_time
        fps = 1.0 / time_diff if time_diff > 0 else 0.0
        prev_time = curr_time
        cv2.putText(
            img,
            f"FPS: {int(fps)}",
            (cam_width - 130, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Virtual Mouse - Hackathon Edition", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    # Close detector safely
    detector.close()