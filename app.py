import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import math
import urllib.request
import os

class VirtualMouse:
    def __init__(self):
        self.cam_width, self.cam_height = 640, 480
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        self.model_path = "hand_landmarker.task"
        self._check_and_download_model()

        self.detector = self._initialize_detector()
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False

        # Exponential Moving Average smoothing parameters
        self.ema_alpha = 0.5  # Defines how much weight the recent point has (lower = smoother but delayed)
        self.ema_x, self.ema_y = None, None

        self.prev_scroll_y = 0
        self.frame_reduction = 100

        self.left_clicked = False
        self.right_clicked = False
        self.prev_time = time.time()
        
        self.vol_debounce_time = 0

        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]

    def _check_and_download_model(self):
        model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        if not os.path.exists(self.model_path):
            print("Downloading hand_landmarker.task (approx. 5.6 MB)...")
            try:
                urllib.request.urlretrieve(model_url, self.model_path)
                print("Download complete.")
            except Exception as e:
                raise RuntimeError(f"Failed to download model file: {e}")

    def _initialize_detector(self):
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=1,
        )
        return mp.tasks.vision.HandLandmarker.create_from_options(options)

    def get_distance(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def draw_skeleton(self, img, hand_landmarks, w, h):
        # Draw connections
        for start_idx, end_idx in self.HAND_CONNECTIONS:
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            start_pt = (int(start.x * w), int(start.y * h))
            end_pt = (int(end.x * w), int(end.y * h))
            cv2.line(img, start_pt, end_pt, (180, 180, 180), 1, cv2.LINE_AA)

        # Draw joints
        for lm in hand_landmarks:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(img, pt, 4, (0, 255, 255), cv2.FILLED)

    def process_hand(self, img, hand_landmarks):
        h, w, _ = img.shape
        mode_text = "IDLE (CALIBRATING)"
        mode_color = (255, 255, 0)
        
        # Fingers
        index_tip = (int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h))
        middle_tip = (int(hand_landmarks[12].x * w), int(hand_landmarks[12].y * h))
        ring_tip = (int(hand_landmarks[16].x * w), int(hand_landmarks[16].y * h))
        pinky_tip = (int(hand_landmarks[20].x * w), int(hand_landmarks[20].y * h))
        thumb_tip = (int(hand_landmarks[4].x * w), int(hand_landmarks[4].y * h))

        wrist = (int(hand_landmarks[0].x * w), int(hand_landmarks[0].y * h))
        middle_mcp = (int(hand_landmarks[9].x * w), int(hand_landmarks[9].y * h))

        # Scale invariance ref
        hand_scale = self.get_distance(wrist, middle_mcp)
        if hand_scale == 0: hand_scale = 1

        # Check upward fingers based on PIP joint comparisons
        pip_6_y = int(hand_landmarks[6].y * h)
        pip_10_y = int(hand_landmarks[10].y * h)
        pip_14_y = int(hand_landmarks[14].y * h)
        pip_18_y = int(hand_landmarks[18].y * h)

        index_up = index_tip[1] < pip_6_y
        middle_up = middle_tip[1] < pip_10_y
        ring_up = ring_tip[1] < pip_14_y
        pinky_up = pinky_tip[1] < pip_18_y

        left_click_dist = self.get_distance(index_tip, thumb_tip) / hand_scale
        right_click_dist = self.get_distance(middle_tip, thumb_tip) / hand_scale

        self.draw_skeleton(img, hand_landmarks, w, h)

        target_pt = None

        # 5. Volume Control Gesture (Spider-Man: Pinky Up, Index Up)
        if pinky_up and index_up and not middle_up and not ring_up:
            mode_text = "VOLUME CONTROL"
            mode_color = (128, 0, 128) # BGR Purple
            
            current_time = time.time()
            if current_time - self.vol_debounce_time > 0.1:
                if getattr(self, 'prev_vol_y', None) is None:
                    self.prev_vol_y = index_tip[1]
                
                vol_delta = index_tip[1] - self.prev_vol_y
                if vol_delta < -15:
                    pyautogui.press('volumeup')
                    self.prev_vol_y = index_tip[1]
                    self.vol_debounce_time = current_time
                elif vol_delta > 15:
                    pyautogui.press('volumedown')
                    self.prev_vol_y = index_tip[1]
                    self.vol_debounce_time = current_time
            return mode_text, mode_color
        else:
            if hasattr(self, 'prev_vol_y'):
                del self.prev_vol_y

        # Handle Left Click / Drag initiation
        if left_click_dist < 0.20:
             left_clicked_now = True
             mode_text = "LEFT CLICK / DRAG"
             mode_color = (0, 255, 0)
        else:
             left_clicked_now = False

        # Apply hysteresis for drag release
        if self.left_clicked and left_click_dist > 0.25:
             # Release drag
             pyautogui.mouseUp()
             self.left_clicked = False
        elif not self.left_clicked and left_clicked_now:
             # Start drag/click
             pyautogui.mouseDown()
             self.left_clicked = True

        if left_clicked_now:
            cv2.circle(img, index_tip, 15, mode_color, cv2.FILLED)
            cv2.line(img, index_tip, thumb_tip, mode_color, 2)
            # Use midpoint between thumb and index as the cursor point during drag
            target_pt = ((index_tip[0] + thumb_tip[0]) // 2, (index_tip[1] + thumb_tip[1]) // 2)

        elif index_up and not middle_up:
            # 1. Cursor Movement Mode (Index finger up, middle finger down)
            mode_text = "MOVE CURSOR"
            mode_color = (255, 100, 0)
            target_pt = index_tip
            cv2.circle(img, index_tip, 15, (255, 0, 0), cv2.FILLED)
            
        if target_pt:
             # Interpolate to Screen Coordinates
             x_screen = np.interp(target_pt[0], (self.frame_reduction, self.cam_width - self.frame_reduction), (0, self.screen_width))
             y_screen = np.interp(target_pt[1], (self.frame_reduction, self.cam_height - self.frame_reduction), (0, self.screen_height))
             
             # Apply Exponential Moving Average
             if self.ema_x is None:
                 self.ema_x, self.ema_y = x_screen, y_screen
             else:
                 self.ema_x = self.ema_alpha * x_screen + (1 - self.ema_alpha) * self.ema_x
                 self.ema_y = self.ema_alpha * y_screen + (1 - self.ema_alpha) * self.ema_y
                 
             pyautogui.moveTo(max(0, min(self.screen_width - 1, self.ema_x)), max(0, min(self.screen_height - 1, self.ema_y)))

        # 3. Right Click
        if right_click_dist < 0.20 and not left_clicked_now:
            mode_text = "RIGHT CLICK"
            mode_color = (0, 0, 255)
            cv2.circle(img, middle_tip, 15, mode_color, cv2.FILLED)
            cv2.line(img, middle_tip, thumb_tip, mode_color, 2)
            if not self.right_clicked:
                pyautogui.rightClick()
                self.right_clicked = True
        elif right_click_dist > 0.25:
            self.right_clicked = False

        # 4. Scrolling Mode
        if index_up and middle_up and right_click_dist >= 0.25 and left_click_dist >= 0.25 and not pinky_up:
            mode_text = "SCROLL MODE"
            mode_color = (0, 242, 255)
            current_scroll_y = (index_tip[1] + middle_tip[1]) / 2
            
            if self.prev_scroll_y == 0:
                self.prev_scroll_y = current_scroll_y
            else:
                scroll_delta = current_scroll_y - self.prev_scroll_y
                if scroll_delta < -15:
                    pyautogui.scroll(120)
                    self.prev_scroll_y = current_scroll_y
                elif scroll_delta > 15:
                    pyautogui.scroll(-120)
                    self.prev_scroll_y = current_scroll_y
        else:
            self.prev_scroll_y = 0

        return mode_text, mode_color

    def run(self):
        print("Virtual Mouse Started. Press 'q' to quit.")
        try:
            while True:
                success, img = self.cap.read()
                if not success:
                    break

                img = cv2.flip(img, 1)
                
                # Tracking Zone visually
                cv2.rectangle(img, (self.frame_reduction, self.frame_reduction),
                              (self.cam_width - self.frame_reduction, self.cam_height - self.frame_reduction),
                              (180, 0, 180), 2)
                cv2.putText(img, "TRACKING ZONE", (self.frame_reduction + 5, self.frame_reduction - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 0, 180), 1, cv2.LINE_AA)

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                results = self.detector.detect(mp_image)

                mode_text = "NO HAND DETECTED"
                mode_color = (0, 0, 255)

                if results.hand_landmarks:
                    # We assume single hand usage (num_hands=1)
                    mode_text, mode_color = self.process_hand(img, results.hand_landmarks[0])

                # Overlay HUD
                hud_overlay = img.copy()
                cv2.rectangle(hud_overlay, (10, 10), (self.cam_width - 10, 50), (45, 45, 45), cv2.FILLED)
                img = cv2.addWeighted(hud_overlay, 0.6, img, 0.4, 0)
                
                # Render HUD text
                cv2.putText(img, f"MODE: {mode_text}", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2, cv2.LINE_AA)

                # FPS Calculation
                curr_time = time.time()
                time_diff = curr_time - self.prev_time
                fps = 1.0 / time_diff if time_diff > 0 else 0.0
                self.prev_time = curr_time
                cv2.putText(img, f"FPS: {int(fps)}", (self.cam_width - 130, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

                cv2.imshow("Virtual Mouse - Enhanced", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.detector.close()

if __name__ == "__main__":
    app = VirtualMouse()
    app.run()