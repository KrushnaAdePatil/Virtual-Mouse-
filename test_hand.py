import urllib.request
import os
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python import BaseOptions

model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
model_path = "hand_landmarker.task"

if not os.path.exists(model_path):
    print("Downloading hand_landmarker.task...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Download complete.")

print("Initializing HandLandmarker...")
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1
)

try:
    detector = HandLandmarker.create_from_options(options)
    print("Success: Detector initialized!")
    detector.close()
except Exception as e:
    print(f"Error: {e}")
