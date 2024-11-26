import cv2
import time
import mediapipe as mp
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from picamera2 import Picamera2

# Constants
MODEL_PATH = "gesture_recognizer.task"
CONFIDENCE_THRESHOLD = 0.70 

# Check if model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()
print("Camera initialized successfully")

# Flag to track if gesture is detected
gesture_detected = False
detected_gesture = None

# Callback function for gesture detection
def save_result(result, output_image, timestamp_ms):
    global gesture_detected, detected_gesture
    if result.gestures:
        gesture = result.gestures[0][0]
        # Only save gesture if confidence is above threshold
        if gesture.score >= CONFIDENCE_THRESHOLD:
            detected_gesture = {
                'name': gesture.category_name,
                'confidence': gesture.score
            }
            gesture_detected = True
            print(f"\nHigh confidence gesture detected!")
        else:
            print(f"\rDetected {gesture.category_name} but confidence too low: {gesture.score:.2f}", end='')

# Initialize the gesture recognizer
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=save_result,
    num_hands=1
)

recognizer = vision.GestureRecognizer.create_from_options(options)
print("Gesture recognizer initialized")

try:
    print(f"Starting gesture detection... (Confidence threshold: {CONFIDENCE_THRESHOLD*100}%)")
    print("Press 'q' to quit")
    
    while not gesture_detected:
        # Capture and process frame
        frame = picam2.capture_array()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Process the frame
        timestamp = int(time.time() * 1000)
        recognizer.recognize_async(mp_image, timestamp)
        
        # Display the frame
        cv2.putText(frame, f"Threshold: {CONFIDENCE_THRESHOLD*100}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Gesture Detection', frame)
        
        # Break if 'q' is pressed
        if cv2.waitKey(1) == ord("q"):
            print("\nQuitting application...")
            break
    
    # If a gesture was detected, display the final result
    if gesture_detected:
        print(f"\nFinal detected gesture: {detected_gesture['name']}")
        print(f"Confidence: {detected_gesture['confidence']:.2f}")

except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"\nAn error occurred: {str(e)}")
    import traceback
    traceback.print_exc()

finally:
    # Clean up
    print("\nCleaning up...")
    cv2.destroyAllWindows()
    picam2.stop()
    recognizer.close()
    print("Application closed successfully")