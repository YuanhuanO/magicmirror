import cv2
import time
import mediapipe as mp
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from picamera2 import Picamera2

class CombinedGestureDetector:
    def __init__(self, model_path="gesture_recognizer.task", confidence_threshold=0.70):
        # Initialize MediaPipe hands for swipe detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        
        # Swipe detection parameters
        self.prev_x = None
        self.movements = []
        self.swipe_threshold = 100
        
        # Static gesture recognition parameters
        self.MODEL_PATH = model_path
        self.CONFIDENCE_THRESHOLD = confidence_threshold
        self.HIGH_CONFIDENCE_THRESHOLD = 0.90  # New threshold for skipping swipe check
        self.gesture_detected = False
        self.detected_gesture = None
        
        # Initialize the gesture recognizer
        if not os.path.exists(self.MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {self.MODEL_PATH}")
            
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self._save_result,
            num_hands=1
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        print("Combined gesture detector initialized")

    def _save_result(self, result, output_image, timestamp_ms):
        """Callback for static gesture detection"""
        if result.gestures:
            gesture = result.gestures[0][0]
            if gesture.score >= self.CONFIDENCE_THRESHOLD:
                self.detected_gesture = {
                    'type': 'static',
                    'name': gesture.category_name,
                    'confidence': gesture.score
                }
                self.gesture_detected = True

    def detect_swipe(self, frame):
        """Detect swipe gestures"""
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        swipe_detected = None
        frame_width = frame.shape[1]
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Track wrist position
            wrist = hand_landmarks.landmark[0]
            curr_x = wrist.x * frame_width
            
            if self.prev_x is not None:
                movement = curr_x - self.prev_x
                self.movements.append(movement)
                
                if len(self.movements) > 5:
                    self.movements.pop(0)
                
                total_movement = sum(self.movements)
                if abs(total_movement) > self.swipe_threshold:
                    swipe_detected = {
                        'type': 'swipe',
                        'direction': "right" if total_movement > 0 else "left"
                    }
            
            self.prev_x = curr_x
        
        return swipe_detected

    def detect_gestures(self, frame):
        """Detect both static and swipe gestures with priority on static gestures"""
        # Reset detection flags
        self.gesture_detected = False
        self.detected_gesture = None
        
        # Check for static gestures first
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int(time.time() * 1000)
        self.recognizer.recognize_async(mp_image, timestamp)
        
        # If static gesture detected with high confidence, return it immediately
        if self.gesture_detected and self.detected_gesture['confidence'] >= self.HIGH_CONFIDENCE_THRESHOLD:
            print(f"High confidence static gesture detected: {self.detected_gesture['name']} "
                  f"({self.detected_gesture['confidence']:.2f})")
            return self.detected_gesture
        
        # If no high-confidence static gesture, check for swipes
        swipe_result = self.detect_swipe(frame)
        if swipe_result:
            print(f"Swipe detected: {swipe_result['direction']}")
            return swipe_result
            
        # If we had a lower confidence static gesture, return it
        if self.gesture_detected:
            print(f"Lower confidence static gesture detected: {self.detected_gesture['name']} "
                  f"({self.detected_gesture['confidence']:.2f})")
            return self.detected_gesture
            
        return None

    def close(self):
        """Clean up resources"""
        self.recognizer.close()
        self.hands.close()




def main():
    # Initialize camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": 'XRGB8888', "size": (640, 480)}
    ))
    picam2.start()
    print("Camera initialized successfully")
    
    # Initialize detector
    detector = CombinedGestureDetector()
    
    try:
        print("Starting combined gesture detection... Press 'q' to quit")
        gesture_count = 0
        last_gesture_time = 0
        GESTURE_COOLDOWN = 1.0  # Seconds between valid gestures
        
        while True:
            frame = picam2.capture_array()
            current_time = time.time()
            
            # Detect gestures
            gesture = detector.detect_gestures(frame)
            
            # Handle detected gesture
            if gesture and (current_time - last_gesture_time) >= GESTURE_COOLDOWN:
                gesture_count += 1
                last_gesture_time = current_time
                
                if gesture['type'] == 'swipe':
                    print(f"Gesture #{gesture_count}: Swipe {gesture['direction']}!")
                else:  # static gesture
                    print(f"Gesture #{gesture_count}: {gesture['name']} "
                          f"(confidence: {gesture['confidence']:.2f})")
            
            # Optional: Display frame (can be removed if no visualization needed)
            cv2.imshow('Gesture Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\nCleaning up... Total gestures detected: {gesture_count}")
        cv2.destroyAllWindows()
        picam2.stop()
        detector.close()
        print("Application closed successfully")

if __name__ == "__main__":
    main()