import cv2
import numpy as np
import mediapipe as mp
import time
from picamera2 import Picamera2

class SwipeGestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.prev_x = None
        self.movements = []
        self.swipe_threshold = 100

    def detect_swipe(self, frame):
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        swipe_detected = None
        frame_width = frame.shape[1]
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Track wrist position
            wrist = hand_landmarks.landmark[0]
            curr_x = wrist.x * frame_width
            
            if self.prev_x is not None:
                movement = curr_x - self.prev_x
                self.movements.append(movement)
                
                # Keep only last 5 movements
                if len(self.movements) > 5:
                    self.movements.pop(0)
                
                # Check for swipe
                total_movement = sum(self.movements)
                if abs(total_movement) > self.swipe_threshold:
                    swipe_detected = "right" if total_movement > 0 else "left"
            
            self.prev_x = curr_x

        return frame, swipe_detected

# Initialize camera and detector
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))
picam2.start()
detector = SwipeGestureDetector()

try:
    print("Starting swipe detection... (Will close after detecting swipe)")
    while True:
        frame = picam2.capture_array()
        frame, swipe = detector.detect_swipe(frame)
        
        if swipe:
            print(f"Swipe {swipe} detected!")
            # Display the result for 2 seconds before closing
            cv2.putText(frame, f"Swipe {swipe}!", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Swipe Detection', frame)
            cv2.waitKey(2000)  # Show result for 2 seconds
            break
        
        cv2.imshow('Swipe Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {str(e)}")

finally:
    cv2.destroyAllWindows()
    picam2.stop()
    print("Application closed")