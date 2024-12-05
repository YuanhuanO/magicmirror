import os
import warnings
warnings.filterwarnings('ignore')
import gpiod
import time
import cv2
import face_recognition
import pickle
import numpy as np
from picamera2 import Picamera2
import requests
import subprocess
from pathlib import Path
from datetime import datetime
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json  # Added for payload handling
import urllib.parse  # Added for URL encoding

class Config:
    """Configuration settings"""
    # GPIO settings
    SIG_PIN = 26
    DISTANCE_THRESHOLD = 50

    # Camera settings
    FACE_CAMERA_FORMAT = "XRGB8888"
    GESTURE_CAMERA_FORMAT = "RGB888"
    CAMERA_SIZE = (320, 240)  # Reduced resolution for better performance
    CV_SCALER = 2             # Adjusted scaler accordingly

    # MagicMirror settings
    MAGIC_MIRROR_PATH = Path('/home/yingshanhu/magicmirror/MagicMirror').resolve()
    MAGIC_MIRROR_HOST = "0.0.0.0"
    MAGIC_MIRROR_PORT = 8080
    BASE_URL = f"http://localhost:{MAGIC_MIRROR_PORT}"
    ALL_MODULES = ['weather', 'calendar', 'newsfeed', 'alert', 'helloworld']

    # Face recognition settings
    FACE_RECOGNITION_ATTEMPTS = 25
    FACE_RECOGNITION_TOLERANCE = 0.6
    ENCODINGS_PATH = "/home/yingshanhu/magicmirror/face_recognition/encodings.pickle"

    # Gesture Detection settings
    GESTURE_MODEL_PATH = "/home/yingshanhu/magicmirror/gesture_control/gesture_recognizer.task"
    GESTURE_CONFIDENCE_THRESHOLD = 0.5  # Decreased threshold for faster detection
    GESTURE_LABELS = [
        'Closed_Fist',
        'Open_Palm',
        'Pointing_Up',
        'Thumb_Down',
        'Thumb_Up',
        'Victory',
        'ILoveYou'
    ]

class MagicMirrorController:
    """Handles MagicMirror operations"""
    @staticmethod
    def ensure_running():
        def check_mm_process():
            try:
                result = subprocess.run(["pgrep", "-f", "node.*magicmirror"],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return result.returncode == 0
            except:
                return False

        try:
            if check_mm_process():
                print("[INFO] MagicMirror is already running")
                return True

            os.chdir(Config.MAGIC_MIRROR_PATH)
            subprocess.run(["pkill", "-f", "node.*magicmirror"], stderr=subprocess.DEVNULL)
            time.sleep(2)

            process = subprocess.Popen("npm run start", shell=True,
                                    cwd=Config.MAGIC_MIRROR_PATH,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    preexec_fn=os.setsid)

            start_time = time.time()
            while time.time() - start_time < 30:
                if check_mm_process():
                    print("[INFO] MagicMirror started successfully")
                    return True
                time.sleep(1)

            print("[ERROR] Failed to start MagicMirror within timeout period")
            return False

        except Exception as e:
            print(f"[ERROR] Could not start MagicMirror: {e}")
            return False

    @staticmethod
    def log_action(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    @staticmethod
    def send_alert(message, title="Authorization Alert", type="notification", time=None):
        alert_endpoint = f"{Config.BASE_URL}/api/module/alert/showalert"
        if time:
            alert_data = {
                "type": type,
                "title": title,
                "message": message,
                "timer": time,
            }
        else:
            alert_data = {
                "type": type,
                "title": title,
                "message": message,
            }
        try:
            response = requests.post(alert_endpoint, json=alert_data)
            MagicMirrorController.log_action(f"Alert sent: {message} (Status: {response.status_code})")
        except Exception as e:
            MagicMirrorController.log_action(f"Error sending alert: {str(e)}")

    @staticmethod
    def control_module(module_name, show=True):
        action = "show" if show else "hide"
        endpoint = f"{Config.BASE_URL}/api/module/{module_name}/{action}"
        try:
            response = requests.get(endpoint)
            MagicMirrorController.log_action(f"{action.capitalize()}ing {module_name}: Status {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            MagicMirrorController.log_action(f"Error controlling {module_name}: {str(e)}")
            return False

    @staticmethod
    def send_notification(notification, payload=None):
        """Send a notification to MagicMirror modules via MMM-Remote-Control"""
        # Build the endpoint URL
        endpoint = f"{Config.BASE_URL}/remote?action=NOTIFICATION&notification={notification}"
        if payload:
            # Encode the payload as a JSON string and URL-encode it
            payload_str = urllib.parse.quote(json.dumps(payload))
            endpoint += f"&payload={payload_str}"
        try:
            response = requests.get(endpoint)
            MagicMirrorController.log_action(f"Notification sent: {notification} (Status: {response.status_code})")
        except Exception as e:
            MagicMirrorController.log_action(f"Error sending notification: {str(e)}")

    @staticmethod
    def text_to_speech(message):
        """Use Raspberry Pi to broadcast"""
        try:
            subprocess.run(['espeak', '-s', '120', '-p', '75', '-a', '150', '-v', 'en+f2', message])
            MagicMirrorController.log_action(f"TTS: {message}")
        except Exception as e:
            MagicMirrorController.log_action(f"Error during TTS: {str(e)}")

class FaceRecognitionSystem:
    """Handles face recognition operations"""
    def __init__(self):
        self.load_encodings()
        self.init_camera()
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []

    def load_encodings(self):
        print("[INFO] Loading face encodings...")
        try:
            with open(Config.ENCODINGS_PATH, "rb") as f:
                data = pickle.loads(f.read())
            self.known_face_encodings = data["encodings"]
            self.known_face_names = data["names"]
        except FileNotFoundError:
            print("[ERROR] Encodings file not found. Ensure 'encodings.pickle' is in the working directory.")
            exit(1)

    def init_camera(self):
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(
            main={"format": Config.GESTURE_CAMERA_FORMAT, "size": Config.CAMERA_SIZE},
            controls={"FrameRate": 15}  # Limit frame rate to reduce CPU usage
        ))

    def process_frame(self, frame, attempts=Config.FACE_RECOGNITION_ATTEMPTS):
        detected_names = []

        for attempt in range(attempts):
            if attempt > 0:
                print("New attempt to find faces")
                frame = self.picam2.capture_array()
                time.sleep(0.5)

            resized_frame = cv2.resize(frame, (0, 0),
                                     fx=(1/Config.CV_SCALER),
                                     fy=(1/Config.CV_SCALER))
            rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            self.face_locations = face_recognition.face_locations(rgb_resized_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_resized_frame, self.face_locations)

            face_names = []
            for face_encoding in self.face_encodings:
                matches = face_recognition.compare_faces(
                    self.known_face_encodings,
                    face_encoding,
                    tolerance=Config.FACE_RECOGNITION_TOLERANCE
                )
                name = "Unknown"

                if matches:
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            print("Face found!")
                            name = self.known_face_names[best_match_index]

                face_names.append(name)

            for name in face_names:
                if name != "Unknown":
                    detected_names.append(name)

            if detected_names:
                return list(set(detected_names))

        return face_names if face_names else ["Unknown"]

class CombinedGestureDetector:
    def __init__(self, model_path=Config.GESTURE_MODEL_PATH, confidence_threshold=Config.GESTURE_CONFIDENCE_THRESHOLD):
        # # Initialize MediaPipe hands for swipe detection
        # self.mp_hands = mp.solutions.hands
        # self.hands = self.mp_hands.Hands(
        #     max_num_hands=1,
        #     min_detection_confidence=0.7,
        #     model_complexity=0  # Set to 0 for faster performance
        # )

        # # Swipe detection parameters
        # self.prev_x = None
        # self.movements = []
        # self.swipe_threshold = 70

        # Static gesture recognition parameters
        self.MODEL_PATH = model_path
        self.CONFIDENCE_THRESHOLD = confidence_threshold

        # Initialize the gesture recognizer
        if not os.path.exists(self.MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {self.MODEL_PATH}")

        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,  # Use IMAGE mode for synchronous processing
            num_hands=1
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

    # def detect_swipe(self, frame):
    #     """Detect swipe gestures"""
    #     if frame.shape[2] == 4:
    #         frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     results = self.hands.process(rgb_frame)

    #     swipe_detected = None
    #     frame_width = frame.shape[1]

    #     if results.multi_hand_landmarks:
    #         hand_landmarks = results.multi_hand_landmarks[0]

    #         # Track wrist position
    #         wrist = hand_landmarks.landmark[0]
    #         curr_x = wrist.x * frame_width

    #         if self.prev_x is not None:
    #             movement = curr_x - self.prev_x
    #             self.movements.append(movement)

    #             if len(self.movements) > 5:
    #                 self.movements.pop(0)

    #             total_movement = sum(self.movements)
    #             if abs(total_movement) > self.swipe_threshold:
    #                 swipe_detected = {
    #                     'type': 'swipe',
    #                     'direction': "right" if total_movement > 0 else "left"
    #                 }
    #                 # Reset movements after detecting a swipe
    #                 self.movements.clear()
    #                 self.prev_x = None
    #                 return swipe_detected

    #         self.prev_x = curr_x
    #     else:
    #         self.prev_x = None  # Reset if no hand is detected

    #     return swipe_detected

    def detect_gestures(self, frame):
        """Detect both swipe and static gestures, returning structured results"""
        gesture_result = {
            'type': None,
            'result': None
        }

        # Check for swipes first
        # swipe_result = self.detect_swipe(frame)
        # if swipe_result:
        #     gesture_result['type'] = 'swipe'
        #     gesture_result['result'] = swipe_result['direction']  # 'left' or 'right'
        #     return gesture_result

        # If no swipe, check for static gestures
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Use synchronous recognition
        result = self.recognizer.recognize(mp_image)

        # Process the result
        if result.gestures:
            gesture = result.gestures[0][0]
            if gesture.score >= self.CONFIDENCE_THRESHOLD:
                gesture_result['type'] = 'static'
                gesture_result['result'] = gesture.category_name
                return gesture_result

        return gesture_result

    def close(self):
        """Clean up resources"""
        self.recognizer.close()
        # self.hands.close()

class SensorController:
    """Handles sensor operations"""
    @staticmethod
    def init_gpio():
        try:
            chip = gpiod.Chip('gpiochip0')
            line = chip.get_line(Config.SIG_PIN)
            line.request(consumer="magic-mirror-sensor", type=gpiod.LINE_REQ_DIR_OUT)
            return line
        except Exception as e:
            print(f"[ERROR] Failed to initialize GPIO: {e}")
            return None

    @staticmethod
    def measure_distance(line, timeout=1.5):
        try:
            line.release()
            line.request(consumer="magic-mirror-sensor", type=gpiod.LINE_REQ_DIR_OUT)
            line.set_value(1)
            time.sleep(0.00001)
            line.set_value(0)

            line.release()
            line.request(consumer="magic-mirror-sensor", type=gpiod.LINE_REQ_DIR_IN)

            start_time = time.time()
            while line.get_value() == 0:
                if time.time() - start_time > timeout:
                    raise TimeoutError("Timeout waiting for echo start")

            echo_start = time.time()
            while line.get_value() == 1:
                if time.time() - echo_start > timeout:
                    raise TimeoutError("Timeout waiting for echo end")

            echo_end = time.time()
            distance = (echo_end - echo_start) * 34300 / 2
            return distance

        except Exception as e:
            print(f"[ERROR] Distance measurement failed: {str(e)}")
            return None

def handle_magicmirror_access(action, name=None, authorized_modules=[]):
    """Control module visibility based on action (authorize, unauthorized, logout)"""
    mm_controller = MagicMirrorController()
    alert_duration = 5000  # Duration in milliseconds

    if action == 'authorize':
        mm_controller.log_action(f"Authorized access for user: {name if name else 'Unknown'}")
        welcome_message = f"Welcome {name}!" if name else "Welcome authorized user!"
        mm_controller.send_alert(welcome_message, "Welcome", time=alert_duration)
        mm_controller.text_to_speech(welcome_message)

        # Hide all modules during the alert
        for module in authorized_modules:
            mm_controller.control_module(module, False)
        mm_controller.control_module("helloworld", False)

        # Record the time when the alert should end
        alert_end_time = time.time() + (alert_duration / 1000.0)
        return alert_end_time  # Return the alert end time

    elif action == 'unauthorized':
        mm_controller.log_action("Unauthorized access - Hiding authorized modules")
        unauthorized_message = f"Access Denied - Sorry {name}" if name else "UNAUTHORIZED ACCESS"
        mm_controller.send_alert(unauthorized_message, "Access Denied", time=alert_duration)
        mm_controller.text_to_speech(unauthorized_message)

        # Hide all modules during the alert
        for module in authorized_modules:
            mm_controller.control_module(module, False)
        # Show helloworld with "Get closer to activate"
        mm_controller.control_module("helloworld", True)
        mm_controller.send_notification("HELLOWORLD_UPDATE", {"message": "Get closer to activate"})
        mm_controller.text_to_speech("Get closer to activate")

        # Record the time when the alert should end
        alert_end_time = time.time() + (alert_duration / 1000.0)
        return alert_end_time  # Return the alert end time

    elif action == 'logout':
        mm_controller.log_action(f"User {name if name else 'Unknown'} logged out")
        logout_message = f"Logged out {name}!" if name else "User logged out!"
        mm_controller.send_alert(logout_message, "Goodbye", time=alert_duration)
        mm_controller.text_to_speech(logout_message)

        # Hide all modules during the alert
        for module in authorized_modules:
            mm_controller.control_module(module, False)
        # Show helloworld with "Get closer to activate the mirror"
        mm_controller.control_module("helloworld", True)
        mm_controller.send_notification("HELLOWORLD_UPDATE", {"message": "Get closer to activate the mirror"})
        mm_controller.text_to_speech("Get closer to activate the mirror")

        # Record the time when the alert should end
        alert_end_time = time.time() + (alert_duration / 1000.0)
        return alert_end_time  # Return the alert end time

    else:
        mm_controller.log_action(f"Unknown action: {action}")
        return None

def main():
    """Main program execution with state management"""
    if not MagicMirrorController.ensure_running():
        print("[ERROR] MagicMirror could not be started.")
        return

    line = SensorController.init_gpio()
    if line is None:
        print("[ERROR] GPIO initialization failed.")
        return

    face_system = FaceRecognitionSystem()
    gesture_detector = CombinedGestureDetector()
    print("[INFO] Face recognition and gesture detection system initialized.")

    # Magic mirror state
    current_state = "OFF"
    TIMEOUT_DURATION = 30 * 60  # 30 minutes in seconds
    last_activity_time = time.time()
    last_gesture_time = 0
    GESTURE_COOLDOWN = 1.0  # Decreased cooldown for faster response

    # Define modules
    authorized_modules = ['newsfeed', 'calendar', 'MMM-Todoist']  # Corrected module loop order
    default_modules = []  # No default modules
    all_modules = authorized_modules + default_modules
    current_module_index = 0
    user_name = None  # Keep track of the authorized user's name
    authorized = False  # Track authorization status

    # Hide all managed modules initially
    for module in all_modules:
        MagicMirrorController.control_module(module, False)
    # Show helloworld module with "Get closer to activate" message
    MagicMirrorController.control_module("helloworld", True)
    MagicMirrorController.send_notification("HELLOWORLD_UPDATE", {"message": "Get closer to activate"})
    MagicMirrorController.text_to_speech("Get closer to activate")

    alert_end_time = None  # Initialize alert_end_time

    try:
        while True:
            current_time = time.time()

            # Check if it's time to show the modules again after alert
            if alert_end_time and current_time >= alert_end_time:
                mm_controller = MagicMirrorController()
                if authorized:
                    # After alert ends, show only 'newsfeed' and 'helloworld' modules
                    for module in authorized_modules:
                        mm_controller.control_module(module, False)
                    mm_controller.control_module("newsfeed", True)
                    mm_controller.control_module("helloworld", True)
                    # Set current_module_index to 0 (newsfeed)
                    current_module_index = 0
                    # Send initial message to helloworld module
                    message = f"Welcome {user_name}! Displaying newsfeed."
                    mm_controller.send_notification("HELLOWORLD_UPDATE", {"message": message})
                    mm_controller.text_to_speech(message)
                    current_state = 'ON'
                else:
                    # Remain in OFF state (helloworld is already shown with message)
                    current_state = 'OFF'
                alert_end_time = None  # Reset alert_end_time

            if current_state == "OFF":
                # Only measure distance when in OFF state
                distance = SensorController.measure_distance(line)
                if not distance or distance > Config.DISTANCE_THRESHOLD:
                    print("[INFO] Distance: No object in range.")
                    time.sleep(0.5)
                    continue
                print(f"[INFO] Distance: {distance:.2f} cm. Object detected.")

                # Face recognition
                face_system.picam2.start()
                camera_frame = face_system.picam2.capture_array()
                print("[INFO] Camera started for face recognition.")
                face_names = face_system.process_frame(camera_frame)
                authorized = any(name != "Unknown" for name in face_names)
                user_name = face_names[0] if authorized else None

                face_system.picam2.stop()
                print("[INFO] Face recognition complete, camera stopped.")

                if authorized:
                    alert_end_time = handle_magicmirror_access(action='authorize', name=user_name, authorized_modules=authorized_modules)
                else:
                    alert_end_time = handle_magicmirror_access(action='unauthorized', name=user_name, authorized_modules=authorized_modules)

                last_activity_time = current_time
                current_state = 'ALERT'
                time.sleep(0.5)

            elif current_state == 'ALERT':
                # Wait until alert ends
                time.sleep(0.1)

            elif current_state == "ON":
                if current_time - last_activity_time >= TIMEOUT_DURATION:
                    print("[INFO] Session timeout, switching to OFF state")
                    alert_end_time = handle_magicmirror_access(action='logout', name=user_name, authorized_modules=authorized_modules)
                    authorized = False
                    current_state = 'ALERT'
                    continue

                face_system.picam2.start()  # Start the camera once
                try:
                    while current_state == "ON":
                        camera_frame = face_system.picam2.capture_array()
                        gesture = gesture_detector.detect_gestures(camera_frame)
                        current_time = time.time()

                        # Handle detected gesture
                        if ((gesture["type"] == 'static' and gesture["result"] in Config.GESTURE_LABELS) or
                            (gesture["type"] == 'swipe')) and \
                           (current_time - last_gesture_time) >= GESTURE_COOLDOWN:
                            last_gesture_time = current_time
                            last_activity_time = current_time
                            print(f"{gesture['type']} gesture: {gesture['result']}!")

                            if gesture['type'] == 'static' and gesture['result'] == 'Closed_Fist':
                                # Logout user
                                alert_end_time = handle_magicmirror_access(action='logout', name=user_name, authorized_modules=authorized_modules)
                                authorized = False
                                # Hide all authorized modules
                                for module in authorized_modules:
                                    MagicMirrorController.control_module(module, False)
                                # 'helloworld' is handled in handle_magicmirror_access (shown with message)
                                current_state = 'ALERT'
                                print("[INFO] User logged out due to Closed_Fist gesture.")
                                break
                            else:
                                # Any other gesture switches modules
                                # Hide current module
                                MagicMirrorController.control_module(authorized_modules[current_module_index], False)
                                # Update index to next module
                                current_module_index = (current_module_index + 1) % len(authorized_modules)
                                # Show next module
                                MagicMirrorController.control_module(authorized_modules[current_module_index], True)
                                # Log the module switch
                                module_name = authorized_modules[current_module_index]
                                MagicMirrorController.log_action(f"Switched to {module_name}")
                                # Update helloworld module with the message
                                message = f"Switched to {module_name}"
                                mm_controller.send_notification("HELLOWORLD_UPDATE", {"message": message})
                                mm_controller.text_to_speech(message)
                        # Small delay to prevent CPU overload
                        time.sleep(0.03)
                finally:
                    face_system.picam2.stop()  # Ensure the camera is stopped when exiting the loop
                print("[INFO] Gesture detection complete, camera stopped.")
                time.sleep(0.5)

    except KeyboardInterrupt:
        print(f"[INFO] Stopping system.")
    finally:
        if line:
            line.release()
        face_system.picam2.stop()
        gesture_detector.close()
        print("[INFO] Resources released.")

if __name__ == "__main__":
    main()
