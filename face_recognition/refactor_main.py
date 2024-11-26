import os
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

class Config:
    """Configuration settings"""
    # GPIO settings
    SIG_PIN = 26
    DISTANCE_THRESHOLD = 50
    
    # Camera settings
    CAMERA_FORMAT = "XRGB8888"
    CAMERA_SIZE = (640, 480)
    CV_SCALER = 4
    
    # MagicMirror settings
    MAGIC_MIRROR_PATH = Path('/home/yingshanhu/magicmirror/MagicMirror').resolve()
    MAGIC_MIRROR_HOST = "0.0.0.0"
    MAGIC_MIRROR_PORT = 8080
    BASE_URL = f"http://localhost:{MAGIC_MIRROR_PORT}"
    ALL_MODULES = ['weather', 'calendar', 'newsfeed', 'alert']
    
    # Face recognition settings
    FACE_RECOGNITION_ATTEMPTS = 25
    FACE_RECOGNITION_TOLERANCE = 0.6
    ENCODINGS_PATH = "/home/yingshanhu/magicmirror/face_recognition/encodings.pickle"

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
    def send_alert(message, title="Authorization Alert"):
        alert_endpoint = f"{Config.BASE_URL}/api/module/alert/showalert"
        alert_data = {
            "type": "notification",
            "title": title,
            "message": message,
            "timer": 10000
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
            main={"format": Config.CAMERA_FORMAT, "size": Config.CAMERA_SIZE}))

    def process_frame(self, frame, attempts=Config.FACE_RECOGNITION_ATTEMPTS):
        detected_names = []
        
        for attempt in range(attempts):
            if attempt > 0:
                print("new attempt to find faces")
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
                            print("face found!")
                            name = self.known_face_names[best_match_index]

                face_names.append(name)

            for name in face_names:
                if name != "Unknown":
                    detected_names.append(name)

            if detected_names:
                return list(set(detected_names))

        return face_names if face_names else ["Unknown"]

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

def handle_magicmirror_access(authorized, name=None):
    """Control module visibility based on authorization status"""
    mm_controller = MagicMirrorController()
    mm_controller.log_action(f"Authorization status: {'Authorized' if authorized else 'Unauthorized'}")
    
    if authorized:
        mm_controller.log_action(f"Authorized access for user: {name if name else 'Unknown'}")
        welcome_message = f"Welcome {name}!" if name else "Welcome authorized user!"
        mm_controller.send_alert(welcome_message, "Welcome")
        
        for module in Config.ALL_MODULES:
            mm_controller.control_module(module, show=True)
    else:
        mm_controller.log_action("Unauthorized access - Hiding all modules except alert")
        unauthorized_message = f"Access Denied - Sorry {name}" if name else "UNAUTHORIZED ACCESS"
        mm_controller.send_alert(unauthorized_message, "Access Denied")
        
        for module in Config.ALL_MODULES:
            mm_controller.control_module(module, show=(module == 'alert'))

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
    face_system.picam2.start()
    print("[INFO] System initialized and running.")

    # State management
    current_state = "OFF"
    last_detection_time = 0
    DETECTION_COOLDOWN = 5  # Seconds between detection attempts
    TIMEOUT_DURATION = 30 * 60  # 30 minutes in seconds
    current_auth_status = None
    current_user = None

    try:
        while True:
            current_time = time.time()
            distance = SensorController.measure_distance(line)
            print(f"Current state: {current_state}, trying to measure distance")

            if distance and distance <= Config.DISTANCE_THRESHOLD:
                if current_state == "OFF" or (current_time - last_detection_time >= DETECTION_COOLDOWN):
                    print(f"[INFO] Distance: {distance:.2f} cm. Object detected.")
                    frame = face_system.picam2.capture_array()
                    face_names = face_system.process_frame(frame)
                    authorized = any(name != "Unknown" for name in face_names)
                    user_name = face_names[0] if authorized else None

                    # Only update if authorization status changes
                    if authorized != current_auth_status or user_name != current_user:
                        handle_magicmirror_access(authorized, user_name)
                        current_auth_status = authorized
                        current_user = user_name
                        current_state = "ON"
                        print(f"[INFO] State changed to ON - User: {user_name if user_name else 'Unknown'}")

                    last_detection_time = current_time
            else:
                # If no detection for 30 minutes, switch to OFF state
                if current_state == "ON" and (current_time - last_detection_time >= TIMEOUT_DURATION):
                    print("[INFO] No detection for 30 minutes, switching to OFF state")
                    handle_magicmirror_access(False)  # Hide modules
                    current_state = "OFF"
                    current_auth_status = None
                    current_user = None

            time.sleep(0.5)  # Prevent CPU overload

    except KeyboardInterrupt:
        print("[INFO] Stopping system.")
    finally:
        if line:
            line.release()
        face_system.picam2.stop()
        print("[INFO] Resources released.")
        
if __name__ == "__main__":
    main()