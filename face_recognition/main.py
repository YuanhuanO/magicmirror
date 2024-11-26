import os
import gpiod
import time
import cv2
import face_recognition
import pickle
import numpy as np
from picamera2 import Picamera2
import threading
import requests
import subprocess
from pathlib import Path
import requests
import time
from datetime import datetime

# GPIO pin for ultrasonic sensor
SIG_PIN = 26

# MagicMirror configuration
MAGIC_MIRROR_PATH = Path('/home/yingshanhu/magicmirror/MagicMirror').resolve()
MAGIC_MIRROR_HOST = "0.0.0.0"
MAGIC_MIRROR_PORT = 8080
MAGIC_MIRROR_URL = f"http://{MAGIC_MIRROR_HOST}:{MAGIC_MIRROR_PORT}"

# Load pre-trained face encodings
print("[INFO] Loading face encodings...")
try:
    with open("encodings.pickle", "rb") as f:
        data = pickle.loads(f.read())
    known_face_encodings = data["encodings"]
    known_face_names = data["names"]
except FileNotFoundError:
    print("[ERROR] Encodings file not found. Ensure 'encodings.pickle' is in the working directory.")
    exit(1)

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 480)}))

# Variables for face recognition
cv_scaler = 4
face_locations = []
face_encodings = []
face_names = []

def ensure_magicmirror_running():
    """Ensure MagicMirror is running."""
    def check_mm_process():
        try:
            # Check if MagicMirror process is running
            result = subprocess.run(
                ["pgrep", "-f", "node.*magicmirror"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return result.returncode == 0
        except:
            return False

    try:
        if check_mm_process():
            print("[INFO] MagicMirror is already running")
            return True

        os.chdir(MAGIC_MIRROR_PATH)

        # Kill any existing MagicMirror processes
        subprocess.run(["pkill", "-f", "node.*magicmirror"], stderr=subprocess.DEVNULL)
        time.sleep(2)

        # Start MagicMirror in the background
        process = subprocess.Popen(
            "npm run start",
            shell=True,
            cwd=MAGIC_MIRROR_PATH,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )

        # Wait for the process to start
        print("[INFO] Waiting for MagicMirror to start...")
        start_time = time.time()
        while time.time() - start_time < 30:  # 30 second timeout
            if check_mm_process():
                print("[INFO] MagicMirror started successfully")
                return True
            time.sleep(1)

        # If we get here, the process didn't start in time
        print("[ERROR] Failed to start MagicMirror within timeout period")
        return False

    except Exception as e:
        print(f"[ERROR] Could not start MagicMirror: {e}")
        return False

def init_gpio():
    """Initialize GPIO line using gpiod."""
    try:
        chip = gpiod.Chip('gpiochip0')
        line = chip.get_line(SIG_PIN)
        line.request(consumer="magic-mirror-sensor", type=gpiod.LINE_REQ_DIR_OUT)
        return line
    except Exception as e:
        print(f"[ERROR] Failed to initialize GPIO: {e}")
        return None

def measure_distance(line, timeout=1.5):
    """Measure distance using the ultrasonic sensor."""
    try:
        line.release()
        line.request(consumer="magic-mirror-sensor", type=gpiod.LINE_REQ_DIR_OUT)
        line.set_value(1)
        time.sleep(0.00001)  # 10 µs pulse
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
        elapsed_time = echo_end - echo_start
        distance = (elapsed_time * 34300) / 2
        return distance
    except TimeoutError as e:
        print(f"[ERROR] Timeout: {e}")
        return None
    except PermissionError as e:
        print(f"[ERROR] Permission issue: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return None

def process_frame(frame, attempts=25):
    """Process a frame to detect and recognize faces with multiple attempts."""
    global face_locations, face_encodings, face_names

    detected_names = []

    # Try face detection multiple times
    for attempt in range(attempts):
        # Get a fresh frame for each attempt
        if attempt > 0:
            print("new attempt to find faces")
            frame = picam2.capture_array()
            time.sleep(0.5)  # Short delay between attempts

        resized_frame = cv2.resize(frame, (0, 0), fx=(1 / cv_scaler), fy=(1 / cv_scaler))
        rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_resized_frame)
        face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    print("face found!")
                    name = known_face_names[best_match_index]

            face_names.append(name)

        # If we found any known faces, add them to our list
        for name in face_names:
            if name != "Unknown":
                detected_names.append(name)

        # If we've found any known faces, we can return early
        if detected_names:
            return list(set(detected_names))

    # If we haven't found any known faces after all attempts, return the last face_names
    return face_names if face_names else ["Unknown"]

def handle_magicmirror_access(authorized, name=None):
    """Control module visibility based on authorization status and show personalized messages"""
    BASE_URL = "http://localhost:8080"
    ALL_MODULES = ['weather', 'calendar', 'newsfeed', 'alert']
    
    def log_action(message):
        """Print timestamped log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def send_alert(message, title="Authorization Alert"):
        """Send alert message to MagicMirror"""
        alert_endpoint = f"{BASE_URL}/api/module/alert/showalert"
        alert_data = {
            "type": "notification",
            "title": title,
            "message": message,
            "timer": 10000  # 10 seconds
        }
        try:
            response = requests.post(alert_endpoint, json=alert_data)
            log_action(f"Alert sent: {message} (Status: {response.status_code})")
        except Exception as e:
            log_action(f"Error sending alert: {str(e)}")
    
    def control_module(module_name, show=True):
        """Control module visibility"""
        action = "show" if show else "hide"
        endpoint = f"{BASE_URL}/api/module/{module_name}/{action}"
        try:
            response = requests.get(endpoint)
            log_action(f"{action.capitalize()}ing {module_name}: Status {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            log_action(f"Error controlling {module_name}: {str(e)}")
            return False

    log_action("Starting authorization-based module control")
    log_action(f"Authorization status: {'Authorized' if authorized else 'Unauthorized'}")
    print("-" * 50)

    if authorized:
        # Show authorized modules and personalized welcome
        log_action(f"Authorized access for user: {name if name else 'Unknown'}")
        modules_to_show = ['weather', 'calendar', 'newsfeed', 'alert']
        
        welcome_message = f"Welcome {name}!" if name else "Welcome authorized user!"
        send_alert(welcome_message, "Welcome")
        
        for module in ALL_MODULES:
            if module in modules_to_show:
                control_module(module, show=True)
            else:
                control_module(module, show=False)
                
    else:
        # Hide all modules except alert with unauthorized message
        log_action("Unauthorized access - Hiding all modules except alert")
        unauthorized_message = f"Access Denied - Sorry {name}" if name else "UNAUTHORIZED ACCESS"
        send_alert(unauthorized_message, "Access Denied")
        
        for module in ALL_MODULES:
            if module != 'alert':
                control_module(module, show=False)
            else:
                control_module(module, show=True)

def run_sensor_with_camera():
    """Run ultrasonic sensor and activate camera with face recognition."""
    line = init_gpio()
    if line is None:
        print("[ERROR] GPIO initialization failed.")
        return

    try:
        picam2.start()
        print("[INFO] Sensor and camera are running.")
        while True:
            distance = measure_distance(line)
            print("now trying to measure distance")
            if distance and distance <= 50:
                print(f"[INFO] Distance: {distance:.2f} cm. Object detected.")
                frame = picam2.capture_array()
                face_names = process_frame(frame)
                authorized = any(name != "Unknown" for name in face_names)
                handle_magicmirror_access(authorized, face_names[0] if authorized else None)
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[INFO] Stopping sensor and camera.")
    finally:
        if line:
            line.release()
        picam2.stop()
        print("[INFO] Resources released.")

if __name__ == "__main__":
    if ensure_magicmirror_running():
        run_sensor_with_camera()
    else:
        print("[ERROR] MagicMirror could not be started.")

