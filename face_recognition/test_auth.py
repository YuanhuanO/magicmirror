import gpiod
import time
import cv2
import face_recognition
import pickle
import numpy as np
from picamera2 import Picamera2
import socketio
import requests
sio = socketio.Client()

@sio.on("connect")
def on_connect():
    print("[INFO] Connected to MagicMirror WebSocket")

@sio.on("disconnect")
def on_disconnect():
    print("[INFO] Disconnected from MagicMirror WebSocket")

# 连接到 MagicMirror 的 WebSocket 服务
sio.connect("http://localhost:8080")  # 假设 MagicMirror 在本地运行

def send_authorized_status(is_authorized):
    """向 MagicMirror 发送授权状态"""
    if is_authorized:
        sio.emit("SHOW_AUTHORIZED")
    else:
        sio.emit("SHOW_UNAUTHORIZED")

def send_auth_status(is_authorized):
    endpoint = "/MMM-AuthDisplay"
    data = {"status": "authorized" if is_authorized else "unauthorized"}
    try:
        requests.post(f"{MAGICMIRROR_URL}{endpoint}", json=data)
        print(f"Status sent: {'authorized' if is_authorized else 'unauthorized'}")
    except Exception as e:
        print(f"Error sending status: {e}")

# GPIO pin for ultrasonic sensor
SIG_PIN = 26

# Load pre-trained face encodings
print("[INFO] Loading face encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 480)}))

# Variables for face recognition
cv_scaler = 4
face_locations = []
face_encodings = []
face_names = []

def init_gpio():
    """Initialize GPIO line using gpiod."""
    try:
        chip = gpiod.Chip('gpiochip0')
        line = chip.get_line(SIG_PIN)
        line.request(consumer="easyTest", type=gpiod.LINE_REQ_DIR_OUT)
        return line
    except Exception as e:
        print(f"[ERROR] Failed to initialize GPIO: {e}")
        return None

def measure_distance(line, timeout=1.5):
    """Measure distance using the ultrasonic sensor."""
    try:
        # Set line to output mode and send a 10-microsecond trigger pulse
        line.release()
        line.request(consumer="easyTest", type=gpiod.LINE_REQ_DIR_OUT)
        line.set_value(1)
        time.sleep(0.00001)  # 10 µs pulse
        line.set_value(0)

        # Switch line to input mode for echo
        line.release()
        line.request(consumer="easyTest", type=gpiod.LINE_REQ_DIR_IN)

        # Wait for echo signal to start
        start_time = time.time()
        while line.get_value() == 0:
            if time.time() - start_time > timeout:
                raise TimeoutError("Timeout waiting for echo start")

        echo_start = time.time()

        # Wait for echo signal to end
        while line.get_value() == 1:
            if time.time() - echo_start > timeout:
                raise TimeoutError("Timeout waiting for echo end")

        echo_end = time.time()

        # Calculate distance
        elapsed_time = echo_end - echo_start
        distance = (elapsed_time * 34300) / 2  # Distance in cm
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

def process_frame(frame):
    """Process a frame to detect and recognize faces."""
    global face_locations, face_encodings, face_names 
    resized_frame = cv2.resize(frame, (0, 0), fx=(1 / cv_scaler), fy=(1 / cv_scaler))
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations)

    face_names = []
    is_authorized = False
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            is_authorized = True
        face_names.append(name)
    send_auth_status(is_authorized)
    return frame

def draw_results(frame):
    """Draw the results on the frame."""
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return frame

def run_sensor_with_camera():
    """Run ultrasonic sensor and activate camera with face recognition."""
    line = init_gpio()
    if line is None:
        print("[ERROR] GPIO initialization failed.")
        return

    try:
        picam2.start()
        print("[INFO] Sensor and camera are running. Press Ctrl+C to stop.")
        while True:
            distance = measure_distance(line)
            if distance:
                print(f"[INFO] Distance: {distance:.2f} cm")
                if distance <= 50:
                    print("[INFO] Object detected within range.")
                    frame = picam2.capture_array()

                    # Process and recognize faces in the frame
                    processed_frame = process_frame(frame)
                    display_frame = draw_results(processed_frame)

                    # Display the camera feed
                    cv2.imshow("Camera Feed", display_frame)


                    # Exit the loop if 'q' is pressed
                    if cv2.waitKey(1) == ord("q"):
                        break
            else:
                print("[INFO] No valid distance measured.")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[INFO] Stopping sensor and camera.")
    finally:
        if line:
            line.release()
        picam2.stop()
        cv2.destroyAllWindows()
        print("[INFO] Resources released.")

if __name__ == "__main__":
    run_sensor_with_camera()
