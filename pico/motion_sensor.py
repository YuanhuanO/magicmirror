from machine import Pin
from utime import sleep, ticks_ms

# Setup PIR sensor on pin 18
miniPir = Pin(18, Pin.IN)
# Setup LED on pin 16
led = Pin(16, Pin.OUT)

# Parameters for close detection
REQUIRED_DETECTIONS = 5  # Number of detections needed to trigger
DETECTION_WINDOW = 500   # Time window in milliseconds (0.5 seconds)
detections = 0
last_detection_time = 0

while True:
    led.value(0) 
    if miniPir.value() == 1:
        current_time = ticks_ms()
        
        # Check if this detection is within our time window
        if current_time - last_detection_time < DETECTION_WINDOW:
            detections += 1
        else:
            detections = 1
        
        last_detection_time = current_time
        
        # Only trigger if we get multiple quick detections (indicating close proximity)
        if detections >= REQUIRED_DETECTIONS:
            print('Close Motion Detected!')
            led.value(1)    # Turn LED on
            sleep(5)        # Keep LED on for 5 seconds
            led.value(0)    # Turn LED off
            detections = 0  # Reset detection count
            
    sleep(0.05)  # Shorter sleep for more responsive detection