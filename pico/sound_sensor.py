from lcd1602 import LCD1602
from machine import I2C, Pin, ADC
import utime
import network
import requests
import time
import requests
import json


i2c_lcd = I2C(1, scl=Pin(7), sda=Pin(6), freq=400000)  
d = LCD1602(i2c_lcd, 2, 16)
utime.sleep(1)

SOUND_SENSOR = ADC(1)  

display_mode = "datetime"
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect('yingshanhu@172.20.106.105', 'yingshanhu')
def get_location():
    try:
        # Using ip-api.com for location data (free, no API key required)
        response = requests.get('http://ip-api.com/json/')
        data = response.json()
        if response.status_code == 200:
            return f"{data['city']}, {data['country']}"
        return "Location Unavailable"
    except:
        return "Location Unavailable"

def get_current_datetime():
    year, month, day, hour, minute, second, *_ = utime.localtime()
    return f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"

while True:
    average = 0
    for i in range (1000):
        noise = SOUND_SENSOR.read_u16()/256
        average += noise
    noise = average/1000
    print (noise)
    if noise >= 30:
        display_mode = "weather" if display_mode == "datetime" else "datetime"
        utime.sleep (0.6)
    d.clear()
    d.home()
    if display_mode == "datetime":
        current_datetime = get_current_datetime()
        d.print("Date/Time:")
        d.setCursor(0, 1)
        d.print(current_datetime)
    
    elif display_mode == "weather":
        temperature = get_location()
        d.print("Temp (C):")
        d.setCursor(0, 1)
        d.print(str(temperature) if temperature is not None else "N/A")
    
