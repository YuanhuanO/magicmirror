#!/usr/bin/python3
import time
from PIL import Image, ImageDraw, ImageFont
import ST7789
from datetime import datetime
import requests
import json

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

# Create ST7789 LCD display class
disp = ST7789.ST7789(
    height=240,
    width=320,
    rotation=180,  # Needed for correct orientation
    port=10,        # SPI port
    cs=0,          # SPI CS pin
    dc=9,          # Data/Command pin
    backlight=13,  # Backlight pin
    spi_speed_hz=60000000
)

# Initialize display
disp.begin()

# Create blank image for drawing
width = disp.width
height = disp.height

# Load font
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
    small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
except:
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()

def create_display():
    # Create new image with black background
    image = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, height), fill=(0, 0, 0))
    return image, draw

def update_display():
    try:
        while True:
            # Get current time
            current_time = datetime.now().strftime("%H:%M:%S")
            current_date = datetime.now().strftime("%Y-%m-%d")
            location = get_location()

            # Create new image for this update
            image, draw = create_display()

            # Draw time
            time_position = (width//2 - 80, height//4)
            draw.text(time_position, current_time, font=font, fill=(255, 255, 255))

            # Draw date
            date_position = (width//2 - 70, height//2)
            draw.text(date_position, current_date, font=small_font, fill=(0, 255, 0))

            # Draw location
            location_position = (width//2 - 100, 3*height//4)
            draw.text(location_position, location, font=small_font, fill=(0, 150, 255))

            # Display image
            disp.display(image)

            # Update every second
            time.sleep(1)

    except Exception as e:
        print(f"Error: {str(e)}")
        # Create error display
        image, draw = create_display()
        draw.text((10, height//2), f"Error: {str(e)}", font=small_font, fill=(255, 0, 0))
        disp.display(image)

if __name__ == "__main__":
    try:
        print("Starting clock display...")
        update_display()
    except KeyboardInterrupt:
        print("\nExiting program")
