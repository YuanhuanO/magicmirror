from displayhatmini import DisplayHATMini
from PIL import Image, ImageDraw, ImageFont
import time
import datetime

img = Image.new('RGB', (320, 240), color=(0, 0, 0))
display = DisplayHATMini(img)

draw = ImageDraw.Draw(img)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 50)

try:
    while True:
        draw.rectangle((0, 0, 320, 240), (0, 0, 0))
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        draw.text((60, 100), current_time, font=font, fill=(255, 255, 255))
        display.display(img)
        time.sleep(1)

except KeyboardInterrupt:
    display.clear()
