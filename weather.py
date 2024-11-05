import time
import requests
import geocoder
import json
import ST7789
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

class WeatherDisplay:
    def __init__(self):
        # Initialize the display
        self.disp = ST7789.ST7789(
            height=240,
            width=320,
            rotation=180,  # Adjust this if display is upside down
            port=0,
            cs=ST7789.BG_SPI_CS_FRONT,
            dc=9,
            backlight=19,
            spi_speed_hz=80 * 1000 * 1000,
            offset_left=0,
            offset_top=0
        )

        # Initialize display
        self.disp.begin()
        
        # Load fonts (assuming these files exist in the same directory)
        try:
            self.font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
            self.font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except OSError:
            # Fallback to default font if custom font not found
            self.font_large = ImageFont.load_default()
            self.font_medium = ImageFont.load_default()
            self.font_small = ImageFont.load_default()

        self.api_key = "419d2064c27c92162bc69f88af6cd7b2"  # OpenWeatherMap API key
        
    def get_location(self):
        """Get current location using IP geolocation"""
        try:
            g = geocoder.ip('me')
            return g.latlng
        except Exception as e:
            print(f"Location error: {e}")
            return None
            
    def get_weather(self, lat, lon):
        """Fetch weather data from OpenWeatherMap"""
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Weather API error: {e}")
            return None

    def create_weather_image(self, weather_data):
        """Create weather display image"""
        # Create new image with white background
        image = Image.new('RGB', (320, 240), 'white')
        draw = ImageDraw.Draw(image)
        
        if not weather_data:
            # Display error message if no weather data
            draw.text((10, 100), "Weather data", fill='black', font=self.font_medium)
            draw.text((10, 130), "unavailable", fill='black', font=self.font_medium)
            return image
            
        # Extract weather data
        temp = round(weather_data['main']['temp'])
        humidity = weather_data['main']['humidity']
        condition = weather_data['weather'][0]['main']
        description = weather_data['weather'][0]['description'].title()
        
        # Get current time
        current_time = datetime.now().strftime("%H:%M")
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Draw time and date
        draw.text((10, 10), current_time, fill='black', font=self.font_large)
        draw.text((160, 20), current_date, fill='black', font=self.font_small)
        
        # Draw temperature
        temp_text = f"{temp}°C"
        draw.text((10, 70), temp_text, fill='black', font=self.font_large)
        
        # Draw weather condition
        draw.text((10, 130), condition, fill='black', font=self.font_medium)
        draw.text((10, 160), description, fill='black', font=self.font_small)
        
        # Draw humidity
        humidity_text = f"Humidity: {humidity}%"
        draw.text((10, 190), humidity_text, fill='black', font=self.font_medium)
        
        # Draw border
        draw.rectangle([(0, 0), (319, 239)], outline='black', width=2)
        
        return image

    def display_weather(self, weather_data):
        """Display weather on LCD"""
        image = self.create_weather_image(weather_data)
        self.disp.display(image)
        
    def run(self):
        """Main loop"""
        print("Starting Weather Display...")
        
        try:
            while True:
                location = self.get_location()
                if location:
                    lat, lon = location
                    weather = self.get_weather(lat, lon)
                    self.display_weather(weather)
                else:
                    # Create error image
                    image = Image.new('RGB', (320, 240), 'white')
                    draw = ImageDraw.Draw(image)
                    draw.text((10, 100), "Location services", fill='black', font=self.font_medium)
                    draw.text((10, 130), "unavailable", fill='black', font=self.font_medium)
                    self.disp.display(image)
                
                # Wait 5 minutes before next update
                time.sleep(300)
                
        except KeyboardInterrupt:
            # Clear display on exit
            image = Image.new('RGB', (320, 240), 'black')
            self.disp.display(image)
            print("\nDisplay stopped by user")

if __name__ == "__main__":
    display = WeatherDisplay()
    display.run()
