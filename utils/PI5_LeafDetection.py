from picamera2 import Picamera2, Preview
import cv2
import numpy as np
from time import sleep

def detect_green_color(image):
    # Convert image to HSV color space for easier color detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define range of green color in HSV
    lower_orange = np.array([44, 100, 100])
    upper_orange = np.array([86, 255, 255])
    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    # Check if there is any green in the frame
    return np.sum(mask) > 0

def setup_camera(index):
    # Initialize the camera
    picam = Picamera2(index)
    # Start preview for configuration
    picam.start_preview(Preview.QTGL)
    # Create a preview configuration
    config = picam.create_preview_configuration()
    # Set up the camera with a specific resolution
    config['main']['size'] = (1024, 1024)  # Set the desired resolution
    # Configure the camera with the preview configuration
    picam.configure(config)
    # Start the camera
    picam.start()
    return picam

def capture_image(picam, file_path):
    picam.capture_file(file_path)

def monitor_and_capture(picam0, picam1, base_file_path0, base_file_path1, detection_number):
    try:
        while True:
            # Capture an image array for processing
            buffer0 = picam0.capture_array()
            buffer1 = picam1.capture_array()
            # Check for the color orange in the image
            if detect_green_color(buffer0) or detect_green_color(buffer1):
                print(f"Leaf detected!(Detection #{detection_number})...")
                sleep(1.6)
                print(f"Capturing high-resolution images (Detection #{detection_number})...")
                capture_image(picam0, f"{base_file_path0}_{detection_number}_1.jpg")
                capture_image(picam1, f"{base_file_path1}_{detection_number}_1.jpg")
                sleep(0.2)
                capture_image(picam0, f"{base_file_path0}_{detection_number}_2.jpg")
                capture_image(picam1, f"{base_file_path1}_{detection_number}_2.jpg")
                sleep(0.2)
                capture_image(picam0, f"{base_file_path0}_{detection_number}_3.jpg")
                capture_image(picam1, f"{base_file_path1}_{detection_number}_3.jpg")
                detection_number += 1
                sleep(4)
    finally:
        picam0.stop()
        picam1.stop()
        picam0.stop_preview()
        picam1.stop_preview()

def main():
    picam0 = setup_camera(0)
    picam1 = setup_camera(1)
    
    base_file_path_camera0 = "/home/"
    base_file_path_camera1 = "/home/"
    
    detection_number =1  # Initialize detection number
    
    # Monitoring and capturing for both cameras
    monitor_and_capture(picam0, picam1, base_file_path_camera0, base_file_path_camera1, detection_number)

if __name__ == "__main__":
    main()

