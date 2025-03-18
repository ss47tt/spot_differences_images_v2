import cv2
import numpy as np
import os

def rotate_image(image, angle):
    # Get the image dimensions (height, width)
    height, width = image.shape[:2]
    
    # Calculate the center of the image
    center = (width // 2, height // 2)
    
    # Get the rotation matrix using cv2.getRotationMatrix2D
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # angle, scale
    
    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    return rotated_image

# Example usage
image_path = 'test_v2/test_0.jpeg'  # Input image path
output_path = 'test_v2_rotated/test_0.png'  # Output image path

# Check if the image is loaded successfully
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Unable to load image at {image_path}")
else:
    # Rotate the image by 10 degrees
    rotated_image = rotate_image(image, 3)
    
    # Ensure the output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the rotated image
    if cv2.imwrite(output_path, rotated_image):
        print(f"Image saved successfully to {output_path}")
    else:
        print("Error: Unable to save the rotated image.")