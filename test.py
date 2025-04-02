import cv2
import numpy as np

image_path = "scene_l_0003.jpeg"
image = cv2.imread(image_path)

print(image.shape[:2])