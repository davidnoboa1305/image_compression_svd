import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('images.jpeg')

img.show()

image_matrix = np.array(img)

print(f"Matrix shape: {image_matrix.shape}")  # e.g., (512, 512)
print(f"Data type: {image_matrix.dtype}")     # typically uint8 (0-255)
print(f"Value range: {image_matrix.min()} to {image_matrix.max()}")

