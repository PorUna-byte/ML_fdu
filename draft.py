import os
import numpy as np
from PIL import Image
image_path = 'data/train/sample/20220513_00001.jpg'
image = Image.open(image_path).convert('L')  # Convert to grayscale
image = image.resize((512, 512))
image = np.array(image)
print(image.shape)