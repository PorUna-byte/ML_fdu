import os
import numpy as np
from PIL import Image
from tqdm import tqdm


# Output directory to save the preprocessed data
train_dir = 'data/train'
val_dir = 'data/val'
test_dir = 'data/test'

# Dimensions to resize images to
img_height, img_width = 256, 256

# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((img_width, img_height))
    image = np.array(image) / 255.0  # Normalize to range [0, 1]
    return image

for cdir in [train_dir, val_dir, test_dir]: 
    sample_dir = cdir + '/sample'
    mask_dir = cdir + '/mask'
    output_dir = cdir
    # Get list of sample image files
    sample_files = sorted([f for f in os.listdir(sample_dir) if os.path.isfile(os.path.join(sample_dir, f))])
    
    # Initialize arrays to hold the preprocessed data
    X = np.empty((len(sample_files), img_height, img_width, 1), dtype=np.float32)
    Y = np.empty(len(sample_files), dtype=np.int32)
    Z = np.empty((len(sample_files), img_height, img_width, 1), dtype=np.float32)
    
    # Loop through the sample images and preprocess them
    for i, sample_file in enumerate(tqdm(sample_files, desc='Preprocessing images')):
        sample_path = os.path.join(sample_dir, sample_file)
        X[i, :, :, 0] = load_and_preprocess_image(sample_path)
        
        mask_file = sample_file.split('.')[0] + '_mask.jpg'
        mask_path = os.path.join(mask_dir, mask_file)
        
        if os.path.exists(mask_path):
            Y[i] = 1
            Z[i, :, :, 0] = load_and_preprocess_image(mask_path)
        else:
            Y[i] = 0
            Z[i, :, :, 0] = np.zeros((img_height, img_width))
    
    # Save the preprocessed data as NumPy arrays
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'Y.npy'), Y)
    np.save(os.path.join(output_dir, 'Z.npy'), Z)

print('Preprocessing completed and data saved!')