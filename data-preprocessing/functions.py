import numpy as np
import cv2
from PIL import Image
import os

### FUNCTIONS ###
def concatenate(image_1, image_2):
    # Concatenate the image
    return np.concatenate((image_1, image_2), axis=1)

def normalize(image):
    # Normalize the image
    return image/255

def resize(image, width, height):
    # Resize an image to a fixed size
    return cv2.resize(image, (width, height))

def bgr_to_rgb(image):
    # Changes the images from BGR to RGB
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def image_from_array(image):
    # Changes the array to image
    return Image.fromarray(image)

def flatten(image, width, height):
    # Flatten the images
    return image.flatten().reshape(1, width*height*3)

def to_float(image):
    # Convert to float32
    return image.astype(np.float32)

def normalize_depth(image, min_depth, max_depth):
    # Clip to ensure values fall within the specified range
    image = np.clip(image, min_depth, max_depth)
    # Normalize to the range [0, 1]
    return (image - min_depth) / (max_depth - min_depth)

# def handle_invalid_depth_values(image, invalid_value=0, replacement_method='constant', replacement_value=0):
#     if replacement_method == 'constant':
#         image[image == invalid_value] = replacement_value
#     elif replacement_method == 'interpolate':
#         # Example of a simple interpolation strategy
#         mask = (image == invalid_value)
#         image[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), image[~mask])
#     return image

# def preprocess_image(path, sensor, j, width, height):
#     filename = f"init_{sensor if 'icub' not in sensor else 'color_' + sensor}_{j}.png"
#     init = cv2.imread(os.path.join(path, filename))
    
#     filename = f"effect_{sensor if 'icub' not in sensor else 'color_' + sensor}_{j}.png"
#     effect = cv2.imread(os.path.join(path, filename))

#     # Pre-processing steps
#     init, effect = [resize(img, width, height) for img in [init, effect]]
#     init, effect = [bgr_to_rgb(img) for img in [init, effect]]
#     init, effect = [normalize(img) for img in [init, effect]]
    
#     image = concatenate(init, effect)
#     return image.reshape(1, *image.shape)

def preprocess_image(path, sensor, j, width, height):
    filename = f"init_{sensor if 'icub' not in sensor else 'color_' + sensor}_{j}.png"
    init = cv2.imread(os.path.join(path, filename))
    
    filename = f"effect_{sensor if 'icub' not in sensor else 'color_' + sensor}_{j}.png"
    effect = cv2.imread(os.path.join(path, filename))

    # Pre-processing steps
    init, effect = [resize(img, width, height) for img in [init, effect]]

    if 'depth' in sensor:
        # Depth image processing
        # init, effect = [handle_invalid_depth_values(img) for img in [init, effect]]
        init, effect = [normalize_depth(img, np.min(img), np.max(img)) for img in [init, effect]]
    else:
        # Color image processing
        init, effect = [bgr_to_rgb(img) for img in [init, effect]]
        init, effect = [normalize(img) for img in [init, effect]]
    
    image = concatenate(init, effect)
    return image.reshape(1, *image.shape)
