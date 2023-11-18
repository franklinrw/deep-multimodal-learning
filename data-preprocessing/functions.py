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

def load_image(file_path, is_depth):
        """ Load an image as depth or color based on the is_depth flag. """
        if is_depth:
            return cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
        else:
            return cv2.imread(file_path)

def preprocess_image(path, sensor, j, width, height):

    depth = "depth" in sensor

    filename = f"init_{sensor if 'icub' not in sensor else 'color_' + sensor}_{j}.png"
    init = load_image(os.path.join(path, filename), depth)
    
    filename = f"effect_{sensor if 'icub' not in sensor else 'color_' + sensor}_{j}.png"
    effect = load_image(os.path.join(path, filename), depth)

    #print("shape", init.shape)

    # Pre-processing steps
    init, effect = [resize(img, width, height) for img in [init, effect]]

    #print("shape after resize", init.shape)

    if 'SKIP' in sensor:
        # Depth image processing
        # init, effect = [handle_invalid_depth_values(img) for img in [init, effect]]
        #print("depth image")
        init, effect = [normalize_depth(img, np.min(img), np.max(img)) for img in [init, effect]]
        #print("shape after normalized", init.shape)
    else:
        # Color image processing
        init, effect = [bgr_to_rgb(img) for img in [init, effect]]
        init, effect = [normalize(img) for img in [init, effect]]
    
    image = concatenate(init, effect)
    #print("shape after concatenate", image.shape)
    image = image.reshape(1, *image.shape)
    #print("shape after reshape", image.shape)
    return image
