import cv2
import numpy as np
import random

# Augmentation
def flip_image(image, steering):
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        steering = -steering
        
    return image, steering


def adjust_brightness(image):
    if random.random() < 0.5:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        brightness = random.uniform(0.3, 1.2)

        hsv[:, :, 2] = np.clip(
            hsv[:, :, 2] * brightness,
            0,
            255
        )

        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return image

def random_translate(image, steering, max_shift=40, angle_per_pixel=0.004):
    height, width = image.shape[:2]

    x_shift = random.randint(-max_shift, max_shift)
    y_shift = random.randint(-10, 10)  # small vertical jitter for robustness

    # Adjust steering proportionally to how far we shifted horizontally.
    steering = steering + (x_shift * angle_per_pixel)

    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    image = cv2.warpAffine(image, M, (width, height))

    return image, steering

def random_zoom(image):
    if random.random() < 0.3:

        height, width = image.shape[:2]

        crop = random.uniform(0.8, 1.0)

        new_h = int(height * crop)
        new_w = int(width * crop)

        y = random.randint(0, height-new_h)
        x = random.randint(0, width-new_w)

        image = image[y:y+new_h, x:x+new_w]

        image = cv2.resize(image, (width, height))

    return image


def augment(image, steering):
    image, steering = flip_image(image, steering)
    image, steering = random_translate(image, steering)
    image = adjust_brightness(image)
    image = random_zoom(image)

    return image, steering

# Model preprocessing
def preprocess_image(image):

    # Crop road area
    image = image[60:135, :, :]

    # Convert RGB -> YUV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    # Nvidia input size
    image = cv2.resize(image, (200,66))

    # Blur
    image = cv2.GaussianBlur(image,(3,3),0)

    # Normalize
    image = image / 255.0

    return image