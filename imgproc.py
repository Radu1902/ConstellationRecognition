import cv2 as cv
import numpy as np
import sys

def meanFilter(img, ksize = 3):
    kernel = np.ones((ksize, ksize), np.float32) / (ksize ** 2)
    img = cv.filter2D(img, ddepth=-1,kernel=kernel)
    return img

def gaussianBlur(img, kernel_size=3, sigma=1.0):
    blurred = cv.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    return blurred

def threshold(img, thresh):
    _, threshed = cv.threshold(img, thresh, 255, cv.THRESH_BINARY)
    return threshed

def otsu(img):
    _, threshed = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return threshed

def resize(image, max_dimension=700):    
    original_height, original_width = image.shape
    
    if original_height <= max_dimension and original_width <= max_dimension:
        print("No resizing needed.")
        return image
    
    if original_width > original_height:
        scaling_factor = max_dimension / float(original_width)
    else:
        scaling_factor = max_dimension / float(original_height)
    
    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)
    
    resized_image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
        
    print(f"Image resized to {new_width}, {new_height}")
    return resized_image

def dilate(threshed):
    kernel = np.ones((3, 3), np.uint8) 
    dilated = cv.dilate(threshed, kernel, iterations=1)
    return dilated