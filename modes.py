from enum import Enum

class Identification_mode(Enum):
    SIMPLE = "Simple blob detector"
    OTSU = "Otsu thresholding"
    THRESHOLD = "Thresholding"

def get_identification_modes():
    return [mode.value for mode in Identification_mode]

class Filter_mode(Enum):
    NONE = "None"
    MEAN = "Mean filter"
    GAUSS = "Gaussian filter"

def get_filter_modes():
    return [mode.value for mode in Filter_mode]