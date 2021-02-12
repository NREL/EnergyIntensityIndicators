import pandas as pd
import numpy as np

# @staticmethod
def logarithmic_average(x, y):
    """The logarithmic average of two positive numbers x and y
    """ 
    try:
        x = float(x)
        y = float(y)
    except TypeError:
        L = np.nan
        return L       
    if x > 0 and y > 0:
        if x != y:
            difference = x - y
            log_difference = np.log(x) - np.log(y)
            L = difference / log_difference
        else:
            L = x
    else: 
        L = np.nan

    return L

