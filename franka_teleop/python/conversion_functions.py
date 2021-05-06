import numpy as np

def identity(arr):
    return arr

def grippers_open(arr):
    gripper_thresh = 0.038
    ## right_open, left_open    
    return np.array([np.where(arr[8] > gripper_thresh, 1, 0),
                     np.where(arr[16] > gripper_thresh, 1, 0)])
