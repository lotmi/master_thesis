"""
Functions to apply CT scan windowing and normalization.

"""

def apply_window(Input,W,L):
    # Bone window settings: W = 1800 and L = 400 
    min_HU=L-(0.5*W) # lower grey level 
    max_HU=L+(0.5*W) # upper grey level
    Input[Input<min_HU]=min_HU
    Input[Input>max_HU]=max_HU
    return Input

def apply_normalization(Input):
    minimum = Input.min()
    maximum = Input.max()
    Input = (Input - minimum) / (maximum - minimum)
    return Input
