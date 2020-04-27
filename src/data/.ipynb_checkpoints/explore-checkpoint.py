"""
Project: Object Recognising Traffic Signs Using Deep Learning
Mentor: Prof. Ankur Sinha

This file contains all functions related to data exploration.

@author: Ashish Panchal(aashish.panchal@gmail.com)
"""

import matplotlib.pyplot as plt
import random
import os
import sys
from util import Util



def list_images(dataset, dataset_y, ylabel="", cmap=None):
        """
        Display a list of images in a single figure with matplotlib.
        Parameters:
        images: An np.array compatible with plt.imshow.
        label (Default = No label): A string to be used as a label for each image.
        cmap (Default = None): Used to display gray images.
        """
        print('Listing called')