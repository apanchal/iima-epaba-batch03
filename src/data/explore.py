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
import numpy as np


from util import Util


util1 = Util()
signs = util1.class_lables

def list_images(dataset, dataset_y, ylabel="", cmap=None):
    """
    Display a list of images in a single figure with matplotlib.
    Parameters:
    images: An np.array compatible with plt.imshow.
    label (Default = No label): A string to be used as a label for each image.
    cmap (Default = None): Used to display gray images.
    """
    #print('Listing called')
    plt.figure(figsize=(15, 16))
    for i in range(6):
        plt.subplot(1, 6, i+1)
        indx = random.randint(0, len(dataset))
        #Use gray scale color map if there is only one channel
        cmap = 'gray' if len(dataset[indx].shape) == 2 else cmap
        plt.imshow(dataset[indx], cmap = cmap)
        plt.xlabel(signs[dataset_y[indx]])
        plt.ylabel(ylabel)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()


def histogram_plot(dataset, label,n_classes):
    """
    Plots a histogram of the input data.
        Parameters:
            dataset: Input data to be plotted as a histogram.
            lanel: A string to be used as a label for the histogram.
    """
    hist, bins = np.histogram(dataset, bins=n_classes)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.xlabel(label)
    plt.ylabel("Image count")
    plt.show()
