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
import statistics


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


def plot_dataset_distributions(datasets, set_descs):
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 4))
    ret_data_bins = []    
    for dataset, set_desc in zip(datasets, set_descs):
        # Create a histogram of the classes
        data_bins = np.bincount(dataset)
        # Convert to percent
        data_bins = data_bins / len(dataset) * 100
        ret_data_bins.append(data_bins)

        print("Dataset {} contains {} samples".format(set_desc, len(dataset)))
        print("median={:.2f}%  mean={:.2f}%".format(statistics.median(data_bins), statistics.mean(data_bins)))

        ax0.plot(range(len(data_bins)), data_bins, label=set_desc)
        ax0.set_title('% of samples per class')
        ax0.set_xlabel('class')
        ax0.set_ylabel('% samples')

        ax1.hist(data_bins, bins=20, label=set_desc)
        ax1.set_title('sample distribution')
        ax1.set_xlabel('% samples')
        ax1.set_ylabel('# classes')
    ax0.legend(loc=1)
    ax1.legend(loc=1)
    plt.savefig('../plots/data_explore/data_distribution.png')
    plt.show()
    return ret_data_bins

def get_image_per_class(X, y,n_classes):
    """ 
    Plot a representatative of each image class in a 5x10 image grid

    The training dataset is traversed until a sample of each class
    is encountered and cached.

    Another loop then travereses all of the cached images and displays them.
    The two loops are required because we want to display the image samples
    in class order, not in the order they are encountered.
    """
    signs_left = n_classes
    class_images = ['None' for x in range(signs_left)]
    i = 0
    while signs_left>0:
        if class_images[y[i]] == 'None':
            image = X[i].squeeze()
            class_images[y[i]] = image
            signs_left -= 1
        i += 1
    return class_images

def plot_image(image, nr, nc, i, label=""):
    """
    Plot a single image.
    If 'i' is greater than 0, then plot this image as 
    a subplot of a larger plot.
    """
    
    if i>0: 
        plt.subplot(nr, nc, i)
    else:
        plt.figure(figsize=(nr,nc))
        
    plt.xticks(())
    plt.yticks(())
    plt.xlabel(label)
    plt.tight_layout()
    plt.imshow(image, cmap="gray")
    
def summarize_stats(class_images, y_train, y_valid):
    """
    'class_images' is a list of images, one per class.
    This function plots this images list, and print underneath each one its class, 
    the number of training samples, the percent of training samples, 
    and the percent of validation samples
    """
    # Create a histogram of the classes
    y_train_hist = np.bincount(y_train)
    y_valid_hist = np.bincount(y_valid)

    nr = 5; nc = 9
    plt.figure(figsize=(nr,nc))
    for image,i in zip(class_images, range(len(class_images))):
        label = (str(i) + "\n"                                            # class
              + str(y_train_hist[i]) + "\n"                               # no. of training samples
              + "{:.1f}%".format(100 * y_train_hist[i]/sum(y_train_hist))  + "\n"   # representation in training samples
              + "{:.1f}%".format(100 * y_valid_hist[i]/sum(y_valid_hist)))     # representation in validation samples
        plot_image(image, nr, nc, i+1, label)
    plt.savefig('../plots/data_explore/Figure-2.2_data_summary.png')        
