import csv
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from numpy.random import default_rng
from sklearn.decomposition import PCA

# TODO: MODULE PURPOSE
#   --Loading images and labels
#   --Equalizing dimensions of images


def read_traffic_signs(root_path, image_range=43):
    """
    Reading images and labels from files

    Args:
        root_path: path to the folder Images
        image_range: how many classes are being used (less than 43 for testing)

    Returns:
        images: a numpy array of images
        labels: a numpy array of labels of corresponding images,
                    values in interval [0,42]
    """

    images = []
    labels = []
    for c in range(0, image_range):
        # OLD VERSION: prefix = root_path + '/' + 'Images' + '/' + format(c, '05d') + '/'
        prefix = root_path + 'Images' + '/' + format(c, '05d') + '/'  # path name

        gt_file = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # read CSV
        gt_reader = csv.reader(gt_file, delimiter=';')

        next(gt_reader)

        for row in gt_reader:
            images.append(plt.imread(prefix + row[0]))
            labels.append(row[7])

        gt_file.close()
    return images, labels


def determine_minimum_dimensions(images):
    """
    Determine minimum image dimension out of all images given as input
    """

    min_m = math.inf
    min_n = math.inf
    for i in range(len(images)):
        image_shape = images[i].shape
        m = image_shape[0]
        n = image_shape[1]

        if m < min_m:
            min_m = m
        if n < min_n:
            min_n = n

    print("Minimum dimensions: ", min_m, " i ", min_n)

    return min_m, min_n


def resize_images(images, dimension_m=30, dimension_n=30):
    """
    Resize all images to dimensions specified as parameter
    """
    images_resized = []

    for i in range(len(trainImages)):
        images_resized.append(
            cv2.resize(images[i], dsize=(dimension_m, dimension_n), interpolation=cv2.INTER_CUBIC))

    return images_resized


if __name__ == '__main__':

    trainImages, trainLabels = read_traffic_signs('', image_range=5)

    for k in range(len(trainLabels)):
        if k%100 == 0:
            print("Label: ", trainLabels[k])
            # plt.imshow(trainImages[k])
            # plt.show()
