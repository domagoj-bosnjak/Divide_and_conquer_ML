import csv
import cv2
import math
import matplotlib.pyplot as plt
# import numpy as np
import numpy as np

from skimage.color import rgb2gray


# TODO: MODULE PURPOSE
#   --Loading images and labels         [DONE]
#   --Equalizing dimensions of images   [DONE]


def read_traffic_signs(root_path, image_range=43, grayscale=True):
    """
    Reading images and labels from files

    :param root_path    : path to the folder Images
    :param image_range  : how many classes are being used (less than 43 for testing)
    :param grayscale    : should the images be converted from RGB to grayscale

    :return: images: a numpy array of images
    :return: labels: a numpy array of labels of corresponding images,
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

    # Can this be avoided?!
    if grayscale:
        images = images_grayscale(images)

    return images, labels


def images_grayscale(images):
    """
    Convert a list of images from RGB to grayscale
    """

    images_gray = [rgb2gray(image) for image in images]

    return images_gray


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

    for i in range(len(images)):
        images_resized.append(
            cv2.resize(images[i], dsize=(dimension_m, dimension_n), interpolation=cv2.INTER_CUBIC))

    return images_resized


def extract_single_class(images, labels, class_number, features=None, extract_features=False):
    """
    Extract images that belong to one specific class
    (OPTIONAL) extract features alongisde the images

    :param images           : list of images
    :param labels           : list of labels
    :param class_number     : which class number to extract

    :param features         : feature matrix, if the features should be extracted as well
    :param extract_features : should features be extracted as well
    """
    if not 0 <= int(class_number) <= 42:
        raise ValueError("Wrong class number!")

    images_extracted = []
    features_extracted = []

    for i in range(len(images)):
        if labels[i] == class_number:
            images_extracted.append(images[i])
            if extract_features:
                features_extracted.append(features[i])

    # print(np.asarray(images).shape)
    # print(np.asarray(images_extracted).shape)
    # print(np.asarray(features).shape)
    # print(np.asarray(features_extracted).shape)

    if extract_features:
        return images_extracted, features_extracted
    else:
        return images_extracted


def test_input():
    """
    Function to be used for testing purposes only!!

    :return: train_images_resized : A list of images (grayscale and resized)
    :return: train_labels : A list of labels
    """
    train_images, train_labels = read_traffic_signs('', image_range=5)
    train_images_resized = resize_images(train_images)

    return train_images_resized, train_labels
