import csv
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from imgaug import augmenters as iaa

from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
from PIL import Image
import random

# TODO: MODULE PURPOSE
#   --Loading images and labels         [DONE]
#   --Equalizing dimensions of images   [DONE]

def augment_images_function(images, p):
    """
    Augmentations with probability p
    """
    augs = iaa.SomeOf((2,4),
                      [
                          iaa.Crop(px=(0,4)), # crop images from each size 0-4px (randomly chosen)
                          iaa.Affine(scale={"x": (0.8,1.2), "y":(0.8, 1.2)}),
                          iaa.Affine(rotate=(-45,45)), # rotate by -45 to +45 degrees
                          iaa.Affine(shear=(-10,10)) # shear by -10 to +10 degrees
                      ])

    sequential = iaa.Sequential([iaa.Sometimes(p, augs)])
    result = sequential.augment_images(images)
    return result

def augmentation(images, labels, min_images_in_class=400):
    class_size = [0] * 43
    class_indexes = [[] for i in range(43)]

    for i in range(len(labels)):
        current_class = int(labels[i])
        class_size[current_class] += 1
        class_indexes[current_class].append(i)
    #
    # for i in range(43):
    #     print(class_size[i], len(class_indexes[i]))

    for i in range(43):
        if class_size[i] < min_images_in_class:
            print("Class", i, "is too small! Size is", class_size[i])
            num_missing = min_images_in_class - class_size[i]
            images_for_augmentation = []
            labels_for_augmentation = []

            for j in range(num_missing):
                image_index = random.choice(class_indexes[i])
                images_for_augmentation.append(images[image_index])
                labels_for_augmentation.append(labels[image_index])

            augmented_class = augment_images_function(images_for_augmentation, 1)
            augmented_class = np.array(augmented_class)
            augmented_labels = np.array(labels_for_augmentation)

            images = np.concatenate((images, augmented_class), axis=0)
            labels = np.concatenate((labels, augmented_labels), axis=0)

    return images, labels


def read_traffic_signs(root_path, image_range=43, grayscale=True, augmentation_flag=False):
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
        prefix = root_path + 'Images' + '/' + format(c, '05d') + '/'  # path name

        gt_file = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # read CSV
        gt_reader = csv.reader(gt_file, delimiter=';')

        next(gt_reader)

        for row in gt_reader:
            images.append(plt.imread(prefix + row[0]))
            labels.append(row[7])

        gt_file.close()

    if grayscale:
        images = images_grayscale(images)

    if augmentation_flag:
        print("PRIJE AUGMENTACIJE")
        print(len(images))
        print(len(labels))
        images, labels = augmentation(images, labels, min_images_in_class=400)
        print("NAKON AUGMENTACIJE")
        print(images.size)
        print(labels.size)

    return images, labels


def read_test_data(root_path='', grayscale=True):
    images = []
    labels = []

    prefix = root_path + 'Images_test/' # path name

    gt_file = open(prefix + 'GT-final_test.csv')  # read CSV
    gt_reader = csv.reader(gt_file, delimiter=';')

    next(gt_reader)

    for row in gt_reader:
        images.append(plt.imread(prefix + row[0]))
        labels.append(row[7])

    gt_file.close()

    images_resized = resize_images(images)

    if grayscale:
        images_resized = images_grayscale(images_resized)

    return images_resized, labels


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

    if extract_features:
        return images_extracted, features_extracted
    else:
        return images_extracted


def test_input(grayscale=True, image_range=5):
    """
    Function to be used for testing purposes only!!

    :return: train_images_resized : A list of images (grayscale and resized)
    :return: train_labels : A list of labels
    """
    train_images, train_labels = read_traffic_signs('', image_range=image_range, grayscale=grayscale, augmentation_flag=True)
    train_images_resized = resize_images(train_images)

    return train_images_resized, train_labels


def test_input_alternate(grayscale=False, image_range=43):

    train_images = []
    test_images = []

    train_labels = []
    test_labels = []

    images, labels = test_input(grayscale=grayscale, image_range=image_range)

    current_start_index = 0

    for i in range(43):
        images_class = extract_single_class(images, labels, str(i))
        len_class = len(images_class)

        labels_class = [str(i)] * len_class

        X_train, X_test, y_train, y_test = train_test_split(images_class, labels_class, test_size=0.001, random_state=42)

        train_images.extend(X_train)
        train_labels.extend(y_train)

        test_images.extend(X_test)
        test_labels.extend(y_test)

        class_length = len_class
        current_start_index = current_start_index + class_length

    return train_images, train_labels, test_images, test_labels


# if __name__ == "__main__":
#     read_test_data()