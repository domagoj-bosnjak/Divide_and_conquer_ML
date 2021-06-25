# Global imports
import math
import numpy as np
import time

from s_dbw import S_Dbw
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import OPTICS

# Local imports
import input
import features

# TODO: MODULE PURPOSE
#   --Clustering(s)          [DONE]
#   --Clustering evaluation  [DONE]
#   --Reduction              [DONE]

dr_status = {}


def prepare_images_for_clustering(images):
    """
    TODO: [PROBABLY OBSOLETE]
    Convert images to numpy, and flatten them as:
            (num_of_images, dim1, dim2) ---->>> (num_of_images, dim1 * dim2)

    :param images       : list of images to be processed
    :return: images_np  : processed images
    """
    images_np = np.asarray(images)
    images_np = np.reshape(images_np, (images_np.shape[0], images_np.shape[1] * images_np.shape[2]))

    return images_np


def clustering(features_matrix, optics_min_samples=2):
    """
    :param features_matrix      : features matrix to be used for clustering

    #clustering parameters
    :param optics_min_samples:  : min_samples parameter for OPTICS algorithm

    :return: clustering_labels  : labels after clustering
    """
    clustering_optics = OPTICS(min_samples=optics_min_samples).fit(features_matrix)

    noise_points = np.count_nonzero(clustering_optics.labels_ == -1)

    # print("All together ", len(np.unique(clustering_optics.labels_)) + noise_points - 1, "clusters and noise points.")

    return clustering_optics.labels_


def best_clustering_by_score(features_matrix, optics_min_samples_list):
    """
    Find the best clustering per S_Dbw with respect to the given parameter list

    :param features_matrix          : feature matrix based upon which the clustering is performed
    :param optics_min_samples_list  : list of parameters to be tested (for the OPTICS clustering)

    :return:
    """
    best_score = math.inf
    best_parameter = 0
    best_labels = []

    for min_samples_parameter in optics_min_samples_list:
        clustering_labels = clustering(features_matrix, optics_min_samples=min_samples_parameter)
        score = S_Dbw(features_matrix, clustering_labels, metric='correlation')

        if score < best_score:
            best_score = score
            best_parameter = min_samples_parameter
            best_labels = clustering_labels

    return best_labels, best_parameter


def data_reduction(images, clustering_labels):
    """
    Function that reduces the amount of images in a *specific* class
        (all input images belong to the same class)

    :param images               : list of images in class
    :param clustering_labels    : labels as per clustering

    :return: indices_to_save    : indices of images that remaing after clustering
    """
    labels_set = set(clustering_labels)
    labels_set.remove(-1)

    indices_to_save = []

    for i in range(len(images)):
        cluster_label = clustering_labels[i]

        if cluster_label in labels_set and cluster_label != -1:  # if it is not a noise point and not yet chosen
            indices_to_save.append(i)

            labels_set.remove(cluster_label)

        if cluster_label == -1:     # save noise points
            indices_to_save.append(i)

    return indices_to_save


def data_reduction_main(num_of_classes=43, output_filename='./model/reduced_indices.csv', images=None, labels=None, augmentation_flag=False):
    dr_start = time.time()

    if not images and labels:
        print("No precomputed images for data reduction.(default state)")
        train_images_resized, train_labels = input.test_input(grayscale=True, image_range=num_of_classes, augmentation_flag=augmentation_flag)
    else:
        print("Using precomputed images for data reduction.")
        train_images_resized = images
        train_labels = labels

    feature_matrix = features.test_features(train_images_resized)

    current_start_index = 0  # First index of image in a class

    print("Images shape before reduction: ", np.asarray(train_images_resized).shape)
    print("Labels shape before reduction: ", np.asarray(train_labels).shape)
    print("Feature matrix dimensions:", np.asarray(feature_matrix).shape)

    dr_status['Feature_matrix_shape'] = np.asarray(feature_matrix).shape

    for i in range(num_of_classes):
        print("CLASS: ", str(i))
        # Extract single class
        images_class, features_class = input.extract_single_class(images=train_images_resized, labels=train_labels,
                                                                  class_number=str(i), extract_features=True,
                                                                  features=feature_matrix)
        features_class = np.asarray(features_class)

        # Clustering and data reduction
        best_labels, best_parameter = best_clustering_by_score(features_class,
                                                               optics_min_samples_list=[2])

        # data reduction
        indices_dr = data_reduction(images_class, best_labels)
        indices_to_save = np.asarray(indices_dr) + current_start_index

        if i == 0:
            data_reduction_indices = indices_to_save
        else:
            temp = list(data_reduction_indices)
            temp.extend(list(indices_to_save))
            data_reduction_indices = np.asarray(temp)

        class_length = len(images_class)
        current_start_index = current_start_index + class_length

    dr_end = time.time()
    print("Data reduction total time in minutes: ", (dr_end-dr_start)/60.0)
    dr_status['Total_time'] = (dr_end-dr_start)/60.0
    dr_status['Total number of images'] = len(train_images_resized)
    dr_status['Total number of images kept'] = len(data_reduction_indices)

    data_reduction_indices = np.asarray([int(x) for x in data_reduction_indices])
    np.savetxt(output_filename, data_reduction_indices, delimiter=',')


def data_reduction_alternate(num_of_classes=43, output_filename='./model/reduced_indices_alternate.csv', augmentation_flag=False):

    train_images_resized, train_labels, test_images, test_labels = input.test_input_alternate(grayscale=False,
                                                                                              image_range=num_of_classes,
                                                                                              augmentation_flag=augmentation_flag)
    data_reduction_main(num_of_classes=num_of_classes, output_filename=output_filename,
                        images=input.images_grayscale(train_images_resized), labels=train_labels)

    return train_images_resized, train_labels, test_images, test_labels

#
# if __name__ == "__main__":
#     data_reduction_main(43)
