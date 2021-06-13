# Global imports
import math

import numpy as np
from s_dbw import S_Dbw

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN


# Local imports
import input
import features

# TODO: MODULE PURPOSE
#   --Clustering(s)          [    ]
#   --Clustering evaluation  [    ]
#   --Reduction              [    ]


def prepare_images_for_clustering(images):
    """
    TODO: [OBSOLETE]
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
    # clustering_optics = DBSCAN(eps=3, min_samples=2).fit(features_matrix)

    noise_points = np.count_nonzero(clustering_optics.labels_ == -1)

    print("There are: ", len(features_matrix), "images.")
    print("There are: ", len(np.unique(clustering_optics.labels_)), "clusters.")
    print("There are: ", noise_points, " noise points.")
    print("All together that is ", len(np.unique(clustering_optics.labels_)) + noise_points, "clusters and noise points.")
    print("\n")

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

    for min_samples_parameter in optics_min_samples_list:
        clustering_labels = clustering(features_matrix, optics_min_samples=min_samples_parameter)
        score = S_Dbw(features_matrix, clustering_labels, metric='correlation')

        print("The score for parameter", min_samples_parameter, "is", score)
        if score < best_score:
            best_score = score
            best_parameter = min_samples_parameter

    print("The ultimate score is truly: ", best_score, "when we set min_samples=", best_parameter)
    return best_parameter


if __name__ == "__main__":

    train_images_resized, train_labels = input.test_input()

    feature_matrix = features.test_features()

    images_class_two, features_class_two = input.extract_single_class(images=train_images_resized, labels=train_labels,
                                                                      class_number='2', extract_features=True,
                                                                      features=feature_matrix)

    features_class_two = np.asarray(features_class_two)

    best_clustering_by_score(features_class_two, optics_min_samples_list=[2, 3, 4, 5, 6, 7, 8, 9, 10])
