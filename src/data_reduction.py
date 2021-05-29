import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Local imports
import input
import features

# TODO: MODULE PURPOSE
#   --Clustering(s)          [    ]
#   --Reduction              [    ]


def clustering(images, labels, feature_matrix):
    """
    :param images:
    :param labels:
    :param feature_matrix:
    :return:
    """

    # IMPLEMENT
    # -> CLASS BY CLASS CLUSTERING
    # K-MEANS OR SOMETHING HIERARCHICAL

if __name__ == "__main__":

    train_images_resized, train_labels = input.test_input()

    feature_matrix = features.test_features()

    # clustering(train_images_resized, train_labels, feature_matrix)