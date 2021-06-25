# Global imports
import numpy as np

from skimage import transform
from skimage.feature import haar_like_feature

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif

# TODO: MODULE PURPOSE
#   --Features          [DONE]
#   --Feature selection [DONE] (implemented in data_reduction script)


def integral_image_multiple(images):
    """
    Calculate integral images for the given list of images
    """

    integral_images = [transform.integral.integral_image(image) for image in images]

    return integral_images


def haar_compute(integral_images, top_left_x, top_left_y, width_, height_, feature_type_):
    """
    Compute Haar-like features with given parameters

    :param integral_images  : list of integral images for feature computation
    :param top_left_x       : top left window corner(row)
    :param top_left_y       : top left window corner(col)
    :param width_           : window width
    :param height_          : window height
    :param feature_type_    : type of Haar-like feature(check scikit-image docs)

    :return: haar_feature_vector: a vector of haar like features for each image
                                    shape: (no_of_images, no_of_features)
                                    -> one row corresponds to one image
    """

    haar_feature_vector = [haar_like_feature(int_image=integral_image,
                                             r=top_left_x,
                                             c=top_left_y,
                                             width=width_,
                                             height=height_,
                                             feature_type=feature_type_)
                           for integral_image in integral_images]

    return haar_feature_vector


def haar_feature_pipeline(images):
    """
    A few example feature computation choices

    :param images   : a list of images

    :return:feature_vector : A complete feature vector
    """
    integral_images = integral_image_multiple(images)

    print("Computing type-2 features.")
    # TYPE 2-x
    haar = haar_compute(integral_images=integral_images,
                        top_left_x=5,
                        top_left_y=15,
                        width_=6,
                        height_=6,
                        feature_type_='type-2-x')
    feature_matrix = np.array(haar)
    haar = haar_compute(integral_images=integral_images,
                        top_left_x=5,
                        top_left_y=5,
                        width_=6,
                        height_=6,
                        feature_type_='type-2-x')
    feature_matrix = np.concatenate((feature_matrix, np.asarray(haar)), axis=1)

    haar = haar_compute(integral_images=integral_images,
                        top_left_x=15,
                        top_left_y=5,
                        width_=6,
                        height_=6,
                        feature_type_='type-2-x')
    feature_matrix = np.concatenate((feature_matrix, np.asarray(haar)), axis=1)

    haar = haar_compute(integral_images=integral_images,
                        top_left_x=15,
                        top_left_y=15,
                        width_=6,
                        height_=6,
                        feature_type_='type-2-x')
    feature_matrix = np.concatenate((feature_matrix, np.asarray(haar)), axis=1)

    #TYPE 2-y
    haar = haar_compute(integral_images=integral_images,
                        top_left_x=5,
                        top_left_y=15,
                        width_=6,
                        height_=6,
                        feature_type_='type-2-y')
    feature_matrix = np.concatenate((feature_matrix, np.asarray(haar)), axis=1)

    haar = haar_compute(integral_images=integral_images,
                        top_left_x=5,
                        top_left_y=5,
                        width_=6,
                        height_=6,
                        feature_type_='type-2-y')
    feature_matrix = np.concatenate((feature_matrix, np.asarray(haar)), axis=1)

    haar = haar_compute(integral_images=integral_images,
                        top_left_x=15,
                        top_left_y=5,
                        width_=6,
                        height_=6,
                        feature_type_='type-2-y')
    feature_matrix = np.concatenate((feature_matrix, np.asarray(haar)), axis=1)

    haar = haar_compute(integral_images=integral_images,
                        top_left_x=15,
                        top_left_y=15,
                        width_=6,
                        height_=6,
                        feature_type_='type-2-y')
    feature_matrix = np.concatenate((feature_matrix, np.asarray(haar)), axis=1)
    
    # TYPE 3
    print("Computing type-3 features.")
    haar = haar_compute(integral_images=integral_images,
                        top_left_x=5,
                        top_left_y=15,
                        width_=6,
                        height_=6,
                        feature_type_='type-3-x')
    feature_matrix = np.concatenate((feature_matrix, np.asarray(haar)), axis=1)

    haar = haar_compute(integral_images=integral_images,
                        top_left_x=5,
                        top_left_y=5,
                        width_=6,
                        height_=6,
                        feature_type_='type-3-y')
    feature_matrix = np.concatenate((feature_matrix, np.asarray(haar)), axis=1)

    haar = haar_compute(integral_images=integral_images,
                        top_left_x=15,
                        top_left_y=5,
                        width_=6,
                        height_=6,
                        feature_type_='type-3-x')
    feature_matrix = np.concatenate((feature_matrix, np.asarray(haar)), axis=1)

    haar = haar_compute(integral_images=integral_images,
                        top_left_x=15,
                        top_left_y=15,
                        width_=6,
                        height_=6,
                        feature_type_='type-3-y')
    feature_matrix = np.concatenate((feature_matrix, np.asarray(haar)), axis=1)

    print("Computing type-4 features.")
    # TYPE 4 (chessboard)
    haar = haar_compute(integral_images=integral_images,
                        top_left_x=10,
                        top_left_y=10,
                        width_=10,
                        height_=10,
                        feature_type_='type-4')
    feature_matrix = np.concatenate((feature_matrix, np.asarray(haar)), axis=1)

    return feature_matrix


def test_features(images, feature_selection=False, labels=None, feature_selection_percent=50):
    """
    Function to be used for testing purposes only!!

    :return: feature_matrix
    """
    feature_matrix = haar_feature_pipeline(images)

    print("Performing feature selection, reduction by", feature_selection_percent, "percent.")
    if feature_selection:
        feature_matrix = SelectPercentile(f_classif, percentile=feature_selection_percent).fit_transform(feature_matrix,
                                                                                                         labels)

    return feature_matrix


