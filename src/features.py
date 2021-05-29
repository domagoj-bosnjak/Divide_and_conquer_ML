import numpy as np
from skimage import transform
from skimage.feature import haar_like_feature

# Local imports:
import input


# TODO: MODULE PURPOSE
#   --Features          [DONE]
#   --Feature selection [    ]


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

    haar = haar_compute(integral_images=integral_images,
                        top_left_x=5,
                        top_left_y=15,
                        width_=6,
                        height_=5,
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
                        top_left_x=5,
                        top_left_y=15,
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

    return feature_matrix


def test_features():
    """
    Function to be used for testing purposes only!!

    :return: feature_matrix
    """
    train_images_resized, _ = input.test_input()

    feature_matrix = haar_feature_pipeline(train_images_resized)

    return feature_matrix
