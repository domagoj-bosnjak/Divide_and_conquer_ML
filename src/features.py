import numpy as np
from skimage import transform
from skimage.feature import haar_like_feature

# Local imports:
import input


# TODO: MODULE PURPOSE
#   --Features
#   --Feature selection


def integral_image_multiple(images):
    """
    Calculate integral images for the given list of images
    """

    integral_images = [transform.integral.integral_image(image) for image in images]

    return integral_images


def haar_compute(integral_images, top_left_x, top_left_y, width_, height_, feature_type_):
    """
    Compute Haar-like features with given parameters

    :param integral_images: list of integral images for feature computation
    :param top_left_x: top left window corner(row)
    :param top_left_y: top left window corner(col)
    :param width_: window width
    :param height_: window height
    :param feature_type_: type of Haar-like feature(check skimage docs)

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


if __name__ == '__main__':

    trainImages, trainLabels = input.read_traffic_signs('', image_range=5)
    trainImagesResized = input.resize_images(trainImages)

    trainIntegralImages = integral_image_multiple(trainImagesResized)

    hfv = haar_compute(trainIntegralImages, 0, 0, 4, 4, 'type-2-x')
    print(np.asarray(hfv).shape)

    img_ii = trainIntegralImages[0]

