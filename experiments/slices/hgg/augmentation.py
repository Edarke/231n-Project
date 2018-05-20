import numpy as np
import random as rng
import scipy.ndimage.interpolation as interpolate
import cv2
import scipy.misc
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from pylab import imshow, show, get_cmap
from numpy import random


def next_bool(p):
    return rng.random() < p


def elastic_transform(image, label, alpha=720, sigma=24):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    # Params taken from https://arxiv.org/pdf/1705.03820.pdf
    shape = image.shape

    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    result = np.empty_like(image)
    for channel in range(image.shape[-1]):
        result[..., channel] = map_coordinates(image[..., channel], indices, order=1).reshape(shape)
    return result, map_coordinates(label, indices, order=1).reshape(shape)


# Inspired by https://arxiv.org/pdf/1705.03820.pdf
def train_augmentation(sample, label):
    '''
    :param sample: shape (h, w, 4) or (h, w, d, 4)
    :return: sample of the same shape
    '''
    haxis = 0
    waxis = 1
    axial_plane = [haxis, waxis]

    # brighness_factor = rng.uniform(.9, 1.1)
    # sample *= brighness_factor
    # sample *= np.random.rand(*sample.shape[0:-1], 1) * .1

    # flipping
    if next_bool(.5):
        sample = np.flip(sample, haxis)
        label = np.flip(label, haxis)
    if next_bool(.5):
        sample = np.flip(sample, waxis)
        label = np.flip(label, waxis)

    # 180 degree rotation
    if next_bool(.5):
        sample = np.rot90(sample, 2, axes=axial_plane)
        label = np.rot90(label, 2, axes=axial_plane)

    # zoom_factor = rng.uniform(.9, 1.1)
    #sample = interpolate.zoom(sample, zoom=zoom_factor)
    #label = interpolate.zoom(label, zoom=zoom_factor, )
    #
    # rotation_degree = rng.uniform(-10, 10)
    # sample = scipy.misc.imrotate(sample, rotation_degree, interpolate='bilinear')
    # label = scipy.misc.imrotate(label, rotation_degree, interpolate='nearest')
    # sample = interpolate.rotate(sample, angle=rotation_degree, axes=axial_plane)
    # label = interpolate.rotate(label, angle=rotation_degree, axes=axial_plane)
    sample, label = elastic_transform(sample, label)
    return sample, label


def test_augmentation(sample, label):
    return sample, label



if __name__ == '__main__':
    pass
