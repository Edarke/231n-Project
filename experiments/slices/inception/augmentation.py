import numpy as np
import random
import scipy.misc
from scipy.ndimage.interpolation import map_coordinates, rotate, zoom
from scipy.ndimage.filters import gaussian_filter
from numpy import random
from read_data import BRATSReader


def next_bool(p):
    return random.random() < p


num_filters = 16
alpha = 500
sigma = 20
dxs = np.random.uniform(-1, 1, (num_filters, 224, 224, 4))
dys = np.random.uniform(-1, 1, (num_filters, 224, 224, 4))

for i in range(num_filters):
    dxs[i] = gaussian_filter(dxs[i], sigma, mode="constant", cval=0) * alpha
    dys[i] = gaussian_filter(dys[i], sigma, mode="constant", cval=0) * alpha

x, y, z = np.meshgrid(np.arange(224), np.arange(224), np.arange(4))


def elastic_transform(image, label):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    # Params taken from https://arxiv.org/pdf/1705.03820.pdf
    dx = dxs[np.random.randint(0, len(dxs))]
    dy = dys[np.random.randint(0, len(dys))]

    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    distored_label = map_coordinates(np.expand_dims(label, -1), indices, order=1, mode='reflect')

    img, lab = distored_image.reshape(image.shape), distored_label.reshape(image.shape)[:, :, 0]
    return img, lab


def crop_center(img, h, w):
    '''
    Crop center of ndarray so result as size h, w, c.

    :param img:
    :param h:
    :param w:
    :return:
    '''
    hh, ww = img.shape[:2]
    starth = (hh - h) // 2
    startw = (ww - w) // 2
    img = img[starth:starth + h, startw:startw + w, ...]
    return img

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
    sample = sample.astype(np.float32)
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

    # zoom_factor = random.uniform(.9, 1.1)
    # sample = zoom(sample, zoom=zoom_factor, order=0)
    # label = zoom(label, zoom=zoom_factor, order=0)
    #

    if False:  # This is like super slow
        rotation_degree = random.uniform(-10, 10)
        sample = rotate(sample, angle=rotation_degree, axes=axial_plane, mode='nearest')
        label = rotate(label, angle=rotation_degree, axes=axial_plane, mode='nearest')

        sample = crop_center(sample, 224, 224)
        label = crop_center(label, 224, 224)
    sample, label = elastic_transform(sample, label)

    return sample, label


def test_augmentation(sample, label):
    return sample, label


def preprocess(data, labels):
    """
    :param data: (batch, height, width, depth)
    :param labels:  (batch, height, width, depth)
    :return: tuple of processed tuple list data, and labels
    """

    N, H, W, D = data.shape

    # Crop or pad to 224x224x224
    h_diff = 224 - H
    w_diff = 224 - W

    pad_dims = [(0, 0)]
    if h_diff > 0:
        pad_top = h_diff // 2
        pad_bottom = h_diff // 2 + h_diff % 2
        pad_dims.append((pad_top, pad_bottom))
    elif h_diff < 0:
        slice_top = -1 * h_diff // 2
        slice_bottom = -1 * (h_diff // 2 - h_diff % 2)
        data = data[:, slice_top:-slice_bottom, :, :]
        labels = labels[:, slice_top:-slice_bottom, :, :]
        pad_dims.append((0, 0))

    if w_diff > 0:
        pad_left = w_diff // 2
        pad_right = w_diff // 2 + w_diff % 2
        pad_dims.append((pad_left, pad_right))
    elif w_diff < 0:
        slice_left = -1 * w_diff // 2
        slice_right = -1 * (w_diff // 2 - w_diff % 2)
        data = data[:, :, slice_left:-slice_right, :]
        labels = labels[:, :, slice_left:-slice_right, :]
        pad_dims.append((0, 0))

    pad_dims.append((0, 0))

    data = np.pad(data, pad_dims, mode='constant', constant_values=0)
    labels = np.pad(labels, pad_dims, mode='constant', constant_values=0)
    return data, labels


if __name__ == '__main__':
    ones = np.ones((1, 250, 250, 4), dtype=np.uint8)
    ones, labels = preprocess(ones, ones, None)
    print(ones[0, :, :, 2].shape)
    ones = ones[0, :, :, :]
    train, label = train_augmentation(ones, ones[:, :, 0])
    print(train.shape, label.shape)

    brats = BRATSReader(use_hgg=True, use_lgg=False)
    # print(brats.get_mean_dev(.15, 't1ce'))
    train_ids, val_ids, test_ids = brats.get_case_ids(.5)
    case = brats.get_case(train_ids[0])

    label = case['labels']
    slice = np.empty((224, 224, 4))
    slice_index = np.argmax(label.sum(0).sum(0), axis=0)
    orig = label[:, :, slice_index]
    slice[:, :, 0] = case['t1ce'][:, :, slice_index]
    slice[:, :, 1] = case['t1'][:, :, slice_index]
    slice[:, :, 2] = case['t2'][:, :, slice_index]
    slice[:, :, 3] = case['flair'][:, :, slice_index]

    orig_slice = slice
    random.seed()
    np.random.seed()
    slice, label = train_augmentation(slice, label[:, :, slice_index])
    slice = slice[:, :, 0]
    scipy.misc.toimage(orig_slice[:, :, 0], mode='L').show(title='orig data')
    scipy.misc.toimage(slice, mode='L').show(title='data')
    scipy.misc.toimage(label * 255, mode='L').show(title='augmented label')
    scipy.misc.toimage(orig * 255, mode='L').show(title='original label')
