import config as configuration

from read_data import ATLASReader
import tensorflow as tf
import numpy as np

config = configuration.Config()



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

    pad_dims = [(0,0)]
    if h_diff > 0:
        pad_top = h_diff//2
        pad_bottom = h_diff//2 + h_diff%2
        pad_dims.append((pad_top, pad_bottom))
    elif h_diff < 0:
        slice_top = -1 * h_diff//2
        slice_bottom = -1 * (h_diff//2 - h_diff%2)
        data = data[:,slice_top:-slice_bottom+1,:,:]
        labels = labels[:,slice_top:-slice_bottom+1,:,:]
        pad_dims.append((0,0))

    if w_diff > 0:
        pad_left = w_diff//2
        pad_right = w_diff//2 + w_diff%2
        pad_dims.append((pad_left, pad_right))
    elif w_diff < 0:
        slice_left = -1 * w_diff//2
        slice_right = -1 * (w_diff//2 - w_diff%2)
        data = data[:,:,slice_left:-slice_right+1,:]
        labels = labels[:,:,slice_left:-slice_right+1,:]
        pad_dims.append((0,0))

    pad_dims.append((0,0))

    data = np.pad(data, pad_dims, mode='constant', constant_values=0)
    labels = np.pad(labels, pad_dims, mode='constant', constant_values=0)

    return (data - config.mean) / config.std, labels


if __name__ == '__main__':
    pass

reader = ATLASReader()
ids = reader.get_case_ids()

case = reader.get_case(ids[0])
case_data = np.expand_dims(case['data'], 0)
case_labels = np.expand_dims(case['labels'], 0)

data, labels = preprocess(case_data, case_labels)
batch_size, height, width, depth = data.shape
print('Shape is ', (height, width, depth))

input = tf.placeholder(dtype=tf.float16, shape=[None, height, width, 1])
input = tf.stack([input, input, input], axis=3)

resnet = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
  #  input_shape=None,
    pooling=None,
)
#

resnet_output5 = resnet.graph.get_tensor_by_name("activation_48/Relu:0") # (b, 7, 7, 2048)
resnet_output4 = resnet.graph.get_tensor_by_name("activation_39/Relu:0") # (b, 14, 14, f)
resnet_output3 = resnet.graph.get_tensor_by_name("activation_21/Relu:0") # (b, 28, 28, f)
resnet_output2 = resnet.graph.get_tensor_by_name("activation_9/Relu:0") # (b, 56, 56, f)
resnet_output1 = resnet.graph.get_tensor_by_name("activation/Relu:0")    # (b, 112, 112, f)
resnet_output0 = input                                                   # (b, 224, 224, 3)

print(resnet_output0.shape)
print(resnet_output1.shape)
print(resnet_output2.shape)
print(resnet_output3.shape)
print(resnet_output4.shape)
print(resnet_output5.shape)


with tf.Session() as sess:
    tf.summary.FileWriter(config.output_path, sess.graph)



#
# class Net2D(object):
#
#
#     def __init__(self, config):
#         self.config = config
#
#
#
#     def __build_block(self, input, name):
#
#
#     def build(self):
#         input = tf.placeholder(dtype=self.config.dtype, shape=[None, None, None])
#
