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
    return (data - config.mean) / config.std, label


if __name__ == '__main__':
    pass

reader = ATLASReader()
ids = reader.get_case_ids()




height, width, depth = reader.get_case(ids[0])['labels'].shape
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
# resnet_output = resnet.graph.get_tensor_by_name("avg_pool:0")
# print(resnet_output.shape)

with tf.Session() as sess:
    tf.summary.FileWriter(config.output_path, sess.graph)


print([l.name for l in resnet.layers])

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
