import config
from read_data import ATLASReader
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    config = config.Config()


reader = ATLASReader()
ids = reader.get_case_ids()

mean = 0
std = 0
for id in ids:
    d = reader.get_case(ids[0])
    label = d['labels']
    data = d['data']
    mean += np.mean(data)
    std += np.std(data)

mean /= len(ids)
std /= len(ids)

print("Mean is %d, Std deviation is %d" % (mean, std))

height, width, depth = reader.get_case(ids[0])['labels'].shape
print('Shape is ', (height, width, depth))


input = tf.placeholder(dtype=tf.float16, shape=[height, width, 1])
input = tf.tile(input, multiples=3).reshape([height, width, 3])






resnet = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
  #  input_shape=None,
    pooling=None,
)


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
