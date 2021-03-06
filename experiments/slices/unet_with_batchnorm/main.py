import numpy as np
import tensorflow as tf

import config as configuration
import eval
import metrics
from read_data import ATLASReader


class UNet(object):
    def __init__(self, config):
        self.config = config
        self.input_placeholder = tf.placeholder(dtype=self.config.dtype, shape=[None, 224, 224])  # (b, h, w)
        self.labels_placeholder = tf.placeholder(shape=(None, None, None), dtype=self.config.dtype)  # (b, h, w)
        self.atlas_train_op = None

    def _pool_block(self, input, filters, block_num, activation=tf.nn.leaky_relu):
        prefix = "conv" + block_num + "_"
        conv1 = tf.layers.conv2d(input, filters=filters, kernel_size=3, padding='SAME', name=prefix + '1',
                                 activation=activation)
        conv2 = tf.layers.conv2d(conv1, filters=filters, kernel_size=3, padding='SAME', name=prefix + '2',
                                 activation=activation, strides=2)
        print(conv2.shape)
        return conv2

    def _unpool_block(self, pooled, prepooled, block_num, activation=tf.nn.leaky_relu):
        prefix = 'deconv' + block_num + '_'
        filters = pooled.shape[-1]
        deconv1 = tf.layers.conv2d_transpose(pooled, filters=filters, kernel_size=3, strides=2, padding='SAME', name=prefix + '1',
                                             activation=activation)
        print(pooled.shape, deconv1.shape, prepooled.shape)

        concat = tf.concat([prepooled, deconv1], axis=-1, name=prefix + 'concat')
        deconv2 = tf.layers.conv2d(concat, filters=filters, kernel_size=2, padding='SAME', name=prefix + 'conv1',
                                   activation=activation)
        return deconv2

    def build(self):
        filters = 16
        s224 = tf.expand_dims(self.input_placeholder, -1)
        s112 = self._pool_block(s224, filters, '1')
        s56 = self._pool_block(s112, filters * 2, '2')
        s28 = self._pool_block(s56, filters * 4, '3')
        s14 = self._pool_block(s28, filters * 8, '4')
        s7 = self._pool_block(s14, filters * 16, '5')
        u14 = self._unpool_block(s7, s14, '1')
        u28 = self._unpool_block(u14, s28, '2')
        u56 = self._unpool_block(u28, s56, '3')
        u112 = self._unpool_block(u56, s112, '4')
        u224 = self._unpool_block(u112, s224, '5')

        logits = tf.squeeze(tf.layers.conv2d(u224, filters=1, kernel_size=1, name='logits', activation=None))
        probs = tf.sigmoid(logits, 'probs')
        loss = tf.nn.weighted_cross_entropy_with_logits(self.labels_placeholder, logits, pos_weight=48)
        loss = tf.reduce_mean(loss)
        opt = tf.train.AdamOptimizer().minimize(loss)
        return [(loss, probs, opt)], [probs]


class Net2D(object):
    def __init__(self, config):
        self.config = config
        self.input_placeholder = tf.placeholder(dtype=self.config.dtype, shape=[None, 224, 224])  # (b, h, w)
        self.labels_placeholder = tf.placeholder(shape=(None, None, None), dtype=self.config.dtype)  # (b, h, w)
        self.atlas_train_op = None

    def build(self):

        input = tf.stack([self.input_placeholder, self.input_placeholder, self.input_placeholder], axis=-1)
        print(input.shape)

        resnet = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=input,
            pooling=None,
        )

        if config.freeze_resnet:
            for layer in resnet.layers:
                layer.trainable = False

        resnet_output5 = resnet.graph.get_tensor_by_name("activation_48/Relu:0")  # (b, 7, 7, 2048)
        resnet_output4 = resnet.graph.get_tensor_by_name("activation_39/Relu:0")  # (b, 14, 14, 1024)
        resnet_output3 = resnet.graph.get_tensor_by_name("activation_21/Relu:0")  # (b, 28, 28, 512)
        resnet_output2 = resnet.graph.get_tensor_by_name("activation_9/Relu:0")   # (b, 56, 56, 256)
        resnet_output1 = resnet.graph.get_tensor_by_name("activation/Relu:0")     # (b, 112, 112, 64)
        resnet_output0 = input  # (b, 224, 224, 3)

        print('input shape: ' + str(resnet_output0.shape))
        print('resnet_output1 shape: ' + str(resnet_output1.shape))
        print('resnet_output2 shape: ' + str(resnet_output2.shape))
        print('resnet_output3 shape: ' + str(resnet_output3.shape))
        print('resnet_output4 shape: ' + str(resnet_output4.shape))
        print('resnet_output5 shape: ' + str(resnet_output5.shape))

        classification = tf.layers.conv2d(
            inputs=resnet_output5,
            filters=256,
            kernel_size=(3, 3),
            padding='SAME',
            strides=(2, 2),
            activation=tf.nn.relu)  # 4x4

        classification = tf.layers.conv2d(
            inputs=classification,
            filters=64,
            kernel_size=(3, 3),
            padding='SAME',
            strides=(2, 2),
            activation=tf.nn.relu)  # 2x2

        classification = tf.layers.max_pooling2d(classification, pool_size=(2, 2), strides=(2, 2))
        classification = tf.layers.conv2d(
            inputs=classification,
            filters=1,
            kernel_size=(1, 1),
            padding='SAME',
            activation=None  # 1x1
        )
        classification = tf.squeeze(classification)
        has_lesion = tf.minimum(1., tf.reduce_sum(self.labels_placeholder, axis=[1, 2]))

        # Transpose Layer 4: (N, 7, 7, 2048) -> (N, 14, 14, 1024)
        # Transpose Layer 3: (N, 14, 14, 1024) -> (N, 28, 28, 512)
        # Transpose Layer 2: (N, 28, 28, 512) -> (N, 56, 56, 256)
        # Transpose Layer 1: (N, 56, 56, 256) -> (N, 112, 112, 64)

        resnet_outputs = [resnet_output4, resnet_output3, resnet_output2, resnet_output1]
        filter_sizes = [1024, 512, 256, 64, 3]

        in_layer = resnet_output5
        for i, filter_size, resnet_output in zip(range(len(filter_sizes)), filter_sizes, resnet_outputs):
            # Transpose convolution to get (N, H/2, W/2, filter_size)
            print('InLayer Shape:', in_layer.shape)

            t_conv = tf.layers.conv2d_transpose(
                in_layer,
                filter_size,
                (2, 2),
                strides=(2, 2),
                activation=tf.nn.relu)

            # Stack with output of 4th layer of ResNet to get (N, H/2, W/2, filter_size * 2)
            print('Resnet Conv Shape:', resnet_output.shape, "Upsampled Shape", t_conv.shape)
            if resnet_output == resnet_output2:  # only size 55 for some reason?
                resnet_output = tf.pad(resnet_output, [[0, 0], [1, 0], [1, 0], [0, 0]], "CONSTANT")

            t_conv_stacked = tf.concat([resnet_output, t_conv], axis=3)
            print("Concat Shape", t_conv.shape)

            # Convolutions to decrease number of filters to 1024; back to (N, H/2, W/2, filter_size)
            t_conv_a = tf.layers.conv2d(
                inputs=t_conv_stacked,
                filters=filter_size,
                kernel_size=(3, 3),
                padding='SAME',
                strides=(1, 1),
                activation=tf.nn.relu)
            print("Convolved Shape", t_conv.shape)

            in_layer = t_conv_a

            print('output of transpose layer ' + str(i) + ': ' + str(in_layer.shape))

        #
        # Output Layer: (N, 112, 112, 64) -> (N, 224, 224, 1)
        #

        # Transpose convolution to get (N, 224, 224, 1)
        atlas_logits = tf.layers.conv2d_transpose(
            inputs=in_layer,
            filters=64,
            kernel_size=(2, 2),
            strides=(2, 2),
            activation=tf.nn.relu)  # Return logits

        atlas_logits = tf.layers.conv2d(
            inputs=atlas_logits,
            filters=1,
            kernel_size=(1, 1),
            padding='SAME',
            strides=(1, 1),
            activation=None)

        atlas_logits = tf.squeeze(atlas_logits)
        atlas_predictions = tf.nn.sigmoid(atlas_logits)

        print('output_layer shape: ' + str(atlas_logits.shape))
        atlas_loss = metrics.binary_crossentropy(labels=self.labels_placeholder, logits=atlas_logits,
                                                 pos_weight=40)

        # atlas_loss = metrics.soft_dice(y_true=self.labels_placeholder, y_pred=atlas_predictions)
        atlas_train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(atlas_loss)

        return ([(atlas_loss, atlas_predictions, atlas_train_op)], [atlas_predictions])

    def train(self, slice_batch, label_batch):
        pass

    def predict(self, slice_batch):
        pass

    def test(self, slice_batch, label_batch):
        pass


def preprocess(data, labels, config):
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

    return (data - config.mean) / config.std, labels


def create_atlas_slice_iterator(reader, config):
    max_slice_index = 189
    ids = reader.get_case_ids()
    batch_size = config.slice_batch_size

    def atlas_iterator():

        for id in ids:
            case = reader.get_case(id)
            case_data = np.expand_dims(case['data'], 0)
            case_labels = np.expand_dims(case['labels'], 0)
            data, labels = preprocess(case_data, case_labels, config)

            data = np.squeeze(data)
            labels = np.squeeze(labels)


            data = data.transpose([2, 0, 1])
            labels = np.minimum(labels.transpose([2, 0, 1]), 1)

            pos_examples = np.max(labels, axis=(1, 2)).astype(np.bool)

            data = data[pos_examples]
            labels = labels[pos_examples]
            slice_indices = np.arange(len(labels))
            np.random.shuffle(slice_indices)

            for start_slice in range(0, len(slice_indices), batch_size):
                end_slice = min(start_slice + batch_size, max_slice_index)
                yield data[slice_indices[start_slice: end_slice]], labels[slice_indices[start_slice: end_slice]]

    return atlas_iterator


if __name__ == '__main__':
    config = configuration.Config()
    slice_network = UNet(config) #Net2D(config)
    [atlas_train_ops], [atlas_predict_op] = slice_network.build()

    reader = ATLASReader()
    ids = reader.get_case_ids()

    case = reader.get_case(ids[0])
    case_data = np.expand_dims(case['data'], 0)
    case_labels = np.expand_dims(case['labels'], 0)

    data, labels = preprocess(case_data, case_labels, config)
    batch_size, height, width, depth = data.shape
    print('Atlas shape is ', (height, width, depth))

    atlas_iterator = create_atlas_slice_iterator(reader, config)

    with tf.Session() as sess:
        loss_var = tf.Variable(0.)
        writer = tf.summary.FileWriter(config.output_path, sess.graph)
        tf.summary.scalar("loss", loss_var)
        write_op = tf.summary.merge_all()

        sess.run(tf.global_variables_initializer())

        for epoch in range(config.epochs):
            for iteration, (data, labels) in enumerate(atlas_iterator()):
                atlas_loss, pred, _ = sess.run(atlas_train_ops, feed_dict={slice_network.input_placeholder: data,
                                                                           slice_network.labels_placeholder: labels})
                writer.add_summary(sess.run(write_op, {loss_var: atlas_loss}), iteration)
                writer.flush()
                if iteration % 1000 == 0:
                    index = np.argmax(labels.sum(axis=1).sum(axis=1))
                    eval.visualize(config.mean + (data[index] * config.std), pred[index], labels[index]).show()
                print(epoch, iteration, 'Atlas Loss:', atlas_loss)
