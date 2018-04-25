import config as configuration

from read_data import ATLASReader
import tensorflow as tf
import numpy as np
import eval

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

        resnet_output5 = resnet.graph.get_tensor_by_name("activation_48/Relu:0")  # (b, 7, 7, 2048)
        resnet_output4 = resnet.graph.get_tensor_by_name("activation_39/Relu:0")  # (b, 14, 14, 1024)
        resnet_output3 = resnet.graph.get_tensor_by_name("activation_21/Relu:0")  # (b, 28, 28, 512)
        resnet_output2 = resnet.graph.get_tensor_by_name("activation_9/Relu:0")  # (b, 56, 56, 256)
        resnet_output1 = resnet.graph.get_tensor_by_name("activation/Relu:0")  # (b, 112, 112, 64)
        resnet_output0 = input  # (b, 224, 224, 3)

        print('input shape: ' + str(resnet_output0.shape))
        print('resnet_output1 shape: ' + str(resnet_output1.shape))
        print('resnet_output2 shape: ' + str(resnet_output2.shape))
        print('resnet_output3 shape: ' + str(resnet_output3.shape))
        print('resnet_output4 shape: ' + str(resnet_output4.shape))
        print('resnet_output5 shape: ' + str(resnet_output5.shape))

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
            if resnet_output == resnet_output2: # only size 55 for some reason?
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
        atlas_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels_placeholder, logits=atlas_logits))
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
        data = data[:, slice_top:-slice_bottom + 1, :, :]
        labels = labels[:, slice_top:-slice_bottom + 1, :, :]
        pad_dims.append((0, 0))

    if w_diff > 0:
        pad_left = w_diff // 2
        pad_right = w_diff // 2 + w_diff % 2
        pad_dims.append((pad_left, pad_right))
    elif w_diff < 0:
        slice_left = -1 * w_diff // 2
        slice_right = -1 * (w_diff // 2 - w_diff % 2)
        data = data[:, :, slice_left:-slice_right + 1, :]
        labels = labels[:, :, slice_left:-slice_right + 1, :]
        pad_dims.append((0, 0))

    pad_dims.append((0, 0))

    data = np.pad(data, pad_dims, mode='constant', constant_values=0)
    labels = np.pad(labels, pad_dims, mode='constant', constant_values=0)

    return (data - config.mean) / config.std, labels


def create_atlas_slice_iterator(input_placeholder, label_placeholder, reader, config):
    max_slice_index = 189
    ids = reader.get_case_ids()
    slice_indices = list(range(max_slice_index))
    batch_size = config.slice_batch_size

    def atlas_iterator():
        while True:
            np.random.shuffle(slice_indices)

            for id in ids:
                case = reader.get_case(id)
                case_data = np.expand_dims(case['data'], 0)
                case_labels = np.expand_dims(case['labels'], 0)
                data, labels = preprocess(case_data, case_labels, config)

                data = np.squeeze(data)
                labels = np.squeeze(labels)

                data = data.transpose([2, 0, 1])
                labels = labels.transpose([2, 0, 1])

                for start_slice in range(0, max_slice_index, batch_size):
                    end_slice = min(start_slice + batch_size, max_slice_index)
                    yield {input_placeholder: data[start_slice: end_slice], label_placeholder: labels[start_slice: end_slice]}
                print('Finished ID', id)

    return atlas_iterator


if __name__ == '__main__':
        config = configuration.Config()
        slice_network = Net2D(config)
        [atlas_train_ops], [atlas_predict_op] = slice_network.build()

        reader = ATLASReader()
        ids = reader.get_case_ids()

        case = reader.get_case(ids[0])
        case_data = np.expand_dims(case['data'], 0)
        case_labels = np.expand_dims(case['labels'], 0)

        data, labels = preprocess(case_data, case_labels, config)
        batch_size, height, width, depth = data.shape
        print('Shape is ', (height, width, depth))

        altas_iterator = create_atlas_slice_iterator(slice_network.input_placeholder, slice_network.labels_placeholder,
                                                     reader, config)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            tf.summary.FileWriter(config.output_path, sess.graph)

            for epoch in range(config.epochs):
                for iteration, atlas_feed_dict in enumerate(altas_iterator()):
                    atlas_loss, pred, _ = sess.run(atlas_train_ops, feed_dict=atlas_feed_dict)
                    if iteration % 100 == 0:
                        eval.visualize(config.mean+(atlas_feed_dict[slice_network.input_placeholder][0] * config.std), pred[0], atlas_feed_dict[slice_network.labels_placeholder][0]).show()
                    print('Atlas Loss:', atlas_loss)
                print("Finished Epoch")
