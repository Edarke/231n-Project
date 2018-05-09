# https://github.com/zhixuhao/unet

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.callbacks import *
from keras.layers import *
from keras.models import Model
from keras.optimizers import *
from keras import backend as K
import config as configuration
import metrics
from keras_slice_generator import SliceGenerator
from read_data import BRATSReader
from predict_callback import PredictCallback

# FIXME: bug in Keras makes batchnorm fail with float16, but float16 can be a lot faster if there's a fix.
K.set_floatx('float32')


class myUnet(object):

    def __init__(self, config):
        self.config = config
        self.img_rows = 224
        self.img_cols = 224

    def __pool_layer(self, input, filters, block_num, drop_prob=None, activation='relu', padding='same', init='he_uniform'):
        block_num = str(block_num)
        prefix = 'conv' + block_num + '_'
        reg = tf.keras.regularizers.l2(.001)

        conv1 = Conv2D(filters, 3, activation=activation, padding=padding, kernel_initializer=init, kernel_regularizer=reg, name=prefix+'1')(input)
        conv1 = BatchNormalization()(conv1)  # TODO: Try LayerNorm
        conv1 = Conv2D(filters, 3, activation=activation, padding=padding, kernel_initializer=init, kernel_regularizer=reg, name=prefix+'2')(conv1)
        conv1 = BatchNormalization()(conv1)

        # TODO: Try AveragePooling, or strided convolutions
        pool1 = MaxPooling2D(pool_size=(2, 2), name='pool' + block_num)(conv1)
        if drop_prob is not None:
            pool1 = Dropout(drop_prob, name='drop' + block_num)(pool1)
        return conv1, pool1

    def __unpool_block(self, pooled, prepooled, block_num, drop_prob=None, activation='relu', padding='same', init='he_uniform'):
        filters = prepooled._keras_shape[-1]
        block_num = str(block_num)
        prefix = 'upconv' + block_num + '_'
        reg = tf.keras.regularizers.l2(.001)

        # conv1 = Deconv2D(filters=filters, kernel_size=(3, 3), strides=2, padding=padding, activation=activation, kernel_initializer=init, kernel_regularizer=reg, name=prefix + '1')(pooled)
        up1 = UpSampling2D(size=(2, 2), name='upsample' + block_num)(pooled)
        conv1 = Conv2D(filters=filters, kernel_size=2, activation=activation, padding=padding, kernel_initializer=init, kernel_regularizer=reg, name=prefix+'1')(up1)
        conv1 = BatchNormalization()(conv1)

        merged = concatenate([prepooled, conv1], axis=3, name='merge' + block_num)
        conv1 = Conv2D(filters=filters, kernel_size=2, activation=activation, padding=padding, kernel_initializer=init, kernel_regularizer=reg, name=prefix + '2')(merged)
        conv1 = BatchNormalization()(conv1)

        conv1 = Conv2D(filters=filters, kernel_size=2, activation=activation, padding=padding, kernel_initializer=init, kernel_regularizer=reg, name=prefix + '3')(conv1)
        conv1 = BatchNormalization()(conv1)

        if drop_prob is not None:
            conv1 = Dropout(drop_prob, name='drop' + block_num)(conv1)
        return conv1

    def __get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 1))  # (b, 224, 224, 1)
        filters = 64

        conv224, p112 = self.__pool_layer(inputs, filters=filters, block_num=1)  # (b, 112, 112, 64)
        conv112, p56 = self.__pool_layer(p112, filters=filters*2, block_num=2)   # (b, 56, 56, 128)
        conv56, p28 = self.__pool_layer(p56, filters=filters*4, block_num=3)    # (b, 28, 28, 256)
        conv28, p14 = self.__pool_layer(p28, filters=filters*8, block_num=4)    # (b, 14, 14, 512)
        conv14, _ = self.__pool_layer(p14, filters=filters*16, block_num=5)    # (b, 14, 14, 512)

       # conv14 = Dropout(0.5)(conv14)

        u28 = self.__unpool_block(pooled=conv14, prepooled=conv28, block_num=1)
        u56 = self.__unpool_block(pooled=u28, prepooled=conv56, block_num=2)
        u112 = self.__unpool_block(pooled=u56, prepooled=conv112, block_num=3)
        u224 = self.__unpool_block(pooled=u112, prepooled=conv224, block_num=4)

        predictions = Conv2D(filters=1, kernel_size=1, activation='sigmoid', name='predictions')(u224)
        masked_predictions = Lambda(metrics.keras_mask_predictions)([inputs, predictions])

        model = Model(input=inputs, output=masked_predictions)

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy', metrics.hard_dice])

        return model

    def train(self, train_gen, val_gen):
        model = self.__get_unet()
        print('Fitting model...')

        predict_callback = PredictCallback(train_gen, self.config)
        model_checkpoint = ModelCheckpoint(self.config.results_path + '/unet.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
        logger = CSVLogger(self.config.results_path + '/results.csv')
        earlystopping = EarlyStopping(monitor='val_loss', patience=50)
        callbacks = [TerminateOnNaN(), earlystopping, model_checkpoint, predict_callback, logger]
        model.fit_generator(generator=train_gen, steps_per_epoch=len(train_gen), validation_data=val_gen, validation_steps=len(val_gen), epochs=9000, verbose=1, callbacks=callbacks)

    def save_img(self):
        print("array to image")
        # TODO: save validation predictions
        imgs = np.load('imgs_mask_test.npy')
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img.save("../results/%d.jpg" % (i))


if __name__ == '__main__':
    config = configuration.Config()

    brats = BRATSReader(use_hgg=True, use_lgg=True)
    # print(brats.get_mean_dev(.15, 't1ce'))
    train_ids, val_ids = brats.get_case_ids(config.brats_val_split)

    height, width, slices = brats.get_dims()
    train_datagen = SliceGenerator(brats, slices, train_ids, dim=(config.slice_batch_size, height, width, 1), config=config)
    val_datagen = SliceGenerator(brats, slices, val_ids, dim=(config.slice_batch_size, height, width, 1), config=config)

    myunet = myUnet(config)
    myunet.train(train_datagen, val_datagen)
    myunet.save_img()
