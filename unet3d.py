# https://github.com/zhixuhao/unet

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.callbacks import *
from keras.layers import *
from keras.models import Model
from keras.optimizers import *
from keras import backend as K
import config as configuration
import metrics
from keras_slice_generator import SliceGenerator, SliceGenerator3D
from read_data import BRATSReader
import keras.metrics
from predict_callback import PredictCallback
import keras.losses as losses
from tkinter import *
import augmentation

# FIXME: bug in Keras makes batchnorm fail with float16, but float16 can be a lot faster if there's a fix.
K.set_floatx('float32')


class UNet3D(object):

    def __init__(self, config):
        self.config = config
        self.img_rows = 224
        self.img_cols = 224
        self.img_depth = 128

    def __pool_layer(self, input, filters, block_num, drop_prob=.2, activation='relu', padding='same', init='he_uniform'):
        block_num = str(block_num)
        prefix = 'conv' + block_num + '_'
        reg = tf.keras.regularizers.l2(.0)

        conv1 = Conv3D(filters, 3, activation=activation, padding=padding, kernel_initializer=init, kernel_regularizer=reg, name=prefix+'1')(input)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv3D(filters, 3, activation=activation, padding=padding, kernel_initializer=init, kernel_regularizer=reg, name=prefix+'2')(conv1)
        conv1 = BatchNormalization()(conv1)

        # TODO: Try AveragePooling, or strided convolutions
        pool1 = MaxPooling3D(pool_size=(2, 2, 2), name='pool' + block_num)(conv1)
        if drop_prob is not None:
            pool1 = Dropout(drop_prob, name='dropdown' + block_num)(pool1)
        return conv1, pool1

    def __unpool_block(self, pooled, prepooled, block_num, drop_prob=.2, activation='relu', padding='same', init='he_uniform'):
        filters = prepooled._keras_shape[-1]
        block_num = str(block_num)
        prefix = 'upconv' + block_num + '_'
        reg = tf.keras.regularizers.l2(.0)

        # conv1 = Deconv2D(filters=filters, kernel_size=(3, 3), strides=2, padding=padding, activation=activation, kernel_initializer=init, kernel_regularizer=reg, name=prefix + '1')(pooled)
        up1 = UpSampling3D(size=(2, 2, 2), name='upsample' + block_num)(pooled)
        conv1 = Conv3D(filters=filters, kernel_size=2, activation=activation, padding=padding, kernel_initializer=init, kernel_regularizer=reg, name=prefix+'1')(up1)
        conv1 = BatchNormalization()(conv1)

        merged = concatenate([prepooled, conv1], axis=4, name='merge' + block_num)
        conv1 = Conv3D(filters=filters, kernel_size=2, activation=activation, padding=padding, kernel_initializer=init, kernel_regularizer=reg, name=prefix + '2')(merged)
        conv1 = BatchNormalization()(conv1)

        conv1 = Conv3D(filters=filters, kernel_size=2, activation=activation, padding=padding, kernel_initializer=init, kernel_regularizer=reg, name=prefix + '3')(conv1)
        conv1 = BatchNormalization()(conv1)

        if drop_prob is not None:
            conv1 = Dropout(drop_prob, name='dropup' + block_num)(conv1)
        return conv1

    def __get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, self.img_depth, 4))  # (b, 224, 224, 1)
        filters = 8 # 64

        conv224, p112 = self.__pool_layer(inputs, filters=filters, block_num=1)  # (b, 112, 112, 16)
        conv112, p56 = self.__pool_layer(p112, filters=filters*2, block_num=2)   # (b, 56, 56, 32)
        conv56, p28 = self.__pool_layer(p56, filters=filters*4, block_num=3)    # (b, 28, 28, 64)
        conv28, p14 = self.__pool_layer(p28, filters=filters*8, block_num=4)    # (b, 14, 14, 128)
        conv14, _ = self.__pool_layer(p14, filters=filters*16, block_num=5)    # (b, 14, 14, 256)

       # conv14 = Dropout(0.5)(conv14)

        u28 = self.__unpool_block(pooled=conv14, prepooled=conv28, block_num=1)
        u56 = self.__unpool_block(pooled=u28, prepooled=conv56, block_num=2)
        u112 = self.__unpool_block(pooled=u56, prepooled=conv112, block_num=3)
        u224 = self.__unpool_block(pooled=u112, prepooled=conv224, block_num=4)

        predictions = Conv3D(filters=4, kernel_size=1, activation='softmax', name='predictions')(u224)
        mask = Lambda(metrics.compute_mask)(inputs)
        masked_predictions = Lambda(lambda mask_n_preds: mask_n_preds[0] * mask_n_preds[1])([mask, predictions])  # multiply([mask, predictions])

        model = Model(inputs=inputs, outputs=masked_predictions)
        model.compile(optimizer=Adam(lr=1e-3), loss=metrics.keras_dice_coef_loss(), metrics=[metrics.category_dice_score(1), metrics.category_dice_score(2), metrics.category_dice_score(3)])

        return model

    def train(self, train_gen, val_gen):
        model = self.__get_unet()
        print('Fitting model...')

        predict_train_callback = PredictCallback(train_gen, self.config, 'train')
        predict_val_callback = PredictCallback(val_gen, self.config, 'val')

        model_checkpoint = ModelCheckpoint(self.config.results_path + '/unet.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
        logger = CSVLogger(self.config.results_path + '/results.csv')
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,  write_graph=True, write_images=True)
        earlystopping = EarlyStopping(monitor='val_loss', patience=50)
        callbacks = [TerminateOnNaN(), earlystopping, model_checkpoint, predict_train_callback, predict_val_callback, logger, tensorboard]
        model.fit_generator(generator=train_gen, steps_per_epoch=len(train_gen), validation_data=val_gen, validation_steps=len(val_gen), epochs=9000, verbose=1, callbacks=callbacks)

    def save_img(self):
        pass


if __name__ == '__main__':
    config = configuration.Config()

    brats = BRATSReader(use_hgg=True, use_lgg=True)
    # print(brats.get_mean_dev(.15, 't1ce'))
    train_ids, val_ids, test_ids = brats.get_case_ids(config.brats_val_split)

    height, width, slices = brats.get_dims()
    train_datagen = SliceGenerator3D(brats, slices, train_ids, dim=(1, height, width, slices, 4), config=config, augmentor=None)
    val_datagen = SliceGenerator3D(brats, slices, val_ids, dim=(1, height, width, slices, 4), config=config, augmentor=None)

    myunet = UNet3D(config)
    myunet.train(train_datagen, val_datagen)
    myunet.save_img()
