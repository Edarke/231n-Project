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
import keras.metrics
from predict_callback import PredictCallback
import keras.losses as losses
from tkinter import *
import augmentation
import eval


# FIXME: bug in Keras makes batchnorm fail with float16, but float16 can be a lot faster if there's a fix.
K.set_floatx('float32')


class myUnet(object):
    def __init__(self, config):
        self.config = config
        self.img_rows = 224
        self.img_cols = 224
        self.model = self.__get_unet()



    def __inception_pool(self, input, filters, block_num, drop_prob=.3, activation='relu', padding='same',
                     init='he_uniform'):
        block_num = str(block_num)
        prefix = 'conv' + block_num + '_'
        reg = tf.keras.regularizers.l2(.0)

        ones = Conv2D(filters//2, 1, activation=activation, padding=padding, kernel_initializer=init,
                       kernel_regularizer=reg, name=prefix + '1x1_1')(input)

        threes = Conv2D(filters//4, 1, activation=activation, padding=padding, kernel_initializer=init,
                      kernel_regularizer=reg, name=prefix + '1x1_3')(input)
        threes = Conv2D(filters//2, 3, activation=activation, padding=padding, kernel_initializer=init,
                      kernel_regularizer=reg, name=prefix + '3x3_3')(threes)

        fives = Conv2D(filters//4, 1, activation=activation, padding=padding, kernel_initializer=init,
                      kernel_regularizer=reg, name=prefix + '1x1_5')(input)
        fives = Conv2D(filters // 2, 3, activation=activation, padding=padding, kernel_initializer=init,
                        kernel_regularizer=reg, name=prefix + '3x3_5p1')(fives)
        fives = Conv2D(filters // 2, 3, activation=activation, padding=padding, kernel_initializer=init,
                        kernel_regularizer=reg, name=prefix + '3x3_5p2')(fives)

        pool = MaxPool2D(pool_size=(3, 3), strides=1, padding=padding, name=block_num + 'pool3x3')(input)
        pool = Conv2D(filters//4, 1, activation=activation, padding=padding, kernel_initializer=init,
                      kernel_regularizer=reg, name=prefix + '1x1')(pool)
        bottle_neck = concatenate([ones, threes, fives, pool], axis=3, name=block_num + '_bottleneck')

        pooled = MaxPooling2D(pool_size=(2, 2), name='pool' + block_num)(bottle_neck)
        if drop_prob is not None:
            pooled = Dropout(drop_prob, name='dropdown' + block_num)(pooled)
        return bottle_neck, pooled

    def __pool_layer(self, input, filters, block_num, drop_prob=.3, activation='relu', padding='same',
                     init='he_uniform'):
        block_num = str(block_num)
        prefix = 'conv' + block_num + '_'
        reg = tf.keras.regularizers.l2(.0)

        conv1 = Conv2D(filters, 3, activation=activation, padding=padding, kernel_initializer=init,
                       kernel_regularizer=reg, name=prefix + '1')(input)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(filters, 3, activation=activation, padding=padding, kernel_initializer=init,
                       kernel_regularizer=reg, name=prefix + '2')(conv1)
        conv1 = BatchNormalization()(conv1)

        # TODO: Try AveragePooling, or strided convolutions
        pool1 = MaxPooling2D(pool_size=(2, 2), name='pool' + block_num)(conv1)
        if drop_prob is not None:
            pool1 = Dropout(drop_prob, name='dropdown' + block_num)(pool1)
        return conv1, pool1

    def __unpool_block(self, pooled, prepooled, block_num, drop_prob=.2, activation='relu', padding='same',
                       init='he_uniform'):
        filters = prepooled._keras_shape[-1]
        block_num = str(block_num)
        prefix = 'upconv' + block_num + '_'
        reg = tf.keras.regularizers.l2(.0)

        # conv1 = Deconv2D(filters=filters, kernel_size=(3, 3), strides=2, padding=padding, activation=activation, kernel_initializer=init, kernel_regularizer=reg, name=prefix + '1')(pooled)
        up1 = UpSampling2D(size=(2, 2), name='upsample' + block_num)(pooled)
        conv1 = Conv2D(filters=filters, kernel_size=2, activation=activation, padding=padding, kernel_initializer=init,
                       kernel_regularizer=reg, name=prefix + '1')(up1)
        conv1 = BatchNormalization()(conv1)

        merged = concatenate([prepooled, conv1], axis=3, name='merge' + block_num)
        conv1 = Conv2D(filters=filters, kernel_size=2, activation=activation, padding=padding, kernel_initializer=init,
                       kernel_regularizer=reg, name=prefix + '2')(merged)
        conv1 = BatchNormalization()(conv1)

        conv1 = Conv2D(filters=filters, kernel_size=2, activation=activation, padding=padding, kernel_initializer=init,
                       kernel_regularizer=reg, name=prefix + '3')(conv1)
        conv1 = BatchNormalization()(conv1)

        if drop_prob is not None:
            conv1 = Dropout(drop_prob, name='dropup' + block_num)(conv1)
        return conv1

    def __get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 4))  # (b, 224, 224, 1)
        filters = 16  # 64

        conv224, p112 = self.__inception_pool(inputs, filters=filters, block_num=1)  # (b, 112, 112, 64)
        conv112, p56 = self.__inception_pool(p112, filters=filters * 2, block_num=2)  # (b, 56, 56, 128)
        conv56, p28 = self.__inception_pool(p56, filters=filters * 4, block_num=3)  # (b, 28, 28, 256)
        conv28, p14 = self.__inception_pool(p28, filters=filters * 8, block_num=4)  # (b, 14, 14, 512)
        conv14, _ = self.__inception_pool(p14, filters=filters * 16, block_num=5)  # (b, 14, 14, 512)

        # conv14 = Dropout(0.5)(conv14)

        u28 = self.__unpool_block(pooled=conv14, prepooled=conv28, block_num=1)
        u56 = self.__unpool_block(pooled=u28, prepooled=conv56, block_num=2)
        u112 = self.__unpool_block(pooled=u56, prepooled=conv112, block_num=3)
        u224 = self.__unpool_block(pooled=u112, prepooled=conv224, block_num=4)

        predictions = Conv2D(filters=4, kernel_size=1, activation='softmax', name='predictions')(u224)
        mask = Lambda(metrics.compute_mask)(inputs)
        masked_predictions = Lambda(lambda mask_n_preds: mask_n_preds[0] * mask_n_preds[1])(
            [mask, predictions])  # multiply([mask, predictions])

        model = Model(inputs=inputs, outputs=masked_predictions)
        model.compile(optimizer=Adam(lr=1e-3), loss=metrics.keras_dice_coef_loss(),
                      metrics=[metrics.category_dice_score(1), metrics.category_dice_score(2),
                               metrics.category_dice_score(3)])

        return model

    def train(self, train_gen, val_gen):
        print('Fitting model...')
        weights_file = 'unet.hdf5'
        if os.path.isfile(weights_file):
            print('Loading weights from ', weights_file)
            self.model.load_weights(weights_file)

        predict_train_callback = PredictCallback(train_gen, self.config, 'train')
        predict_val_callback = PredictCallback(val_gen, self.config, 'val')

        model_checkpoint = ModelCheckpoint(self.config.results_path + '/' + weights_file,
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True)
        logger = CSVLogger(self.config.results_path + '/results.csv')
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
        earlystopping = EarlyStopping(monitor='val_loss', patience=50)
        callbacks = [TerminateOnNaN(), earlystopping, model_checkpoint, predict_train_callback, predict_val_callback,
                     logger, tensorboard]
        history = self.model.fit_generator(generator=train_gen,
                                      steps_per_epoch=len(train_gen),
                                      validation_data=val_gen,
                                      validation_steps=len(val_gen),
                                      epochs=9000,
                                      verbose=1,
                                      callbacks=callbacks)
        return history

    def save_img(self):
        pass

    def evalute_train_and_val_set(self, train_gen, val_gen):
        model = self.model  # Make sure unet.hdf5 is in the current directory
        scores, scores_crf = eval.evaluate(model, train_gen)
        print('Training Dice Scores (No CRF)  WT:%f  TC:%f  ET:%f' % (scores[0], scores[1], scores[2]))
        print('Training Dice Scores (With CRF)  WT:%f  TC:%f  ET:%f' % (scores_crf[0], scores_crf[1], scores_crf[2]))
        scores, scores_crf = eval.evaluate(model, val_gen)
        print('Validation Dice Scores (No CRF)  WT:%f  TC:%f  ET:%f' % (scores[0], scores[1], scores[2]))
        print('Validation Dice Scores (With CRF)  WT:%f  TC:%f  ET:%f' % (scores_crf[0], scores_crf[1], scores_crf[2]))


if __name__ == '__main__':
    config = configuration.Config()

    brats = BRATSReader(use_hgg=True, use_lgg=True)
    # print(brats.get_mean_dev(.15, 't1ce'))
    train_ids, val_ids, test_ids = brats.get_case_ids(config.brats_val_split)

    height, width, slices = brats.get_dims()
    train_datagen = SliceGenerator(brats, slices, train_ids, dim=(config.slice_batch_size, height, width, 4),
                                   config=config, augmentor=augmentation.test_augmentation)
    val_datagen = SliceGenerator(brats, slices, val_ids, dim=(config.slice_batch_size, height, width, 4), config=config,
                                 augmentor=augmentation.test_augmentation)

    net = myUnet(config)
    # net.evalute_train_and_val_set(train_datagen, val_datagen)
    net.train(train_datagen, val_datagen)
