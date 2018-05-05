# https://github.com/zhixuhao/unet

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
import config as configuration

from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator

from keras_slice_generator import SliceGenerator
from read_data import BRATSReader
import config

class myUnet(object):
    def __init__(self, config, img_rows, img_cols):
        self.config = config
        self.img_rows = img_rows
        self.img_cols = img_cols


    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 1))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv1_1')(inputs)
        print("conv1 shape:", conv1.shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv1_2')(conv1)
        print("conv1 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print("pool1 shape:", pool1.shape)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv2_1')(pool1)
        print("conv2 shape:", conv2.shape)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv2_2')(conv2)
        print("conv2 shape:", conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print("pool2 shape:", pool2.shape)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv3_1')(pool2)
        print("conv3 shape:", conv3.shape)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv3_2')(conv3)
        print("conv3 shape:", conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print("pool3 shape:", pool3.shape)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv4_1')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv4_2')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv5_1')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv5_2')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='up1_conv1')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='up1_conv2')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='up1_conv3')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='up2_conv1')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='up2_conv2')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='up2_conv3')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='up3_conv1')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='up3_conv2')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='up3_conv3')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='up4_conv1')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='up4_conv2')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='up4_conv3')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='up4_conv4')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid', name='prediction')(conv9)

        model = Model(input=inputs, output=conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self, train_gen, val_gen):
        print("loading data")
        print("loading data done")
        model = self.get_unet()
        print("got unet")

        print('Fitting model...')
        model_checkpoint = ModelCheckpoint(self.config.results_path + 'unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        callbacks = [TerminateOnNaN(), EarlyStopping(monitor='val_loss', patience=5), model_checkpoint]
        model.fit_generator(generator=train_gen, validation_data=val_gen, steps_per_epoch=len(train_gen), validation_steps=len(val_gen), epochs=9000, verbose=1, callbacks=callbacks)
        print('predict test data')
#        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
#        np.save('../results/imgs_mask_test.npy', imgs_mask_test)

    def save_img(self):
        print("array to image")
        imgs = np.load('imgs_mask_test.npy')
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img.save("../results/%d.jpg" % (i))


if __name__ == '__main__':
    config = configuration.Config()

    brats = BRATSReader(use_hgg=True, use_lgg=True)
    train_ids, val_ids = brats.get_case_ids(config.brats_val_split)

    height, width, slices = brats.get_dims()
    train_datagen = SliceGenerator(brats, slices, train_ids, dim=(16, height, width, 1))
    val_datagen = SliceGenerator(brats, slices, val_ids, dim=(16, height, width, 1))

    myunet = myUnet(config, height, width)
    myunet.train(train_datagen, val_datagen)
    myunet.save_img()
