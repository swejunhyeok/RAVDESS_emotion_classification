import joblib
import numpy as np

import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint

def model():
    input_video = keras.layers.Input(shape=(16, 128, 128, 1))
    video_conv1 = keras.layers.Conv3D(64, (3,3,3), padding='same', strides = (1,2,2))(input_video)
    video_relu1 = keras.layers.LeakyReLU()(video_conv1)

    video_conv2 = keras.layers.Conv3D(128, (3,3,3), padding='same', strides = (1,2,2))(video_relu1)
    video_bn2 = keras.layers.BatchNormalization()(video_conv2)
    video_relu2 = keras.layers.LeakyReLU()(video_bn2)

    video_conv3 = keras.layers.Conv3D(256, (3,3,3), padding='same', strides = (2,2,2))(video_relu2)
    video_bn3 = keras.layers.BatchNormalization()(video_conv3)
    video_relu3 = keras.layers.LeakyReLU()(video_bn3)

    video_conv4 = keras.layers.Conv3D(512, (3,3,3), padding='same', strides = (2,2,2))(video_relu3)
    video_bn4 = keras.layers.BatchNormalization()(video_conv4)
    video_relu4 = keras.layers.LeakyReLU()(video_bn4)

    video_conv5 = keras.layers.Conv3D(1024, (3,3,3), padding='same', strides = (2,2,2))(video_relu4)
    video_bn5 = keras.layers.BatchNormalization()(video_conv5)
    video_relu5 = keras.layers.LeakyReLU()(video_bn5)

    video_conv6 = keras.layers.Conv3D(2048, (3,3,3), padding='same', strides = (2,2,2))(video_relu5)
    video_bn6 = keras.layers.BatchNormalization()(video_conv6)
    video_relu6 = keras.layers.LeakyReLU()(video_bn6)

    input_audio = keras.layers.Input(shape=(64916, 1))
    audio_conv1 = keras.layers.Conv1D(64, 25, padding='same', strides = 8)(input_audio)
    audio_relu1 = keras.layers.LeakyReLU()(audio_conv1)

    audio_conv2 = keras.layers.Conv1D(128, 25, padding='same', strides = 8)(audio_relu1)
    audio_bn2 = keras.layers.BatchNormalization()(audio_conv2)
    audio_relu2 = keras.layers.LeakyReLU()(audio_bn2)

    audio_conv3 = keras.layers.Conv1D(256, 25, padding='same', strides = 8)(audio_relu2)
    audio_bn3 = keras.layers.BatchNormalization()(audio_conv3)
    audio_relu3 = keras.layers.LeakyReLU()(audio_bn3)

    audio_conv4 = keras.layers.Conv1D(512, 25, padding='same', strides = 4)(audio_relu3)
    audio_bn4 = keras.layers.BatchNormalization()(audio_conv4)
    audio_relu4 = keras.layers.LeakyReLU()(audio_bn4)

    audio_conv5 = keras.layers.Conv1D(1024, 25, padding='same', strides = 4)(audio_relu4)
    audio_bn5 = keras.layers.BatchNormalization()(audio_conv5)
    audio_relu5 = keras.layers.LeakyReLU()(audio_bn5)

    video_flatten = keras.layers.Flatten()(video_relu6)
    audio_flatten = keras.layers.Flatten()(audio_relu5)

    concat_layer = keras.layers.concatenate([video_flatten, audio_flatten])
    dense = keras.layers.Dense(8)(concat_layer)
    bn = keras.layers.BatchNormalization()(dense)
    softmax = keras.layers.Softmax()(bn)


    model = keras.models.Model(inputs=[input_video, input_audio], outputs=softmax)
    return model

def main():
    train_save_dir = 'RAVDESS_Model/train'
    test_save_dir = 'RAVDESS_Model/test'

    train_video = joblib.load(train_save_dir + '/video.joblib')
    train_audio = joblib.load(train_save_dir + '/audio.joblib')
    train_audio_mfcc = joblib.load(train_save_dir + '/audio_mfcc.joblib')
    train_y = joblib.load(train_save_dir + '/y.joblib')

    test_video = joblib.load(test_save_dir + '/video.joblib')
    test_audio = joblib.load(test_save_dir + '/audio.joblib')
    test_audio_mfcc = joblib.load(test_save_dir + '/audio_mfcc.joblib')
    test_y = joblib.load(test_save_dir + '/y.joblib')


    print(train_video.shape, train_audio.shape, train_audio_mfcc.shape, train_y.shape)
    print(test_video.shape, test_audio.shape, test_audio_mfcc.shape, test_y.shape)

    train_video = np.expand_dims(train_video, axis=4)
    train_audio = np.expand_dims(train_audio, axis=2)
    train_audio_mfcc = np.expand_dims(train_audio_mfcc, axis=2)

    test_video = np.expand_dims(test_video, axis=4)
    test_audio = np.expand_dims(test_audio, axis=2)
    test_audio_mfcc = np.expand_dims(test_audio_mfcc, axis=2)

    print(train_video.shape, train_audio.shape, train_audio_mfcc.shape, train_y.shape)
    print(test_video.shape, test_audio.shape, test_audio_mfcc.shape, test_y.shape)

    classification_model = model()

    opt = keras.optimizers.rmsprop(lr=0.00005)

    classification_model.summary()

    classification_model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    cnnhistory=classification_model.fit([train_video, train_audio], train_y, batch_size=16, epochs=100, shuffle=True, validation_data=([test_video, test_audio], test_y))

if __name__ == "__main__":
    main()