"""
Image Captioning
"""

from keras.models import Sequential
from keras.layers import Dense, Reshape, UpSampling2D, Conv2D, BatchNormalization, LSTM, Flatten


def im2txt(shape):
    cnn = Sequential([
        Dense(1024, input_shape=shape, activation='relu'),
        Conv2D(256, 5, activation="relu"),
        Conv2D(128, 5, activation="relu"),
        Conv2D(1, 2, activation="tanh"),
        Flatten()
    ])
    lstm = Sequential([
        LSTM()
    ])
