"""
Python 深度学习
"""

from keras import layers, models
from keras.layers import *
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator

import os

model = models.Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(150, 150, 3)),
    MaxPool2D((2, 2)),
    Conv2D(64, 3, activation='relu'),
    MaxPool2D((2, 2)),
    Conv2D(128, 3, activation='relu'),
    MaxPool2D((2, 2)),
    Conv2D(128, 3, activation='relu'),
    MaxPool2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['acc'])

train = ImageDataGenerator(rescale=1. / 255)
test = ImageDataGenerator(rescale=1. / 255)

dataset_path = '../dataset/'
train_dir = '../dataset/train/'
validation_dir = '../dataset/validation/'

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

if not os.path.exists(train_dir):
    os.mkdir(train_dir)

if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)

train_generator = train.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = test.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break
