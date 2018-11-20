from keras.layers import SimpleRNN, Embedding, Dense
from keras.models import Sequential
from keras.datasets import imdb
from keras import preprocessing

import numpy as np
import matplotlib.pylab as plt

import os

max_features = 10000
max_len = 500
batch_size = 32

print('Loading data')
(xtrain, ytrain), (xtest, ytest) = imdb.load_data(num_words=max_features)

print('Padding sequence')
xtrain = preprocessing.sequence.pad_sequences(xtrain, maxlen=max_len)
xtest = preprocessing.sequence.pad_sequences(xtest, maxlen=max_len)

model = Sequential([
    Embedding(10000, 32),
    SimpleRNN(32, return_sequences=True),
    SimpleRNN(32, return_sequences=True),
    SimpleRNN(32, return_sequences=True),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])
model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# 加载预训练模型
if os.path.exists('example-6.3.h5'):
    print('Load pretrained model.')
    model.load_weights('example-6.3.h5')
history = model.fit(xtrain, ytrain,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

# 代码清单6-15 绘制结果
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.figure()
