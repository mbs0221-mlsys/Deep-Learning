import numpy as np
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils

D, H, C = 1000, 100, 10

model = Sequential()
model.add(Dense(input_dim=D, units=H))
model.add(Activation(activation='relu'))
model.add(Dense(input_dim=H, units=C))
model.add(Activation(activation='softmax'))

sgd = SGD(lr=1e-3, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

N, N_batch = 1000, 32
X = np.random.randn(N, D)
Y = np.random.randint(C, size=N)
y = np_utils.to_categorical(Y)

model.fit(X, y, nb_epoch=5, batch_size=N_batch, verbose=2)
