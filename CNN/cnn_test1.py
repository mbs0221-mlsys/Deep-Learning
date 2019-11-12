from keras import layers
from keras import models
import matplotlib.pyplot as plt
import os
import numpy as np

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data('../../datasets/mnist.npz')
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_images = train_images[:2000]
train_labels = train_labels[:2000]
test_images = test_images[:1000]
test_labels = test_labels[:1000]

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

if os.path.exists('cnn_test1.hdf5'):
    model.load_weights('cnn_test1.hdf5')

isTraining = False
if isTraining:
    model.fit(train_images, train_labels, epochs=5, batch_size=32)
model.save_weights('cnn_test1.hdf5')

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)

layer_outputs = [layer.output for layer in model.layers[:5]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(np.expand_dims(train_images[1], axis=0))

n_cols = 16
for layer_name, layer_activation in zip([layer.name for layer in model.layers[:5]], activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_rows = n_features // n_cols

    grid = np.reshape(layer_activation[0], (n_rows, n_cols, size, size))
    grid = np.concatenate(grid, axis=1)
    grid = np.concatenate(grid, axis=1)

    plt.figure(figsize=(grid.shape[1] / size, grid.shape[0] / size))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(grid, aspect='auto', cmap='viridis')
    plt.savefig(layer_name + '.jpg')
    plt.show()