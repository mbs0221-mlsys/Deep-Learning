from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data(path='..\\datasets\\mnist.npz')

print(train_images.shape)
print(len(train_labels))
print(train_labels)

print(test_images.shape)
print(len(test_labels))
print(test_labels)

from keras.models import Sequential
from keras.layers import Dense, Conv2D

network = Sequential(
    Dense(512, activation='relu', input_shape=(28, 28)),
    Dense(10, activation='softmax')
)
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
