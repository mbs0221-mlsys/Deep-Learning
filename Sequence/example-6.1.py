from keras.datasets import imdb
from keras.layers import Flatten, Dense, Embedding, SimpleRNN
from keras import preprocessing
from keras import Sequential

max_features = 1000
max_len = 20

(xtrain, ytrain), (xtest, ytest) = imdb.load_data(num_words=max_features)
xtrain = preprocessing.sequence.pad_sequences(xtrain, maxlen=max_len)
xtest = preprocessing.sequence.pad_sequences(xtest, maxlen=max_len)

model = Sequential([
    Embedding(10000, 8, input_length=max_len),
    Flatten(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
model.summary()
history = model.fit(xtrain, ytrain,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)
