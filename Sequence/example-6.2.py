from keras import Sequential
from keras.layers import Flatten, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import matplotlib.pylab as plt

import os

"""
Python 深度学习
6.2 理解循环神经网络
"""

# 数据集 http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
imdb_dir = '../../datasets/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding='UTF-8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

# 代码清单6-9 队IMDB原始数据的文本进行分词

max_len = 100
train_samples = 200
validation_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=max_len)

labels = np.asarray(labels)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

xtrain = data[:train_samples]
ytrain = labels[:train_samples]
xtest = data[train_samples:train_samples + validation_samples]
ytest = labels[train_samples:train_samples + validation_samples]

# 代码清单 6-10 解析GloVe词嵌入文件
glove_dir = '../../glove/glove.6B'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='UTF-8')
lines = f.readlines()
for line in lines:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s words vectors.' % len(embeddings_index))

# 代码清单6-11 准备GloVe词嵌入矩阵
embedding_dim = 100
embeddings_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector

# 代码清单6-12 定义模型
model = Sequential([
    Embedding(max_words, embedding_dim, input_length=max_len),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.summary()

# 代码清单6-13 将预训练的词嵌入加载到Embedding层中
model.layers[0].set_weights([embeddings_matrix])
model.layers[0].trainable = False

# 代码清单6-14 训练与评估
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

# 加载预训练模型
if os.path.exists('example-6.2.h5'):
    print('Load pretrained model.')
    model.load_weights('example-6.2.h5')

history = model.fit(xtrain, ytrain,
                    epochs=20,
                    batch_size=32,
                    validation_data=(xtest, ytest))
model.save_weights('example-6.2.h5')

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
plt.savefig('fig-6.2-accuracy.jpg')
plt.show()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.savefig('fig-6.2-loss.jpg')
plt.show()
