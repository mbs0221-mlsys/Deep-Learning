import keras
import numpy as np
import random
import sys
import os

from keras import layers
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 下载并解析初始文本文件
path = keras.utils.get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt'
)

text = open(path).read()
print('Corpus length:', len(text))

# 代码清单8-3 讲字符序列向量化
maxlen = 60
step = 31
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i:i + maxlen])
    next_chars.append(text[i + maxlen])

print('Number of sequences:', len(sentences))

chars = sorted(list(set(text)))
print('Unique characters:', len(chars))
char_indices = dict((char, chars.index(char)) for char in chars)

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# 代码清单8-4 用于预测下一个字符的单层LSTM模型
model = Sequential([
    LSTM(128, input_shape=(maxlen, len(chars))),
    Dense(len(chars), activation='softmax')
])

# 代码清单8.5 模型编译配置
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# 代码清单8.6 给定模型预测，采样下一个字符的函数
def sample(preds, temp=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temp
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# 加载预训练模型
if os.path.exists('example-8.1.h5'):
    print('Load pretrained model.')
    model.load_weights('example-8.1.h5')
# 代码清单8.7 文本生成循环
for epochs in range(1, 10):
    print('epochs ', epochs)
    model.fit(x, y, batch_size=128, epochs=1)
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index:start_index + maxlen]
    print('--- Generating with seed: "' + generated_text + '"')
    for temp in [0.2, 0.5, 1.0, 1.2]:
        print('------ temp: ', temp)
        print(generated_text)
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1

            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temp)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]

            sys.stdout.write(next_char)
    model.save_weights('example-8.1.h5')