import tensorflow as tf

# 输入数据
from tensorflow.examples.tutorials.mnist import input_data

minit = input_data.read_data_sets('/tmp/data/', one_hot=True)

# 定义网络的超参数
learning_rate = 0.001
training_iter = 200000
batch_size = 128
display_step = 10

# 定义网络的参数
n_input = 784  # 输入的维度
n_classes = 10  # 标记的维度
dropout = 0.75  # Dropout的概率

# 输入占位符
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout


# 定义卷积操作
def conv2d():
    pass


# 定义池化层操作
def maxpool2d():
    pass


# 规范化操作
def norm():
    pass


# 定义所有的网络参数
weights = {
    'wc1': tf.Variable(tf.random_normal([11, 11, 1, 96])),
    'wc2': tf.Variable(tf.random_normal([11, 11, 1, 96])),
    'wc3': tf.Variable(tf.random_normal([11, 11, 1, 96])),
    'wc4': tf.Variable(tf.random_normal([11, 11, 1, 96])),
    'wc5': tf.Variable(tf.random_normal([11, 11, 1, 96])),
    'wd1': tf.Variable(tf.random_normal([11, 11, 1, 96])),
    'wd2': tf.Variable(tf.random_normal([11, 11, 1, 96])),
    'out': tf.Variable(tf.random_normal([11, 11, 1, 96])),
}
biases = {
    'bc1': tf.Variable(tf.random_normal([11, 11, 1, 96])),
    'bc2': tf.Variable(tf.random_normal([11, 11, 1, 96])),
    'bc3': tf.Variable(tf.random_normal([11, 11, 1, 96])),
    'bc4': tf.Variable(tf.random_normal([11, 11, 1, 96])),
    'bc5': tf.Variable(tf.random_normal([11, 11, 1, 96])),
    'bd2': tf.Variable(tf.random_normal([11, 11, 1, 96])),
    'bd3': tf.Variable(tf.random_normal([11, 11, 1, 96])),
    'out': tf.Variable(tf.random_normal([11, 11, 1, 96])),
}


# 定义整个网络
def AlexNet(x, weights, biases, dropout):
    pass
