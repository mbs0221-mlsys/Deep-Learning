"""
9.5 MNIST的循环神经网络
"""
import tensorflow as tf


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# 初始化偏置
def init_biases(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


x = tf.placeholder(tf.float32, [None, 28, 28])
y = tf.placeholder(tf.float32, [None, 10])

weights = {
    'in': init_weights([28, 128]),
    'out': init_weights([128, 10])
}

biases = {
    'in': init_biases([128]),
    'out': init_biases([10])
}


def RNN(x, W, b):
    X = tf.reshape(x, [-1, 28])

    Xin = tf.matmul(X, W['in']) + biases['in']
    Xin = tf.reshape(Xin, [-1, 28, 128])
    # lstm_cell = tf.contrib.rnn.
    pass
