"""
9.4 MNIST的卷积神经网络
"""

# 9.4.1 加载数据
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

# 9.4.2 构建模型

FLAGS = {
    'data_dir': ''
}

# 加载数据
mnist = input_data.read_data_sets(FLAGS['data_dir'], one_hot=True)


# 初始化权重
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# 初始化偏置
def init_biases(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


# 定义卷积操作
def conv2d(name, x, W, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    return tf.nn.relu(x, name=name)  # 使用relu激活函数


# 定义池化层操作
def maxpool2d(name, x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


# 定义dropout
def dropout(x, d):
    return tf.nn.dropout(x, d)


# 整形
def reshape(x, W):
    return tf.reshape(x[-1, W.getshape().as_list()[0]])


# 全连接层
def full(x, W):
    return tf.nn.relu(tf.matmul(x, W))


# 输出层
def output(x, W, b):
    return tf.add(tf.matmul(x, W), b)


weights = {
    'w1': init_weights([3, 3, 1, 32]),
    'w2': init_weights([3, 3, 32, 64]),
    'w3': init_weights([3, 3, 64, 128]),
    'w4': init_weights([128 * 4 * 4, 625]),
    'wo': init_weights([625, 10])
}


def model(x, W, d, dh):
    # 第一组卷积池化层，随机dropout
    conv1 = conv2d('conv1', x, W['w1'])
    pool1 = maxpool2d('pool1', conv1)
    drop1 = dropout(pool1, d)
    # 第二组卷积池化层，随机dropout
    conv2 = conv2d('conv2', drop1, W['w2'])
    pool2 = maxpool2d('pool2', conv2)
    drop2 = dropout(pool2, d)
    # 第三组卷积池化层，随机dropout
    conv3 = conv2d('conv3', drop2, W['w3'])
    pool3 = maxpool2d('pool3', conv3)
    resp3 = reshape(pool3, W['w4'])
    drop3 = dropout(resp3, d)
    # 全连接层，随机dropout
    full4 = full(drop3, W['w4'])
    drop4 = dropout(full4, dh)
    # 输出层
    out = output(drop4, W['wo'])
    # 返回预测值
    return out


if __name__ == '__main__':
    trainX, trainY = mnist.train
    testX, testY = mnist.test

    trainX = trainX.reshape(-1, 28, 28, 1)  # 28*28*1 input imgg
    testX = testX.reshape(-1, 28, 28, 1)  # 28*28*1 input img

    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    Y = tf.placeholder(tf.float32, [None, 10])

    p_keep_conv = tf.placeholder(tf.float32)
    p_keep_hidden = tf.placeholder(tf.float32)

    cnn = model(X, weights, p_keep_conv, p_keep_hidden)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cnn, labels=Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.arg_max(cnn, 1)
    # 9.4.3 训练模型和评估模型
    batch_size = 128
    test_size = 256
    # Launch the graph in a session
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(100):
            training_batch = zip(range(0, len(trainX), batch_size), range(batch_size, len(trainX) + 1, batch_size))
            for start, end in training_batch:
                sess.run(train_op, feed_dict={X: trainX[start:end], Y: trainY[start:end], p_keep_conv: 0.8, p_keep_hidden: 0.5})
            test_indices = np.arange(len(trainX))  # Get a test batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]
            result = sess.run(predict_op, feed_dict={X: testX[test_indices], p_keep_conv: 1.0, p_keep_hidden: 1.0})
            print(i, np.mean(np.argmax(trainX[test_indices], axis=1) == result))
        pass
