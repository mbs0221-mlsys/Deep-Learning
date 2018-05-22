# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']

np.random.seed(1)
tf.set_random_seed(1)

sess = tf.Session()

# 产生数据
iris = datasets.load_iris()
x = iris.data
y = np.array([1 if y == 0 else -1 for y in iris.target])

# 划分数据为训练集和测试集
train_indices = np.random.choice(len(x), round(len(x) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x))) - set(train_indices)))
x_train = x[train_indices]
y_train = y[train_indices]
x_test = x[test_indices]
y_test = y[test_indices]

# 批训练中批的大小
batch_size = 100
x_data = tf.placeholder(shape=[None, 4], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
W = tf.Variable(tf.random_normal(shape=[4, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# 定义损失函数
model_output = tf.matmul(x_data, W) + b
l2_norm = tf.reduce_sum(tf.square(W))

# 软正则化参数
alpha = tf.constant([0.1])

# 定义损失函数
classification_term = tf.reduce_mean(tf.maximum(0., 1. - model_output * y_target))
loss = classification_term + alpha * l2_norm

# 输出
prediction = tf.sign(model_output)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 开始训练
sess.run(tf.global_variables_initializer())
loss_vec = []
train_accuracy = []
test_accuracy = []

for i in range(200):

    rand_index = np.random.choice(len(x_train), size=batch_size)
    rand_x = x_train[rand_index]
    rand_y = np.transpose([y_train[rand_index]])

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})

    loss_vec.append(temp_loss)

    train_acc_temp = sess.run(accuracy, feed_dict={x_data: x_train, y_target: np.transpose([y_train])})
    train_accuracy.append(train_acc_temp)

    test_acc_temp = sess.run(accuracy, feed_dict={x_data: x_test, y_target: np.transpose([y_test])})
    test_accuracy.append(test_acc_temp)

    if (i + 1) % 100 == 0:
        print('Step #' + str(i + 1) + ' W = ' + str(sess.run(W)) + 'b = ' + str(sess.run(b)))
        print('Loss = ' + str(test_acc_temp))

plt.plot(loss_vec)
plt.plot(train_accuracy)
plt.plot(test_accuracy)
plt.legend(['损失', '训练精确度', '测试精确度'])
plt.ylim(0., 1.)
plt.show()


