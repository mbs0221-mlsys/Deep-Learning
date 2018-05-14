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
X = iris.data
Y = np.array([1 if y == 0 else -1 for y in iris.target])

# 划分数据为训练集和测试集
train_indices = np.random.choice(len(X), round(len(X) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(X))) - set(train_indices)))
x_train = X[train_indices]
y_train = Y[train_indices]
x_test = X[test_indices]
y_test = Y[test_indices]

# 批训练中批的大小
batch_size = 100
x = tf.placeholder(shape=[None, 4], dtype=tf.float32)
y = tf.placeholder(shape=[None, 1], dtype=tf.float32)
w = tf.Variable(tf.random_normal(shape=[4, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))


def SVM(x, w, b):
    y = tf.matmul(x, w) + b
    return y


model = SVM(x, w, b)

# 损失函数
alpha = tf.constant([0.1])
loss = tf.reduce_mean(tf.maximum(0., 1. - model * y)) + alpha * tf.reduce_sum(tf.square(w))

# 输出
predict = tf.sign(model)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), tf.float32))
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 开始训练
sess.run(tf.global_variables_initializer())
loss_vec = []
train_accuracy = []
test_accuracy = []

for i in range(200):
    # 随机选取样本
    rand_index = np.random.choice(len(x_train), size=batch_size)
    rand_x = x_train[rand_index]
    rand_y = np.transpose([y_train[rand_index]])
    # 优化模型
    sess.run(optimizer, feed_dict={x: rand_x, y: rand_y})
    # 训练损失
    train_loss = sess.run(loss, feed_dict={x: rand_x, y: rand_y})
    loss_vec.append(train_loss)
    # 训练精度
    train_acc_temp = sess.run(accuracy, feed_dict={x: x_train, y: np.transpose([y_train])})
    train_accuracy.append(train_acc_temp)
    # 测试精度
    test_acc_temp = sess.run(accuracy, feed_dict={x: x_test, y: np.transpose([y_test])})
    test_accuracy.append(test_acc_temp)

    if (i + 1) % 100 == 0:
        print('Step #' + str(i + 1) + ' W = ' + str(sess.run(w)) + 'b = ' + str(sess.run(b)))
        print('Loss = ' + str(test_acc_temp))

plt.plot(loss_vec)
plt.plot(train_accuracy)
plt.plot(test_accuracy)
plt.legend(['损失', '训练精确度', '测试精确度'])
plt.ylim(0., 1.)
plt.show()
