#  GRAPH CODE
# ============

# 导入 Tensorflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ======================
# 定义图
# ======================

# 定义占位符
X = tf.placeholder(tf.float32, [None, 10], name="X")
Y1 = tf.placeholder(tf.int32, [None, 1], name="Y1")
Y2 = tf.placeholder(tf.int32, [None, 1], name="Y2")
Y3 = tf.placeholder(tf.int32, [None, 1], name="Y3")

weights = {
    'sw1': tf.Variable(np.random.rand(10, 10), dtype="float32"),
    'sw2y1': tf.Variable(np.random.rand(10, 1), dtype="int32"),
    'sw2y2': tf.Variable(np.random.rand(10, 1), dtype="int32"),
    'sw2y3': tf.Variable(np.random.rand(10, 1), dtype="int32")
}


# Multi-Task Model
def MTL(X, w):
    shared_layer = tf.nn.relu(tf.matmul(X, w['sw1']))
    y1 = tf.nn.relu(tf.matmul(shared_layer, w['sw2y1']))
    y2 = tf.nn.relu(tf.matmul(shared_layer, w['sw2y2']))
    y3 = tf.nn.relu(tf.matmul(shared_layer, w['sw2y3']))
    return y1, y2, y3


y1, y2, y3 = MTL(X, weights)


# 计算loss
def Loss(Y1, Y2, Y3, y1, y2, y3):
    Y1_Loss = tf.nn.l2_loss(Y1 - y1)
    Y2_Loss = tf.nn.l2_loss(Y2 - y2)
    Y3_Loss = tf.nn.l2_loss(Y3 - y3)
    return Y1_Loss + Y2_Loss + Y3_Loss


# 优化器
loss = Loss(Y1, Y2, Y3, y1, y2, y3)
optimizer = tf.train.AdamOptimizer().minimize(loss)

# 联合训练
# Calculation (Session) Code
# ==========================

# open the session
loss_vec = []
with tf.Session() as session:
    for i in range(200):
        session.run(tf.global_variables_initializer())
        _, train_loss = session.run([optimizer, loss],
                                    {
                                        X: tf.random_normal(shape=[100, 10], dtype=tf.float32),
                                        Y1: tf.random_uniform(shape=[100, 1], dtype=tf.int32),
                                        Y2: tf.random_uniform(shape=[100, 1], dtype=tf.int32),
                                        Y3: tf.random_uniform(shape=[100, 1], dtype=tf.int32)
                                    })
        loss_vec.append(train_loss)

plt.plot(loss_vec)
