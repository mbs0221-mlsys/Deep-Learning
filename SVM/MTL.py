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
Y1 = tf.placeholder(tf.float32, [None, 10], name="Y1")
Y2 = tf.placeholder(tf.float32, [None, 10], name="Y2")

weights = {
    'sw1': tf.Variable(np.random.rand(10, 20), dtype="float32"),
    'sw2y1': tf.Variable(np.random.rand(20, 10), dtype="float32"),
    'sw2y2': tf.Variable(np.random.rand(20, 10), dtype="float32")
}


# 多任务模型
def MTL(X, w):
    shared_layer = tf.nn.relu(tf.matmul(X, w['sw1']))
    y1 = tf.nn.relu(tf.matmul(shared_layer, w['sw2y1']))
    y2 = tf.nn.relu(tf.matmul(shared_layer, w['sw2y2']))
    return y1, y2


y1, y2 = MTL(X, weights)

# 多任务联合损失函数
Y1_Loss = tf.nn.l2_loss(Y1 - y1)
Y2_Loss = tf.nn.l2_loss(Y2 - y2)
loss = Y1_Loss + Y2_Loss

# 优化器
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
                                        X: np.random.rand(200, 10) * 10,
                                        Y1: np.random.rand(200, 10) * 10,
                                        Y2: np.random.rand(200, 10) * 10,
                                    })
        loss_vec.append(train_loss)
        if i % 20 == 1:
            plt.plot(loss_vec)

plt.plot(loss_vec)
