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
X = tf.placeholder("float", [10, 10], name="X")
Y1 = tf.placeholder("float", [10, 20], name="Y1")
Y2 = tf.placeholder("float", [10, 20], name="Y2")
Y3 = tf.placeholder("float", [10, 20], name="Y3")

# 定义权重
initial_shared_layer_weights = np.random.rand(10, 20)
initial_Y1_layer_weights = np.random.rand(20, 20)
initial_Y2_layer_weights = np.random.rand(20, 20)
initial_Y3_layer_weights = np.random.rand(20, 20)

shared_layer_weights = tf.Variable(initial_shared_layer_weights, name="share_W", dtype="float32")
Y1_layer_weights = tf.Variable(initial_Y1_layer_weights, name="share_Y1", dtype="float32")
Y2_layer_weights = tf.Variable(initial_Y2_layer_weights, name="share_Y2", dtype="float32")
Y3_layer_weights = tf.Variable(initial_Y3_layer_weights, name="share_Y3", dtype="float32")

# 使用relu激活函数构建层
shared_layer = tf.nn.relu(tf.matmul(X, shared_layer_weights))
Y1_layer = tf.nn.relu(tf.matmul(shared_layer, Y1_layer_weights))
Y2_layer = tf.nn.relu(tf.matmul(shared_layer, Y2_layer_weights))
Y3_layer = tf.nn.relu(tf.matmul(shared_layer, Y3_layer_weights))

# 计算loss
Y1_Loss = tf.nn.l2_loss(Y1 - Y1_layer)
Y2_Loss = tf.nn.l2_loss(Y2 - Y2_layer)
Y3_Loss = tf.nn.l2_loss(Y3 - Y3_layer)
Joint_Loss = Y1_Loss + Y2_Loss + Y3_Loss

# 优化器
Optimizer = tf.train.AdamOptimizer().minimize(Joint_Loss)

# 联合训练
# Calculation (Session) Code
# ==========================

# open the session
loss_vec = []
with tf.Session() as session:
    for i in range(200):
        session.run(tf.global_variables_initializer())
        _, Joint_Loss = session.run([Optimizer, Joint_Loss],
                                    {
                                        X: tf.random_normal(shape=[100, 10], dtype=tf.float32),
                                        Y1: tf.random_normal(shape=[100, 1], dtype=tf.float32),
                                        Y2: tf.random_normal(shape=[100, 1], dtype=tf.float32),
                                        Y3: tf.random_normal(shape=[100, 1], dtype=tf.float32)
                                    })
        loss_vec.append(Joint_Loss)
    # print(Joint_Loss)

plt.plot(loss_vec)