import tensorflow as tf
import numpy as np

# 8.1.1 生成及加载数据

# 构造满足一元二次方程的函数
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# 加入一些噪声点，使他与x_data的维度保持一致，并且拟合均值为0，方差为0.05的正态分布
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise  # y = x^2 - 0.5 + 噪声

# 接下来定义x和y的占位符作为将要输入神经网络的变量
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])


# 8.1.2 构建网络模型
def add_layer(inputs, in_size, out_size, activation_function=None):
    # 构建权重
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 构建偏置
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 矩阵相乘
    Wx_plus_b = tf.matmul(inputs, weights) + biases
    # 激活函数
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    # 输出数据
    return outputs


# 构建隐藏层
h1 = add_layer(xs, 1, 20, activation_function=tf.nn.relu)
# 构建输出层
prediction = add_layer(h1, 20, 1, activation_function=None)
# 计算预测值和真实值之间的误差
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=1))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 8.1.3 训练模型

init = tf.global_variables_initializer()  # 初始化所有变量
sess = tf.Session()
sess.run(init)

# 训练1000次
for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    # 每50次打印一次损失值
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
