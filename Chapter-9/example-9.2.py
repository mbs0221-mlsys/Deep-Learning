"""
9.2 MNIST的分类问题
"""

# 9.2.1 加载数据
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = {
    'data_dir': ''
}

# 加载数据
mnist = input_data.read_data_sets(FLAGS['data_dir'], one_hot=True)

# 9.2.2 构建回归模型
# 定义回归模型
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b  # 预测值

# 定义损失函数和优化器
y_ = tf.placeholder(tf.float32, [None, 10])  # 输入的真实值的占位符
# 使用交叉熵作为评估函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
# 使用SGD作为优化器
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 9.2.3 训练模型
# 使用交互式会话
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 9.2.4 评估模型
# 评估训练好的模型
correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(y_, 1))  # 计算预测值和真实值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
