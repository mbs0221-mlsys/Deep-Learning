# -*- coding :utf-8 -*-

import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

# 设置GPU配置
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# 导入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print(mnist.train.images.shape)

lr = 1e-3

input_size = 28
time_step_size = 28
hidden_size = 256
layer_num = 2
class_num = 10

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, class_num])
keep_prob = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, [])


# 输入
x = tf.reshape(X, [-1, 28, 28])
lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
mlstm_cell = rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)
init_state = mlstm_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
# 方法1，dynamic rnn
outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=x, initial_state=init_state, time_major=False)
h_state = outputs[:, -1, :]
# h_state = state[-1][1]
# 方法2，按时间步展开计算
# outputs = list()
# state = init_state
# with tf.variable_scope('RNN'):
#     for time_step in range(time_step_size):
#         if time_step > 0:
#             tf.get_variable_scope().reuse_variables()
#         (cell_output, state) = mlstm_cell(x[:, time_step, :], state)
#         outputs.append(cell_output)
#
# h_state = outputs[-1]

# 开始训练和测试
W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
b = tf.Variable(tf.constant(0.1, shape=[class_num]), dtype=tf.float32)
y = tf.nn.softmax(tf.matmul(h_state, W) + b)


# 损失和评估函数
cross_entropy = -tf.reduce_mean(Y * tf.log(y))
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

sess.run(tf.global_variables_initializer())
_batch_size = 128
for i in range(2000):
    batch = mnist.train.next_batch(_batch_size)
    sess.run(train_op, feed_dict={X: batch[0], Y: batch[1], keep_prob: 0.5, batch_size: _batch_size})
    if (i + 1) % 200 == 0:
        feed = {X: batch[0], Y: batch[1], keep_prob: 1.0, batch_size: _batch_size}
        train_accuracy = sess.run(accuracy, feed_dict=feed)
        print('%d\t%d\t%d' % (mnist.train.epochs_completed, i + 1, train_accuracy))

dict = {X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0, batch_size: mnist.test.images.shape[0]}
test_accuracy = sess.run(accuracy, feed_dict=dict)
print('%d\t%d\t%d' % test_accuracy)
