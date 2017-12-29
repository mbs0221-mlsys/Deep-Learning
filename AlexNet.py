import tensorflow as tf

# 输入数据
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

# 定义网络的超参数
learning_rate = 0.001
training_iter = 200000
batch_size = 128
display_step = 10

# 定义网络的参数
n_input = 784  # 输入的维度
n_classes = 10  # 标记的维度
dropout = 0.75  # Dropout的概率

# 输入占位符
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout


# 定义卷积操作
def conv2d(name, x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x, name=name)  # 使用relu激活函数


# 定义池化层操作
def maxpool2d(name, x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


# 规范化操作
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


# 全连接操作
def full(name, x, W, b, dropout):
    fc = tf.reshape(x[-1, W.getshape().as_list()[0]])
    fc = tf.add(tf.matmul(fc, W), b)
    fc = tf.nn.relu(fc)
    # dropout
    fc = tf.nn.dropout(fc, dropout)
    return fc


# 输出层
def output(x, W, b):
    return tf.add(tf.matmul(x, W), b)


# 定义所有的网络参数
weights = {
    'wc1': tf.Variable(tf.random_normal([11, 11, 1, 96])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
    'wd1': tf.Variable(tf.random_normal([4 * 4 * 256, 4096])),
    'wd2': tf.Variable(tf.random_normal([4096, 4096])),
    'out': tf.Variable(tf.random_normal([4096, 10])),
}

biases = {
    'bc1': tf.Variable(tf.random_normal([96])),
    'bc2': tf.Variable(tf.random_normal([256])),
    'bc3': tf.Variable(tf.random_normal([384])),
    'bc4': tf.Variable(tf.random_normal([384])),
    'bc5': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([4096])),
    'bd2': tf.Variable(tf.random_normal([4096])),
    'out': tf.Variable(tf.random_normal([n_classes])),
}


# 定义AlexNet的网络模型

# 定义整个网络
def AlexNet(x, W, b, d):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, -1])

    # 第一层卷积
    # 卷积
    conv1 = conv2d('conv1', x, W['wc1'], b['bc1'])
    # 下采样
    pool1 = maxpool2d('pool1', conv1, k=2)
    # 规范化
    norm1 = norm('norm1', pool1, lsize=4)

    # 第二层卷积
    # 卷积
    conv2 = conv2d('conv2', norm1, W['wc2'], b['bc2'])
    # 最大池化（向下采样）
    pool2 = maxpool2d('pool2', conv2, k=2)
    # 规范化
    norm2 = norm('norm2', pool2, lsize=4)

    # 第三层卷积
    # 卷积
    conv3 = conv2d('conv3', norm2, W['wc3'], b['bc3'])
    # 最大池化（向下采样）
    pool3 = maxpool2d('pool3', conv3, k=2)
    # 规范化
    norm3 = norm('norm3', pool3, lsize=4)

    # 第四层卷积
    conv4 = conv2d('conv4', norm3, W['wc4'], b['bc4'])

    # 第五层卷积
    # 卷积
    conv5 = conv2d('conv5', conv4, W['wc5'], b['bc5'])
    # 下采样
    pool5 = maxpool2d('pool5', conv5, k=2)
    # 规范化
    norm5 = norm('norm5', pool5, lsize=4)

    # 全连接层1
    fc1 = full('fc1', norm5, W['wd1'], b['bd1'], d)

    # 全连接层2
    fc2 = full('fc2', fc1, W['wd2'], b['bd2'], d)

    # 输出层
    out = output(fc2, W['out'], b['out'])
    return out


# 构建模型，定义损失函数和优化器，并构建评估函数

# 构建模型
modal = AlexNet(x, weights, biases, dropout)

# 定义损失函数和优化器
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(modal, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 评估函数
correct_pred = tf.equal(tf.argmax(modal, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    # 开始训练，直到达到training_iter，即200000
    while step * batch_size < training_iter:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        if step % display_step == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
            print('Iter ', str(step * batch_size), ', Mini batch Loss= ', '{:.6f}'.format(loss), '{:.5f}'.format(acc))
        step += 1
    print('Optimization Finished!')
    # 计算测试集的准确度
    acc = sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.images[:256], keep_prob: 1.0})
    print('Testing Accuracy:', acc)
