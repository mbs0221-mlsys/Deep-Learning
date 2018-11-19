import numpy as np
import tensorflow as tf

N, D, H, C = 64, 1000, 100, 10

x = tf.placeholder(tf.float32, shape=[None, D])
y = tf.placeholder(tf.float32, shape=[None, C])

w1 = tf.Variable(1e-3 * np.random.randn(D, H).astype(np.float32))
w2 = tf.Variable(1e-3 * np.random.randn(H, C).astype(np.float32))

o = tf.matmul(x, w1)
o = tf.nn.relu(o)
o = tf.matmul(o, w2)
o = tf.nn.softmax(o)
loss = -tf.reduce_sum(y * tf.log(o))

learning_rate = 1e-2
train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss)

X = np.random.randn(N, D, ).astype(np.float32)
Y = np.zeros((N, C)).astype(np.float32)
Y[np.arange(N), np.random.randint(C, size=N)] = 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t in range(100):
        _, loss_value = sess.run([train_step, loss], feed_dict={x: X, y: Y})
        print(loss_value)
