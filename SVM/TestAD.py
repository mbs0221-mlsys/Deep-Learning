import tensorflow as tf


def func(x):
    a1 = 64 * x
    a2 = 1 - x
    a3 = tf.pow(1 - 2 * x, 2)
    a4 = tf.pow(1 - 8 * x + 8 * x * x, 2)
    output = a1 * a2 * a3 * a4
    return output


x = tf.placeholder(dtype=tf.float32, shape=[None, 2])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
network = func(x)

predict = tf.sign(network)
loss = tf.reduce_mean(tf.maximum(0, 1 - predict * y))
accuracy = tf.reduce_mean(tf.abs(predict - y))

grad = tf.gradients(network, [x])
init = tf.global_variables_initializer()
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss=loss)

X = tf.random_normal(shape=[10, 2], dtype=tf.float32)
Y = tf.random_uniform(shape=[10, 1], dtype=tf.int32)

with tf.Session() as sess:
    sess.run(init)
    val = sess.run(network, feed_dict={x: X})
    grad = sess.run(grad, feed_dict={x: X})
    print(val)
    print(grad)
    loss = sess.run(optimizer, feed_dict={x: X, y: Y})
