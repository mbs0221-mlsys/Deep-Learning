import tensorflow as tf


class TWSVM:
    def TWSVM(self):
        A = tf.placeholder(shape=[None, 4], dtype=tf.float32)
        B = tf.placeholder(shape=[None, 4], dtype=tf.float32)
        Y = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        w1 = tf.Variable(tf.random_normal(shape=[4, 1]))
        b1 = tf.Variable(tf.random_normal(shape=[1, 1]))
        w2 = tf.Variable(tf.random_normal(shape=[4, 1]))
        b2 = tf.Variable(tf.random_normal(shape=[1, 1]))
        C1 = tf.constant([0.1])
        C2 = tf.constant([0.1])
        #
        l2_norm1 = tf.reduce_sum(tf.square(tf.matmul(A, w1) + b1))
        loss1 = 0.5 * l2_norm1 + C1 * tf.reduce_sum(tf.maximum(0, 1 + (tf.matmul(B, w1) + b1)))
        #
        l2_norm2 = tf.reduce_sum(tf.square(tf.matmul(B, w2) + b2))
        loss2 = 0.5 * l2_norm2 + C2 + tf.reduce_sum(tf.maximum(0, 1 - (tf.matmul(A, w2) + b2)))
        # prediction
        prediction = tf.sign((tf.matmul(A, w2) + b2) - (tf.matmul(A, w1) + b1))
        pass

    def Fit(self, xTrain, xTest):
        pass

    def Predict(self, xTest):
        pass


