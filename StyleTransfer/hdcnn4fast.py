"""
Multi-modal Transfer: A Hierarchical Deep Convolutional Neural Network for Fast
"""

import tensorflow as tf

with tf.variable_scope('RGB-block', reuse=True) as scope:
    with tf.variable_scope('conv1', reuse=True) as scope:
        W_conv = weight_variable([9, 9, 3, 32])
