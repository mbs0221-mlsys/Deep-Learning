from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tflearn.datasets.oxflower17 as oxflower17


def AlexNet():
    # 输入数据
    network = input_data(shape=[None, 227, 227, 3])
    # 第一层卷积
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    # 第二层卷积
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    # 第三层卷积
    network = conv_2d(network, 384, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    # 第四层卷积
    network = conv_2d(network, 384, 3, activation='relu')
    # 第五层卷积
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    # 全连接层1
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    # 全连接层2
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    # 输出层
    network = fully_connected(network, 17, activation='softmax')
    network = regression(network, optimizer='momentum', loss='categorical_crossentropy', learning_rate=0.001)
    return network

# 加载数据
X, Y = oxflower17.load_data(dirname='.\\17flowers', one_hot=True, resize_pics=(227, 227))

# 构建模型
alexnet = AlexNet()
modal = tflearn.DNN(alexnet, checkpoint_path='./model/AlexNet/', max_checkpoints=1, tensorboard_verbose=2)
modal.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=200,
          snapshot_epoch=True, run_id='alexnet_oxflower17')

