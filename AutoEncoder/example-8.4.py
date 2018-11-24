import keras
from keras.layers import *
from keras.models import *
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os


class VAELayer(Layer):
    """
    代码清单8-26 用于计算VAE损失的自定义层
    """

    def vae_loss(self, z_mean, z_log_var, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs, **kwargs):
        z_mean, z_log_var, x, z_decoded = inputs
        loss = self.vae_loss(z_mean, z_log_var, x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x


class SamplingLayer(Layer):
    """
    代码清单8-24 潜在空间采样层
    """

    def __init__(self, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

    @staticmethod
    def sampling(z_mean, z_log_var):
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
        return z_mean + K.exp(z_log_var) * epsilon

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        z = self.sampling(z_mean, z_log_var)
        return z


def vae_auto_encoder(img_shape, latent_dim):
    """
    VAE 自编码器

    :param img_shape: 输入形状
    :param latent_dim: 隐含空间维度
    :return: VAE 自编码器
    """
    # 编码器网络
    encoder_input = Input(shape=img_shape)
    x = Conv2D(32, 3, padding='same', activation='relu')(encoder_input)
    x = Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    shape_before_flattening = K.int_shape(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)
    encoder = Model(encoder_input, [z_mean, z_log_var])
    [z_mean, z_log_var] = encoder(encoder_input)
    # 潜在空间采样函数
    z = SamplingLayer(latent_dim)([z_mean, z_log_var])
    # 解码器网络，将潜在空间点映射为图像
    decoder_input = Input(K.int_shape(z[1:]))
    x = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
    x = Reshape(shape_before_flattening[1:])(x)
    x = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x)
    x = Conv2D(1, 3, padding='same', activation='sigmoid')(x)
    decoder = Model(decoder_input, x)
    z_decoded = decoder(z)
    # 计算VAE损失的自定义层
    y = VAELayer()([z_mean, z_log_var, encoder_input, z_decoded])
    # VAE自编码器
    vae = Model(encoder_input, y)
    return [encoder, decoder, vae]


if __name__ == '__main__':
    # 代码清单8-27 训练VAE
    img_shape = (28, 28, 1)
    batch_size = 16
    latent_dim = 5

    [encoder, decoder, vae] = vae_auto_encoder(img_shape, latent_dim)
    vae.compile(optimizer='rmsprop', loss=None)
    vae.summary()

    (x_train, _), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype('float32') / 255
    x_test = x_test.reshape(x_test.shape + (1,))

    model_path = './vae-8.4.hdf5'
    if os.path.exists(model_path):
        vae.load_weights(model_path)
    vae.fit(x=x_train, y=None,
            shuffle=True,
            epochs=10,
            batch_size=batch_size,
            validation_data=(x_test, None))
    vae.save_weights(model_path)

    # 代码清单8-28 从二维潜在空间中采样一组点的网格，将其解码为图像
    n = 15
    digit_size = 28
    digits = []
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    for i, yi in enumerate(grid_x):
        for j, xj in enumerate(grid_y):
            z_sample = np.array([[xj, yi]])
            z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
            x_decoded = decoder.predict(z_sample, batch_size=batch_size)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            digits.append(digit)
    digits = np.reshape(digits, (n, n))
