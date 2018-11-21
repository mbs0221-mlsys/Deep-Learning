import keras
from keras.preprocessing import image
from keras import layers
import numpy as np
import os

"""
代码清单8-29 GAN 生成器网络
"""
latent_dim = 32
height = 32
width = 32
channels = 3

generator_input = keras.Input(shape=(latent_dim,))
# 将输入转换为大小为16*16的128个通道的特征图
x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
# 上采样为32*32
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = keras.models.Model(generator_input, x)
generator.summary()

"""
代码清单8-30 判别器网络
"""
discriminator_input = layers.Input(shape=(width, height, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)
# dropout层
x = layers.Dropout(0.5)(x)
# 分类层
x = layers.Dense(1, activation='sigmoid')(x)

discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()

discriminator_optimizer = keras.optimizers.RMSprop(lr=0.008, clipvalue=1.0, decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

"""
代码清单8-31 对抗网络
"""
discriminator.trainable = False
gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)
gan_optimizer = keras.optimizers.RMSprop(lr=0.004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

"""
代码清单8-32 实现GAN的训练
"""
(xtrain, ytrain), (_, _) = keras.datasets.cifar10.load_data()
xtrain = xtrain[ytrain.flatten() == 6]
xtrain = xtrain.reshape((xtrain.shape[0],) + (height, width, channels)).astype('float32') / 255

iterations = 200
batch_size = 20
save_dir = 'gan-8.5'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if os.path.exists('gan-8.5.h5'):
    print('load pre-trained model')
    gan.load_weights('gan-8.5.h5')

start = 0
for step in range(iterations):
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    # 将这些点解码为虚假图像
    generator_image = generator.predict(random_latent_vectors)
    stop = start + batch_size
    real_image = xtrain[start:stop]
    combined_images = np.concatenate([generator_image, real_image])
    # 合并标签，区分虚假和真实的图像，并向标签中添加随机的噪音
    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    labels += 0.5 * np.random.random(labels.shape)
    # 训练判别器
    d_loss = discriminator.train_on_batch(combined_images, labels)
    # 在潜在空间中采样随机点
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    # 合并标签，全都是真实图像
    misleading_targets = np.zeros((batch_size, 1))
    # 通过GAN模型来训练生成器（冻结判别器权重）
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

    # 从头开始
    start += batch_size
    if start > len(xtrain) - batch_size:
        start = 0

    # 每100步保存并绘图
    if step % 20 == 0:
        # 保存模型权重
        gan.save_weights('gan-8.5.h5')
        # 输出指标
        print('discriminator loss:', d_loss)
        print('adversarial loss:', a_loss)
        # 保存一张生成图像
        img = image.array_to_img(generator_image[0] * 255., scale=False)
        img.save(os.path.join(save_dir, str(step) + '-generated_frog.png'))
        # 保存一张真实图像
        img = image.array_to_img(real_image[0] * 255., scale=False)
        img.save(os.path.join(save_dir, str(step) + '-real_frog.png'))
