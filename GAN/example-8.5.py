import os

from keras.datasets import fashion_mnist, cifar10
from keras.layers import *
from keras.models import *
from keras.optimizers import RMSprop
from keras.preprocessing import image

"""
代码清单8-29 GAN 生成器网络
"""
latent_dim = 32
height = 32
width = 32
channels = 3

generator_input = Input(shape=(latent_dim,))
generator_cnn = Sequential([
    Dense(128 * 16 * 16),
    LeakyReLU(),
    Reshape((16, 16, 128)),
    Conv2D(256, 5, padding='same'),
    LeakyReLU(),
    Conv2DTranspose(256, 4, strides=2, padding='same'),
    LeakyReLU(),
    Conv2D(256, 5, padding='same'),
    LeakyReLU(),
    Conv2D(256, 5, padding='same'),
    LeakyReLU(),
    Conv2D(channels, 7, activation='tanh', padding='same')
])
generator = Model(generator_input, generator_cnn(generator_input))
generator.summary()

"""
代码清单8-30 判别器网络
"""
discriminator_input = Input(shape=(width, height, channels))
discriminator_cnn = Sequential([
    Conv2D(128, 3),
    LeakyReLU(),
    Conv2D(128, 4, strides=2),
    LeakyReLU(),
    Conv2D(128, 4, strides=2),
    LeakyReLU(),
    Conv2D(128, 4, strides=2),
    LeakyReLU(),
    Flatten(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
discriminator = Model(discriminator_input, discriminator_cnn(discriminator_input))
discriminator.summary()
discriminator_optimizer = RMSprop(lr=0.008, clipvalue=1.0, decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

"""
代码清单8-31 对抗网络
"""
discriminator.trainable = False
gan_input = Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)
gan_optimizer = RMSprop(lr=0.004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

"""
代码清单8-32 实现GAN的训练
"""
(xtrain, ytrain), (_, _) = cifar10.load_data()
xtrain = xtrain[ytrain.flatten() == 2]
xtrain = xtrain.reshape((xtrain.shape[0],) + (height, width, channels)).astype('float32') / 255

iterations = 100
batch_size = 20
save_dir = 'gan-8.5'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if os.path.exists('gan-8.5.h5'):
    print('load pre-trained model')
    gan.load_weights('gan-8.5.h5')

start = 0
for step in range(iterations):

    # 真实图像
    stop = start + batch_size
    real_image = xtrain[start:stop]
    # 虚假图像
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    generator_image = generator.predict(random_latent_vectors)
    # 合并真实虚假图像
    combined_images = np.concatenate([generator_image, real_image])
    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    labels += 0.5 * np.random.random(labels.shape)
    # 训练判别器
    d_loss = discriminator.train_on_batch(combined_images, labels)

    # 训练生成器
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    misleading_targets = np.zeros((batch_size, 1))
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
        img.save(os.path.join(save_dir, str(step) + '-generated.png'))
        # 保存一张真实图像
        img = image.array_to_img(real_image[0] * 255., scale=False)
        img.save(os.path.join(save_dir, str(step) + '-real.png'))
