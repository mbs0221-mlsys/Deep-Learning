"""
Tensorflow 技术解析与实战
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, UpSampling2D, Conv2D, Convolution2D, Input, Flatten, Embedding, \
    BatchNormalization
from keras.layers import LeakyReLU, Dropout, Multiply
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing import image
from keras import layers
from keras.layers import merge

import os
import numpy as np
import pickle


def build_generator(num_classes, latent_size):
    cnn = Sequential([
        Dense(1024, input_dim=latent_size, activation='relu'),
        Dense(128 * 7 * 7, activation='relu'),
        Reshape((7, 7, 128)),
        UpSampling2D(size=(2, 2)),
        Conv2D(256, (5, 5), padding="same", activation="relu"),
        UpSampling2D(size=(2, 2)),
        Conv2D(128, (5, 5), padding="same", activation="relu"),
        Conv2D(1, (2, 2), padding="same", activation="tanh")
    ])

    latent = Input(shape=(latent_size,))
    image_class = Input(shape=(1,), dtype='int32')

    cls = Flatten()(Embedding(num_classes, latent_size)(image_class))
    fake_image = cnn(Multiply()([latent, cls]))

    return Model(inputs=[latent, image_class], outputs=[fake_image])


def build_discriminator():
    cnn = Sequential([
        Conv2D(32, 3, strides=(2, 2), padding='same'),
        LeakyReLU(),
        Dropout(0.3),
        Conv2D(64, 3, strides=(1, 1), padding='same'),
        LeakyReLU(),
        Dropout(0.3),
        Conv2D(128, 3, strides=(2, 2), padding='same'),
        LeakyReLU(),
        Dropout(0.3),
        Conv2D(256, 3, strides=(1, 1), padding='same'),
        LeakyReLU(),
        Dropout(0.3),
        Flatten()
    ])
    image = Input(shape=[28, 28, 1])
    features = cnn(image)
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(10, activation='softmax', name='auxiliary')(features)
    return Model(inputs=image, outputs=[fake, aux])


if __name__ == '__main__':

    # num_classes
    num_classes = 10

    # 定义超参数
    nb_epochs = 30
    batch_size = 25
    latent_size = 100

    # 优化器的学习率
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # 构建判别网络
    discriminator = build_discriminator()
    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )

    # 构建生成器网络
    generator = build_generator(num_classes, latent_size)
    generator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy']
    )

    latent = Input(shape=(latent_size,))
    image_class = Input(shape=(1,), dtype='int32')

    # 生成虚假图片
    fake = generator([latent, image_class])

    # 生成组合模型
    discriminator.trainable = False
    fake, aux = discriminator(fake)
    combined = Model([latent, image_class], [fake, aux])

    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )

    # 读取MNIST数据集
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data('../datasets/mnist.npz')
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255

    train_images = train_images[:2000]
    train_labels = train_labels[:2000]
    test_images = test_images[:1000]
    test_labels = test_labels[:1000]
    # train_labels = to_categorical(train_labels)
    # test_labels = to_categorical(test_labels)

    nb_train, nb_test = train_images.shape[0], test_images.shape[0]
    history_train = {'generator': [], 'discriminator': []}
    history_test = {'generator': [], 'discriminator': []}

    HEADER_FMT = '{0:<22s}|{1:4s}|{2:15s}|{3:5s}'
    ROW_FMT = '{0:<22s}|{1:<4.2f}|{2:<15.2f}|{3:<5.2f}'
    save_dir = 'gan-13.3'

    # 创建文件夹
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # 加载检查点
    if os.path.exists('params_generator.hdf5'):
        generator.load_weights('params_generator.hdf5', True)

    if os.path.exists('params_discriminator.hdf5'):
        discriminator.load_weights('params_discriminator.hdf5', True)

    for epochs in range(nb_epochs):
        print('Epoch {} of {}'.format(epochs + 1, nb_epochs))
        nb_batches = int(train_images.shape[0] / batch_size)
        epochs_gen_loss = []
        epochs_disc_loss = []
        for i in range(nb_batches):
            # 产生一个批次的均匀分布
            noise = np.random.uniform(0, 1, (batch_size, latent_size))
            # 获取一个批次的真实数据
            image_batch = train_images[i * batch_size:(i + 1) * batch_size]
            label_batch = train_labels[i * batch_size:(i + 1) * batch_size]
            # 生成一些噪音标记
            sampled_labels = np.random.randint(0, num_classes, batch_size)
            # 产生一个批次的虚假图片
            generated_images = generator.predict([noise, sampled_labels.reshape((-1, 1))], verbose=0)
            # 构造真实、虚假数据集
            X = np.concatenate((image_batch, generated_images))
            y = np.array([1] * batch_size + [0] * batch_size)
            aux_y = np.concatenate((label_batch, sampled_labels), axis=0)
            # 记录判别器损失
            epochs_disc_loss.append(discriminator.train_on_batch(X, [y, aux_y]))
            # 产生两个批次的噪声和标记
            noise = np.random.uniform(0, 1, (2 * batch_size, latent_size))
            sampled_labels = np.random.randint(0, num_classes, 2 * batch_size)
            trick = np.ones(2 * batch_size)
            gen_loss = combined.train_on_batch([noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels])
            epochs_gen_loss.append(gen_loss)
            print('Batches:{}'.format(gen_loss))
        print('\nTesting for epochs {}'.format(epochs + 1))

        # 评估测试集

        # 产生一个新批次的噪声数据
        noise = np.random.uniform(0, 1, (nb_test, latent_size))
        sampled_labels = np.random.randint(0, num_classes, nb_test)
        generated_images = generator.predict([noise, sampled_labels.reshape((-1, 1))], verbose=False)

        X = np.concatenate([test_images, generated_images])
        y = np.array([1] * nb_test + [0] * nb_test)
        aux_y = np.concatenate((test_labels, sampled_labels))

        # 看看判别器能不能鉴别
        discriminator_test_loss = discriminator.evaluate(X, [y, aux_y], verbose=False)
        discriminator_train_loss = np.mean(np.array(epochs_disc_loss), axis=0)

        # 创建两个批次的噪声数据
        noise = np.random.uniform(0, 1, (2 * nb_test, latent_size))
        sampled_labels = np.random.randint(0, num_classes, 2 * nb_test)

        trick = np.ones(2 * nb_test)

        generator_test_loss = combined.evaluate(
            [noise, sampled_labels.reshape((-1, 1))],
            [trick, sampled_labels], verbose=False)

        generator_train_loss = np.mean(np.array(epochs_gen_loss), axis=0)

        # 记录损失值信息
        history_train['generator'].append(generator_train_loss)
        history_train['discriminator'].append(discriminator_train_loss)

        history_test['generator'].append(generator_test_loss)
        history_test['discriminator'].append(discriminator_test_loss)

        print(HEADER_FMT.format('component', *discriminator.metrics_names))
        print('-' * 65)
        print(ROW_FMT.format('generator (train)', *history_train['generator'][-1]))
        print(ROW_FMT.format('generator (test)', *history_test['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)', *history_train['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)', *history_test['discriminator'][-1]))

        # 每一个epochs保存权重
        generator.save_weights('params_generator.hdf5', True)
        discriminator.save_weights('params_discriminator.hdf5', True)

        # 生成一些可视化的数字来看演化过程
        noise = np.random.uniform(0, 1, (30, latent_size))
        sampled_labels = np.array([[i] * 3 for i in range(num_classes)])
        generated_images = generator.predict([noise, sampled_labels.reshape((-1, 1))], verbose=0)

        # 整理到一个方格中
        img = (np.concatenate([r.reshape((-1, 28, 1)) for r in np.split(generated_images, num_classes)], axis=1) * 255).astype(np.uint8)
        img = image.array_to_img(img, scale=False)
        img.save(os.path.join(save_dir, 'generated-f' + str(epochs) + '.png'))

    pickle.dump({'train': history_train, 'test': history_test}, open('acgan-history.pkl', 'wb'))
