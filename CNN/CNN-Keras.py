from keras.models import Sequential, save_model, load_model, model_from_json, model_from_yaml
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K

batch_size = 128
nb_classes = 10  # 分类数
nb_epoch = 12  # 训练轮数

# 输入图片的维度
img_rows, img_cols = 28, 28

# 卷积滤镜的个数
nb_filters = 32

# 最大池化，池化核大小
pool_size = (2, 2)

# 卷积核大小
kernel_size = (3, 3)


# 卷积神经网络
def CNN(nb_filters, kernel_size, pool_size, input_shape):
    model = Sequential()
    # 卷积层
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    # 卷积层
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    # 池化层
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    # 输出层
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    # 交叉熵损失函数，AdaDelta优化
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
if K.image_dim_ordering() == 'th':
    # 使用Theano的顺序
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    # 使用Tensorflow的顺序
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# 将类向量转换为二进制类矩阵
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

cnn = CNN(nb_filters, kernel_size, pool_size, input_shape)
cnn.fit(X_train, Y_train, nb_epoch=5, batch_size=32, verbose=1, validation_data=(X_test, Y_test))

score = cnn.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# 保存模型结构
yaml_string = cnn.to_yaml()
json_string = cnn.to_json()

# 加载模型
model = model_from_yaml(yaml_string)
model = model_from_json(json_string)

# 保存/加载模型权重
cnn.save_weights('my_model_weights.h5')
cnn.load_weights('my_model_weights.h5')
