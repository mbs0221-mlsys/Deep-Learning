from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19
from keras import backend as K
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time

"""
[2] Image Style Transfer Using Convolutional Neural Networks. Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge. CVPR 2016. 
"""

target_image_path = './img/pic1.jpg'
style_reference_image_path = './img/timg2.jpg'

width, height = load_img(target_image_path).size
img_height = 240
img_width = width * img_height // height


def preprocess_image(image_path, size):
    img = load_img(image_path, target_size=size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, :: -1]
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def content_loss(base, combination):
    """
    内容损失

    :param base:
    :param combination:
    :return:
    """
    return K.sum(K.square(combination - base))


def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    return K.dot(features, K.transpose(features))


def style_loss(style, combination):
    """
    样式损失

    :param style:
    :param combination:
    :return:
    """
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C) / (4. * (channels ** 2) * (size ** 2)))


def total_variation_loss(x):
    a = K.square(x[:, :img_height - 1, :img_width - 1, :] -
                 x[:, 1:, :img_width - 1, :])
    b = K.square(x[:, :img_height - 1, :img_width - 1, :] -
                 x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


def transfer_loss(inputs, combination_image,
                  total_variation_weight=1e-4,
                  style_weight=1.,
                  content_weight=0.0025):
    """
    总的损失

    :param inputs:
    :param combination_image:
    :param total_variation_weight:
    :param style_weight:
    :param content_weight:
    :return:
    """
    model = vgg19.VGG19(input_tensor=inputs, weights='imagenet', include_top=False)
    print('Model loaded.')
    output_dict = dict([(layer.name, layer.output) for layer in model.layers])
    # 损失
    loss = K.variable(0.)
    # 内容损失
    content_layer = 'block5_conv2'
    layer_features = output_dict[content_layer]
    target_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(target_image_features, combination_features)
    # 风格损失
    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1',
    ]
    for layer_name in style_layers:
        layer_features = output_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        s1 = style_loss(style_reference_features, combination_features)
        loss = loss + (style_weight / len(style_layers)) * s1
    # 总变差损失
    loss = loss + total_variation_weight * total_variation_loss(combination_image)
    return loss


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_value = None

    def fg(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        self.loss_value = outs[0]
        self.grads_value = outs[1].flatten().astype(np.float64)
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grads_value = np.copy(self.grads_value)
        self.loss_value = None
        self.grads_value = None
        return grads_value


if __name__ == '__main__':
    # 目标图像，样式图像，合成图像
    target_image = K.constant(preprocess_image(target_image_path, (img_height, img_width)))
    style_reference_image = K.constant(preprocess_image(style_reference_image_path, (img_height, img_width)))
    combination_image = K.placeholder((1, img_height, img_width, 3))
    # 输入
    input_tensor = K.concatenate([target_image, style_reference_image, combination_image], axis=0)
    # 损失
    loss = transfer_loss(input_tensor, combination_image)
    # 获取损失相对于生成图像的梯度
    grads = K.gradients(loss, combination_image)[0]
    fetch_loss_and_grads = K.function([combination_image], [loss, grads])

    evaluator = Evaluator()

    result_prefix = 'my_result'
    iterations = 10

    x = preprocess_image(target_image_path, size=(img_height, img_width))
    x = x.flatten()

    for i in range(iterations):
        print('Start of iteration: ', i)
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.fg, x, fprime=evaluator.grads, maxfun=20)
        print('Current loss value: ', min_val)
        img = x.copy().reshape((img_height, img_width, 3))
        img = deprocess_image(img)
        fname = result_prefix + '_iter_%d.jpg' % i
        imsave(fname, img)
        print('Image saved as: ', fname)
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))
