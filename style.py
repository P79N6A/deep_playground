#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import keras
import keras.backend as K
from keras.layers import Input, AveragePooling2D, Conv2D
from keras.models import Model
import numpy as np
from PIL import Image

IMAGENET_MEANS = np.array([103.939, 116.779, 123.68])

def content_loss(content, combination):
    return 0.5 * K.sum(K.square(combination - content))

def gram(x):
    shape = K.shape(x)
    F = K.reshape(x, (shape[0] * shape[1], shape[2]))
    return K.dot(K.transpose(F), F)

def style_loss(style, combination):
    Ml2 = int(style.shape[0] * style.shape[1])**2
    N12 = int(style.shape[2])**2
    return K.sum(K.square(gram(style) - gram(combination))) / (4. * N12 * Ml2)

def total_variation_loss(x, kind='isotropic'):
    h, w = x.shape[1], x.shape[2]
    if kind == 'anisotropic':
        a = K.abs(x[:, :h-1, :w-1, :] - x[:, 1:, :w-1, :])
        b = K.abs(x[:, :h-1, :w-1, :] - x[:, :h-1, 1:, :])
        return K.sum(a + b)
    elif kind == 'isotropic':
        a = K.square(x[:, :h-1, :w-1, :] - x[:, 1:, :w-1, :])
        b = K.square(x[:, :h-1, :w-1, :] - x[:, :h-1, 1:, :])
        return K.sum(K.pow(a + b, 2))
    else:
        raise ValueError("`kind` should be 'anisotropic' or 'isotropic'")

def load_image(image_path, size=None):
    image = Image.open(image_path)
    if size is not None:
        image = image.resize(size)
    return image

def image_to_matrix(image, dtype=np.float32):
    image = np.asarray(image, dtype=dtype)
    image = np.expand_dims(image, axis=0)
    return image

def matrix_to_image(image, channel_range=(0, 255)):
    image = np.clip(image, *channel_range).astype('uint8')
    return Image.fromarray(image)

def normalize(image, inplace=False):
    assert image.shape[3] == IMAGENET_MEANS.shape[0]
    if not inplace:
        image = image.copy()
    image = image[:, :, :, ::-1]
    image[:, :, :, np.arange(IMAGENET_MEANS.shape[0])] -= IMAGENET_MEANS
    return image

def denormalize(image, inplace=False):
    assert image.shape[2] == IMAGENET_MEANS.shape[0]
    if not inplace:
        image = image.copy()
    image[:, :, np.arange(IMAGENET_MEANS.shape[0])] += IMAGENET_MEANS
    image = image[:, :, ::-1]
    return image

content_path = '/data1/ceph/yzchen/img/golden_gate.jpg'
style_path = '/data1/ceph/yzchen/img/the-starry-night.jpg'

content = load_image(content_path)
image_size = content.size
style = load_image(style_path, image_size)

content = image_to_matrix(content)
style = image_to_matrix(style)
content = normalize(content)
style = normalize(style)

content_tensor = K.constant(content, name='Content')
style_tensor = K.constant(style, name='Style')
canvas = K.placeholder(content.shape, name='Canvas')
tensor = K.concatenate([content_tensor, style_tensor, canvas], axis=0)

input_tensor = Input(tensor=tensor, shape=(None, None, 3))
Pool2D = AveragePooling2D
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = Pool2D((2, 2), strides=(2, 2), name='block1_pool')(x)
# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = Pool2D((2, 2), strides=(2, 2), name='block2_pool')(x)
# Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
x = Pool2D((2, 2), strides=(2, 2), name='block3_pool')(x)
# Block 4
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
x = Pool2D((2, 2), strides=(2, 2), name='block4_pool')(x)
# Block 5
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
x = Pool2D((2, 2), strides=(2, 2), name='block5_pool')(x)

model = Model(input_tensor, x)
model.load_weights('/data1/ceph/yzchen/models/vgg19_for_style_weights.h5')

outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
content_layer = 'block5_conv2'
style_layers = ['block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1']

total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025

loss = K.variable(0.0, name='Loss')
layer_features = outputs_dict[content_layer]
content_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(content_features, combination_features)

for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl

loss += total_variation_weight * total_variation_loss(canvas)

grads = K.gradients(loss, canvas)[0]
fetch_loss_and_grads = K.function([canvas], [loss, grads])

initial = content.copy()
learning_rate = 1e-4
loss_history = []
iterations = 1000

for i in range(iterations):
    print(i)
    loss_i, grads_i = fetch_loss_and_grads([initial])
    initial -= learning_rate * grads_i
    loss_history.append(loss_i)
image = initial
K.clear_session()
image = image.reshape(canvas.shape)[0]
image = denormalize(image)
image = matrix_to_image(image)

image.save('/data1/ceph/yzchen/img/transfer.png')

