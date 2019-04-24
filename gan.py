#!/data/miniconda3/bin/python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, Input, preprocessing, applications
print(tf.__version__)
import numpy as np
print(np.__version__)
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

ceph_dir = '/data1/ceph/yzchen'

latent_dim = 32
heigth = 32
width = 32
channels = 3

generator_input = Input(shape=(latent_dim,))
x = layers.Dense(128*16*16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

generator_output = layers.Conv2D(channels, 7 , activation='tanh', padding='same')(x)
generator = models.Model(generator_input, generator_output)


discriminator_input = Input(shape=(heigth,width,channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

x = layers.Dropout(0.4)(x)

discriminator_output = layers.Dense(1, activation='sigmoid')(x)
discriminator = models.Model(discriminator_input, discriminator_output)
discriminator.summary()

discriminator.compile(
    optimizer=keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8),
    loss='binary_crossentropy'
)

discriminator.trainable = False

gan_input = Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = models.Model(gan_input, gan_output)

gan.compile(
    optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8),
    loss = 'binary_crossentropy'
)

if __name__ == '__main__':
    raw_data = unpickle(os.path.join(ceph_dir, 'cf10/data_batch_1'))
    x_train = np.array(raw_data[b'data'])
    y_train = np.array(raw_data[b'labels'])
    for i in range(2,6):
        raw_data = unpickle(os.path.join(ceph_dir, 'cf10/data_batch_'+str(i)))
        x_train = np.concatenate([x_train, np.array(raw_data[b'data'])])
        y_train = np.concatenate([y_train, np.array(raw_data[b'labels'])])

    x_train = x_train[y_train==4]
    x_train = x_train.reshape(x_train.shape[0], channels, heigth, width).transpose([0,2,3,1]).astype('float32') / 255.

    print('x_train shape: ', x_train.shape)

    iterations = 10000
    batch_size = 20

    start = 0
    for step in range(iterations):
        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
        generated_images = generator.predict(random_latent_vectors)
        
        stop = start + batch_size
        real_images = x_train[start:stop]
        combined_images = np.concatenate([generated_images, real_images])
        labels = np.concatenate([
            np.ones((batch_size, 1)), 
            np.zeros((batch_size, 1))])
        labels += 0.05 * np.random.random(labels.shape)
        d_loss = discriminator.train_on_batch(combined_images, labels)
        
        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
        misleading_targets = np.zeros((batch_size, 1))
        a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
        
        start += batch_size
        if start > len(x_train) - batch_size:
            start = 0
        
        if step % 100 == 0:
            gan.save_weights(os.path.join(ceph_dir, 'models/gan.h5'))
            print('discriminator loss: ', d_loss)
            print('adversarial loss: ', a_loss)
            
            img = keras.preprocessing.image.array_to_img(generated_images[0] * 255.,scale=False)
            img.save(os.path.join(ceph_dir, 'gan/generated_img' + str(step) + '.png'))
            
            img = keras.preprocessing.image.array_to_img(real_images[0] * 255.,scale=False)
            img.save(os.path.join(ceph_dir, 'gan/real_img' + str(step) + '.png'))