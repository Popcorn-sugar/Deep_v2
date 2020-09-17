
'''
Utilities for DeepLab

Lei Mao
Department of Computer Science
University of Chicago

dukeleimao@gmail.com
'''

import tensorflow as tf


def atrous_spatial_pyramid_pooling(inputs, filters=256, regularizer=None):
    '''
    Atrous Spatial Pyramid Pooling (ASPP) Block
    '''

    pool_height = tf.shape(input=inputs)[1]
    pool_width = tf.shape(input=inputs)[2]

    resize_height = pool_height
    resize_width = pool_width

    # Atrous Spatial Pyramid Pooling
    # Atrous 1x1
    aspp1x1 = tf.compat.v1.layers.conv2d(inputs=inputs, filters=filters, kernel_size=(1, 1), padding='same', kernel_regularizer=regularizer, name='aspp1x1')
    # Atrous 3x3, rate = 6
    aspp3x3_1 = tf.compat.v1.layers.conv2d(inputs=inputs, filters=filters, kernel_size=(3, 3), padding='same', dilation_rate=(6, 6), kernel_regularizer=regularizer, name='aspp3x3_1')
    # Atrous 3x3, rate = 12
    aspp3x3_2 = tf.compat.v1.layers.conv2d(inputs=inputs, filters=filters, kernel_size=(3, 3), padding='same', dilation_rate=(12, 12), kernel_regularizer=regularizer, name='aspp3x3_2')
    # Atrous 3x3, rate = 18
    aspp3x3_3 = tf.compat.v1.layers.conv2d(inputs=inputs, filters=filters, kernel_size=(3, 3), padding='same', dilation_rate=(18, 18), kernel_regularizer=regularizer, name='aspp3x3_3')

    # Image Level Pooling
    image_feature = tf.reduce_mean(input_tensor=inputs, axis=[1, 2], keepdims=True)
    image_feature = tf.compat.v1.layers.conv2d(inputs=image_feature, filters=filters, kernel_size=(1, 1), padding='same')
    image_feature = tf.image.resize(images=image_feature, size=[resize_height, resize_width], method=tf.image.ResizeMethod.BILINEAR, name='image_pool_feature')

    # Merge Poolings
    outputs = tf.concat(values=[aspp1x1, aspp3x3_1, aspp3x3_2, aspp3x3_3, image_feature], axis=3, name='aspp_pools')
    outputs = tf.compat.v1.layers.conv2d(inputs=outputs, filters=filters, kernel_size=(1, 1), padding='same', kernel_regularizer=regularizer, name='aspp_outputs')

    return outputs
