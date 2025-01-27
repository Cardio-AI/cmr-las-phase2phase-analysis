import tensorflow as tf

from tensorflow import keras
import keras.layers as kl
from keras import backend as K
from keras.layers import Dropout, BatchNormalization, Activation
from keras.layers import Layer

from keras.layers import UpSampling2D as UpSampling2DInterpol
from keras.layers import UpSampling3D
from tensorflow.python.keras.utils import conv_utils
import numpy as np
import sys

__all__ = ['UpSampling2DInterpol', 'UpSampling3DInterpol', 'ConvEncoder', 'conv_layer_fn',
           'downsampling_block_fn', 'upsampling_block_fn',  'ConvDecoder', 'ConvEncoder',
           'get_centers_tf',  'get_angle_tf',  'ConvBlock', 'get_idxs_tf']


class UpSampling3DInterpol(UpSampling3D):

    def __init__(self, size=(1, 2, 2), interpolation='bilinear', **kwargs):
        self.size = conv_utils.normalize_tuple(size, 3, 'size')
        self.x = int(size[1])
        self.y = int(size[2])
        self.interpolation = interpolation
        super(self.__class__, self).__init__(**kwargs)

    def call(self, x):
        """
        :param x:
        :return:
        """
        target_size = (x.shape[2] * self.x, x.shape[3] * self.y)
        # traverse along the 3D volumes, handle the z-slices as batch
        return K.stack(
            tf.map_fn(lambda images:
                      tf.image.resize(
                          images=images,
                          size=target_size,
                          method=self.interpolation,  # define bilinear or nearest neighbor
                          preserve_aspect_ratio=True),
                      x))

    def get_config(self):
        config = super(UpSampling3DInterpol, self).get_config()
        config.update({'interpolation': self.interpolation, 'size': self.size})
        return config

class ConvEncoder(Layer):
    def __init__(self, activation, batch_norm, bn_first, depth, drop_3, dropouts, f_size, filters,
                 kernel_init, m_pool, ndims, pad):
        """
        Convolutional encoder for 2D or 3D input images/volumes.
        The architecture is aligned to the downsampling part of a U-Net
        Parameters
        ----------
        activation :
        batch_norm :
        bn_first :
        depth :
        drop_3 :
        dropouts :
        f_size :
        filters :
        kernel_init :
        m_pool :
        ndims :
        pad :
        """
        super(self.__class__, self).__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        self.bn_first = bn_first
        self.depth = depth
        self.drop_3 = drop_3
        self.dropouts = dropouts
        self.f_size = f_size
        self.filters = filters
        self.kernel_init = kernel_init
        self.m_pool = m_pool
        self.ndims = ndims
        self.pad = pad

        self.downsamplings = []
        filters = self.filters

        for l in range(self.depth):
            db = DownSampleBlock(filters=filters,
                                 f_size=self.f_size,
                                 activation=self.activation,
                                 drop=self.dropouts[l],
                                 batch_norm=self.batch_norm,
                                 kernel_init=self.kernel_init,
                                 pad=self.pad,
                                 m_pool=self.m_pool,
                                 bn_first=self.bn_first,
                                 ndims=self.ndims)
            self.downsamplings.append(db)
            filters *= 2

        self.conv1 = ConvBlock(filters=filters, f_size=self.f_size,
                               activation=self.activation, batch_norm=self.batch_norm, kernel_init=self.kernel_init,
                               pad=self.pad, bn_first=self.bn_first, ndims=self.ndims)

        self.bn = Dropout(self.drop_3)
        self.conv2 = ConvBlock(filters=filters, f_size=self.f_size,
                               activation=self.activation, batch_norm=self.batch_norm, kernel_init=self.kernel_init,
                               pad=self.pad, bn_first=self.bn_first, ndims=self.ndims)

    def call(self, inputs, **kwargs):

        encs = []
        skips = []

        self.first_block = True

        for db in self.downsamplings:

            if self.first_block:
                # first block
                input_tensor = inputs
                self.first_block = False
            else:
                # all other blocks, use the max-pooled output of the previous encoder block as input
                # remember the max-pooled output from the previous layer
                input_tensor = encs[-1]

            skip, enc = db(input_tensor)
            encs.append(enc)
            skips.append(skip)

        # return the last encoding block result
        encoding = encs[-1]
        encoding = self.conv1(inputs=encoding)
        encoding = self.bn(encoding)
        encoding = self.conv2(inputs=encoding)

        # work as u-net encoder or cnn encoder
        return encoding, skips

    def get_config(self):
        """ __init__() is overwritten, need to override this method to enable model.to_json() for this layer"""

        config = super(self.__class__, self).get_config()
        config.update({'activation': self.activation,
                       'batch_norm': self.batch_norm,
                       'bn_first': self.bn_first,
                       'depth': self.depth,
                       'drop_3': self.drop_3,
                       'dropouts': self.dropouts,
                       'f_size': self.f_size,
                       'filters': self.filters,
                       'kernel_init': self.kernel_init,
                       'm_pool': self.m_pool,
                       'ndims': self.ndims,
                       'pad': self.pad})
        return config


class ConvDecoder(Layer):
    def __init__(self, activation, batch_norm, bn_first, depth, drop_3, dropouts, f_size, filters,
                 kernel_init, up_size, ndims, pad):
        """
        Convolutional Decoder path, could be used in combination with the encoder layer,
        or as up-scaling path for super resolution etc.
        Parameters
        ----------
        activation :
        batch_norm :
        bn_first :
        depth :
        drop_3 :
        dropouts :
        f_size :
        filters :
        kernel_init :
        up_size :
        ndims :
        pad :
        """
        super(self.__class__, self).__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        self.bn_first = bn_first
        self.depth = depth
        self.drop_3 = drop_3
        self.dropouts = dropouts
        self.f_size = f_size
        self.filters = filters
        self.kernel_init = kernel_init
        self.up_size = up_size
        self.ndims = ndims
        self.pad = pad
        self.upsamplings = []

        filters = self.filters
        for layer in range(self.depth):
            ub = UpSampleBlock(filters=filters,
                               f_size=self.f_size,
                               activation=self.activation,
                               drop=self.dropouts[layer],
                               batch_norm=self.batch_norm,
                               kernel_init=self.kernel_init,
                               pad=self.pad,
                               up_size=self.up_size,
                               bn_first=self.bn_first,
                               ndims=self.ndims)
            self.upsamplings.append(ub)
            filters /= 2

    def call(self, inputs, **kwargs):

        enc, skips = inputs

        for upsampling in self.upsamplings:
            skip = skips.pop()
            enc = upsampling([enc, skip])

        return enc

    def get_config(self):
        """ __init__() is overwritten, need to override this method to enable model.to_json() for this layer"""

        config = super(self.__class__, self).get_config()
        config.update({'activation': self.activation,
                       'batch_norm': self.batch_norm,
                       'bn_first': self.bn_first,
                       'depth': self.depth,
                       'drop_3': self.drop_3,
                       'dropouts': self.dropouts,
                       'f_size': self.f_size,
                       'filters': self.filters,
                       'kernel_init': self.kernel_init,
                       'up_size': self.up_size,
                       'ndims': self.ndims,
                       'pad': self.pad})
        return config


class ConvBlock(Layer):
    def __init__(self, filters=16, f_size=(3, 3, 3), activation='elu', batch_norm=True, kernel_init='he_normal',
                 pad='same', bn_first=False, ndims=2, strides=1):
        """
        Wrapper for a 2/3D-conv layer + batchnormalisation
        Either with Conv,BN,activation or Conv,activation,BN

        :param filters: int, number of filters
        :param f_size: tuple of int, filterzise per axis
        :param activation: string, which activation function should be used
        :param batch_norm: bool, use batch norm or not
        :param kernel_init: string, keras enums for kernel initialisation
        :param pad: string, keras enum how to pad, the conv
        :param bn_first: bool, decide if we want to apply the BN before the conv. operation or afterwards
        :param ndims: int, define the conv dimension
        :param strides: int, stride of the conv filter
        :return: a functional tf.keras conv block
        expects an numpy or tensor object with (batchsize,z,x,y,channels)
        """
        super(self.__class__, self).__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        self.bn_first = bn_first
        self.f_size = f_size
        self.filters = filters
        self.kernel_init = kernel_init
        self.ndims = ndims
        self.pad = pad
        self.strides = strides
        self.encoder = list()

        # create the layers
        Conv = getattr(kl, 'Conv{}D'.format(self.ndims))
        f_size = self.f_size[:self.ndims]

        self.conv = Conv(filters=self.filters, kernel_size=f_size, kernel_initializer=self.kernel_init,
                         padding=self.pad, strides=self.strides)
        self.conv_activation = Conv(filters=self.filters, kernel_size=f_size, kernel_initializer=self.kernel_init,
                                    padding=self.pad, strides=self.strides, activation=activation)
        self.bn = BatchNormalization(axis=-1)
        self.activation = Activation(self.activation)

    def call(self, inputs, **kwargs):

        if self.bn_first:
            # , kernel_regularizer=regularizers.l2(0.0001)
            conv1 = self.conv(inputs)
            conv1 = self.bn(conv1) if self.batch_norm else conv1
            conv1 = self.activation(conv1)

        else:
            # kernel_regularizer=regularizers.l2(0.0001),
            conv1 = self.conv_activation(inputs)
            conv1 = self.bn(conv1) if self.batch_norm else conv1

        return conv1

    def get_config(self):
        config = super(self.__class__, self).get_config()
        config.update({'activation': self.activation,
                       'batch_norm': self.batch_norm,
                       'bn_first': self.bn_first,
                       'f_size': self.f_size,
                       'filters': self.filters,
                       'kernel_init': self.kernel_init,
                       'ndims': self.ndims,
                       'pad': self.pad})
        return config


class DownSampleBlock(Layer):
    def __init__(self, filters=16, f_size=(3, 3, 3), activation='elu', drop=0.3, batch_norm=True,
                 kernel_init='he_normal', pad='same', m_pool=(2, 2), bn_first=False, ndims=2):
        """
    Create an 2D/3D-downsampling block for the u-net architecture
    :param filters: int, number of filters per conv-layer
    :param f_size: tuple of int, filtersize per axis
    :param activation: string, which activation function should be used
    :param drop: float, define the dropout rate between the conv layers of this block
    :param batch_norm: bool, use batch norm or not
    :param kernel_init: string, keras enums for kernel initialisation
    :param pad: string, keras enum how to pad, the conv
    :param m_pool: tuple of int, size of the max-pooling filters
    :param bn_first: bool, decide if we want to apply the BN before the conv. operation or afterwards
    :param ndims: int, define the conv dimension
    :return: a functional tf.keras upsampling block
    Excpects a numpy or tensor input with (batchsize,z,x,y,channels)
    """

        super(self.__class__, self).__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        self.bn_first = bn_first
        self.drop = drop
        self.f_size = f_size
        self.filters = filters
        self.kernel_init = kernel_init
        self.m_pool = m_pool
        self.ndims = ndims
        self.pad = pad
        self.encoder = list()

        self.m_pool = self.m_pool[-self.ndims:]
        self.pool_fn = getattr(kl, 'MaxPooling{}D'.format(self.ndims))
        self.pool = self.pool_fn(m_pool)
        self.conf1 = ConvBlock(filters=self.filters, f_size=self.f_size, activation=self.activation,
                               batch_norm=self.batch_norm,
                               kernel_init=self.kernel_init, pad=self.pad, bn_first=self.bn_first, ndims=self.ndims)
        self.dropout = Dropout(self.drop)
        self.conf2 = ConvBlock(filters=self.filters, f_size=self.f_size, activation=self.activation,
                               batch_norm=self.batch_norm,
                               kernel_init=self.kernel_init, pad=self.pad, bn_first=self.bn_first, ndims=self.ndims)

    def call(self, x, **kwargs):
        x = self.conf1(x)
        x = self.dropout(x)
        conv1 = self.conf2(x)
        p1 = self.pool(conv1)

        return [conv1, p1]


class UpSampleBlock(Layer):
    def __init__(self, use_upsample=True, filters=16, f_size=(3, 3, 3), activation='elu',
                 drop=0.3, batch_norm=True, kernel_init='he_normal', pad='same', up_size=(2, 2), bn_first=False,
                 ndims=2):
        """
        Create an upsampling block for the u-net architecture
        Each blocks consists of these layers: upsampling/transpose,concat,conv,dropout,conv
        Either with "upsampling,conv" or "transpose" upsampling
        :param use_upsample: bool, whether to use upsampling or transpose layer
        :param filters: int, number of filters per conv-layer
        :param f_size: tuple of int, filter size per axis
        :param activation: string, which activation function should be used
        :param drop: float, define the dropout rate between the conv layers of this block
        :param batch_norm: bool, use batch norm or not
        :param kernel_init: string, keras enums for kernel initialisation
        :param pad: string, keras enum how to pad, the conv
        :param up_size: tuple of int, size of the upsampling filters, either by transpose layers or upsampling layers
        :param bn_first: bool, decide if we want to apply the BN before the conv. operation or afterwards
        :param ndims: int, define the conv dimension
        :return: a functional tf.keras upsampling block
        Expects an input with length 2 lower block: batchsize,z,x,y,channels, skip layers: batchsize,z,x,y,channels
        """

        super(self.__class__, self).__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        self.bn_first = bn_first
        self.drop = drop
        self.f_size = f_size
        self.filters = filters
        self.kernel_init = kernel_init
        self.use_upsample = use_upsample
        self.up_size = up_size
        self.ndims = ndims
        self.pad = pad
        self.encoder = list()

        Conv = getattr(kl, 'Conv{}D'.format(self.ndims))
        UpSampling = getattr(kl, 'UpSampling{}D'.format(self.ndims))
        ConvTranspose = getattr(kl, 'Conv{}DTranspose'.format(self.ndims))

        f_size = self.f_size[-self.ndims:]

        # use upsample&conv or transpose layer
        self.upsample = UpSampling(size=self.up_size)
        self.conv1 = Conv(filters=self.filters, kernel_size=f_size, padding=self.pad,
                          kernel_initializer=self.kernel_init,
                          activation=self.activation)

        self.convTranspose = ConvTranspose(filters=self.filters, kernel_size=f_size, strides=self.up_size,
                                           padding=self.pad,
                                           kernel_initializer=self.kernel_init,
                                           activation=self.activation)

        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        self.convBlock1 = ConvBlock(filters=self.filters, f_size=f_size, activation=self.activation,
                                    batch_norm=self.batch_norm,
                                    kernel_init=self.kernel_init, pad=self.pad, bn_first=self.bn_first,
                                    ndims=self.ndims)
        self.dropout = Dropout(self.drop)
        self.convBlock2 = ConvBlock(filters=self.filters, f_size=f_size, activation=self.activation,
                                    batch_norm=self.batch_norm,
                                    kernel_init=self.kernel_init, pad=self.pad, bn_first=self.bn_first,
                                    ndims=self.ndims)

    def call(self, inputs, **kwargs):

        if len(inputs) == 2:
            skip = True
            lower_input, conv_input = inputs
        else:
            skip = False
            lower_input = inputs

        # use upsample&conv or transpose layer
        if self.use_upsample:

            deconv1 = self.upsample(lower_input)
            deconv1 = self.conv1(deconv1)

        else:
            deconv1 = self.convTranspose(lower_input)

        # if skip given, concat
        if skip:
            deconv1 = self.concatenate([deconv1, conv_input])
        deconv1 = self.convBlock1(inputs=deconv1)
        deconv1 = self.dropout(deconv1)
        deconv1 = self.convBlock2(inputs=deconv1)

        return deconv1

    def get_config(self):
        """ __init__() is overwritten, need to override this method to enable model.to_json() for this layer"""
        config = super(self.__class__, self).get_config()
        config.update({'activation': self.activation,
                       'batch_norm': self.batch_norm,
                       'bn_first': self.bn_first,
                       'drop': self.drop,
                       'f_size': self.f_size,
                       'filters': self.filters,
                       'kernel_init': self.kernel_init,
                       'm_pool': self.m_pool,
                       'ndims': self.ndims,
                       'pad': self.pad})
        return config

#
def conv_layer_fn(inputs, filters=16, f_size=(3, 3, 3), activation='elu', batch_norm=True, kernel_init='he_normal',
                  pad='same', bn_first=False, ndims=2, custom_name=''):
    """
    Wrapper for a 2/3D-conv layer + batchnormalisation
    Either with Conv,BN,activation or Conv,activation,BN

    :param inputs: numpy or tensor object batchsize,z,x,y,channels
    :param filters: int, number of filters
    :param f_size: tuple of int, filterzise per axis
    :param activation: string, which activation function should be used
    :param batch_norm: bool, use batch norm or not
    :param kernel_init: string, keras enums for kernel initialisation
    :param pad: string, keras enum how to pad, the conv
    :param bn_first: bool, decide if we want to apply the BN before the conv. operation or afterwards
    :param ndims: int, define the conv dimension
    :return: a functional tf.keras conv block
    """

    Conv = getattr(kl, 'Conv{}D'.format(ndims))
    f_size = f_size[:ndims]

    if bn_first:
        # , kernel_regularizer=regularizers.l2(0.0001)
        conv1 = Conv(filters=filters, kernel_size=f_size, kernel_initializer=kernel_init, padding=pad)(inputs)
        conv1 = BatchNormalization(axis=-1)(conv1) if batch_norm else conv1
        conv1 = Activation(activation)(conv1)

    else:
        # kernel_regularizer=regularizers.l2(0.0001),
        conv1 = Conv(filters=filters, kernel_size=f_size, activation=activation, kernel_initializer=kernel_init,
                     padding=pad)(inputs)
        conv1 = BatchNormalization(axis=-1)(conv1) if batch_norm else conv1

    return conv1

#
def downsampling_block_fn(inputs, filters=16, f_size=(3, 3, 3), activation='elu', drop=0.3, batch_norm=True,
                          kernel_init='he_normal', pad='same', m_pool=(2, 2), bn_first=False, ndims=2):
    """
    Create an 2D/3D-downsampling block for the u-net architecture
    :param inputs: numpy or tensor input with batchsize,z,x,y,channels
    :param filters: int, number of filters per conv-layer
    :param f_size: tuple of int, filtersize per axis
    :param activation: string, which activation function should be used
    :param drop: float, define the dropout rate between the conv layers of this block
    :param batch_norm: bool, use batch norm or not
    :param kernel_init: string, keras enums for kernel initialisation
    :param pad: string, keras enum how to pad, the conv
    :param m_pool: tuple of int, size of the max-pooling layer
    :param bn_first: bool, decide if we want to apply the BN before the conv. operation or afterwards
    :param ndims: int, define the conv dimension
    :return: a functional tf.keras downsampling block
    """
    m_pool = m_pool[-ndims:]
    pool = getattr(kl, 'MaxPooling{}D'.format(ndims))

    conv1 = conv_layer_fn(inputs=inputs, filters=filters, f_size=f_size, activation=activation, batch_norm=batch_norm,
                          kernel_init=kernel_init, pad=pad, bn_first=bn_first, ndims=ndims)
    conv1 = Dropout(drop)(conv1)
    conv1 = conv_layer_fn(inputs=conv1, filters=filters, f_size=f_size, activation=activation, batch_norm=batch_norm,
                          kernel_init=kernel_init, pad=pad, bn_first=bn_first, ndims=ndims)

    p1 = pool(m_pool, padding=pad)(conv1)

    return (conv1, p1)

#
def upsampling_block_fn(lower_input, conv_input, use_upsample=True, filters=16, f_size=(3, 3, 3), activation='elu',
                        drop=0.3, batch_norm=True, kernel_init='he_normal', pad='same', up_size=(2, 2), bn_first=False,
                        ndims=2):
    """
    Create an upsampling block for the u-net architecture
    Each blocks consists of these layers: upsampling/transpose,concat,conv,dropout,conv
    Either with "upsampling,conv" or "transpose" upsampling
    :param lower_input: numpy input from the lower block: batchsize,z,x,y,channels
    :param conv_input: numpy input from the skip layers: batchsize,z,x,y,channels
    :param use_upsample: bool, whether to use upsampling or not
    :param filters: int, number of filters per conv-layer
    :param f_size: tuple of int, filtersize per axis
    :param activation: string, which activation function should be used
    :param drop: float, define the dropout rate between the conv layers of this block
    :param batch_norm: bool, use batch norm or not
    :param kernel_init: string, keras enums for kernel initialisation
    :param pad: string, keras enum how to pad, the conv
    :param up_size: tuple of int, size of the upsampling filters, either by transpose layers or upsampling layers
    :param bn_first: bool, decide if we want to apply the BN before the conv. operation or afterwards
    :param ndims: int, define the conv dimension
    :return: a functional tf.keras upsampling block
    """

    Conv = getattr(kl, 'Conv{}D'.format(ndims))
    f_size = f_size[-ndims:]

    # use upsample&conv or transpose layer
    if use_upsample:
        # import src.models.KerasLayers as ownkl
        # UpSampling = getattr(ownkl, 'UpSampling{}DInterpol'.format(ndims))
        UpSampling = getattr(kl, 'UpSampling{}D'.format(ndims))
        deconv1 = UpSampling(size=up_size)(lower_input)
        deconv1 = Conv(filters=filters, kernel_size=f_size, padding=pad, kernel_initializer=kernel_init,
                       activation=activation)(deconv1)

    else:
        ConvTranspose = getattr(kl, 'Conv{}DTranspose'.format(ndims))
        deconv1 = ConvTranspose(filters=filters, kernel_size=f_size, strides=up_size, padding=pad,
                                kernel_initializer=kernel_init,
                                activation=activation)(lower_input)

    deconv1 = tf.keras.layers.Concatenate(axis=-1)([deconv1, conv_input])

    deconv1 = conv_layer_fn(inputs=deconv1, filters=filters, f_size=f_size, activation=activation,
                            batch_norm=batch_norm,
                            kernel_init=kernel_init, pad=pad, bn_first=bn_first, ndims=ndims)
    deconv1 = Dropout(drop)(deconv1)
    deconv1 = conv_layer_fn(inputs=deconv1, filters=filters, f_size=f_size, activation=activation,
                            batch_norm=batch_norm,
                            kernel_init=kernel_init, pad=pad, bn_first=bn_first, ndims=ndims)

    return deconv1


flow2direction_lambda = tf.keras.layers.Lambda(
    lambda x: get_angle_tf(x[0], x[1]), name='flow2direction')
minmax_lambda = lambda x: x[1] + (
            ((x[0] - np.min(x[0])) * (x[2] - x[1])) / (np.max(x[0]) - np.min(x[0]) + sys.float_info.epsilon))


def get_focus_tf(p, dim=[12, 12, 12]):
    return tf.cast(
        tf.tile(tf.convert_to_tensor([*p])[tf.newaxis, tf.newaxis, ...],
                (*dim, 1)), tf.float32)


# returns a matrix with the indicies as values, similar to np.indicies
def get_idxs_tf(x):
    return tf.cast(
        tf.reshape(tf.where(tf.ones((x[0], x[1]))), (x[0], x[1], 2)),
        tf.float32)


# returns a matrix with vectors pointing to the center
def get_centers_tf(x):
    return tf.cast(
        tf.tile(tf.convert_to_tensor([x[0] // 2, x[1] // 2, x[2] // 2])[tf.newaxis, tf.newaxis, tf.newaxis, ...],
                (x[0], x[1], x[2], 1))
        if len(x) == 3 else tf.tile(tf.convert_to_tensor([x[0] // 2, x[1] // 2])[tf.newaxis, tf.newaxis, ...],
                                    (x[0], x[1], 1)), tf.float32)


def get_angle_tf(a, b, indegree=False):
    """
    this should work for batches of n-dimensional vectors
    α = arccos[(a · b) / (|a| * |b|)]
    |v| = √(x² + y² + z²)
    in 3D space
    If vectors a = [xa, ya, za], b = [xb, yb, zb], then:
    α = arccos[(xa * xb + ya * yb + za * zb) / (√(xa2 + ya2 + za2) * √(xb2 + yb2 + zb2))]

    Args:
        a (tf.tensor): b,z,y,x,3
        b (tf.tensor): b,z,y,x,3
        indegree (bool): other-wise calc cos(angle)<- this is differentiable

    Returns: tf.tensor with the same shape except of the last axis

    """
    import math as m
    pi = tf.constant(m.pi)
    b = tf.cast(b, dtype=a.dtype)
    inner = tf.einsum('...i,...i->...', a, b)
    norms = tf.norm(a, ord='euclidean', axis=-1) * tf.norm(b, ord='euclidean', axis=-1)  # [...,None]
    cos = inner / (norms + sys.float_info.epsilon)
    cos = tf.clip_by_value(cos, -1.0, 1.0)
    if indegree:
        rad = tf.math.acos(tf.clip_by_value(cos, -1.0, 1.0))  # need to check if this is necessary
        # rad2deg conversion
        deg = rad * (180.0 / pi)
        cos = deg
    return cos[..., tf.newaxis]