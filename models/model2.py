from tensorflow import keras
import tensorflow as tf
from keras.layers import Conv2D, Dense, Concatenate, Reshape, ReLU, Conv1D, LeakyReLU, Lambda
from keras.layers import Input, BatchNormalization, MaxPool2D, GlobalAveragePooling2D, Flatten, Dropout
from keras.models import Sequential, Model
import keras.backend as k
import tensorflow_addons as tfa
from models.model import UNet


class ContrastiveLoss(keras.losses.Loss):
    def __init__(self, margin=1, **kwargs):
        super(ContrastiveLoss, self).__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        squared_pred = k.square(y_pred)
        squared_margin = k.square(k.maximum(self.margin - y_pred, 0))
        loss = k.mean(y_true * squared_pred + (1 - y_true) * squared_margin)
        return loss

    def get_config(self):
        base_config = super(ContrastiveLoss, self).get_config()
        return {**base_config, "margin": self.margin}


def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = k.sum(k.square(featsA - featsB), axis=1,
                       keepdims=True)
    # return the euclidean distance between the vectors
    return k.sqrt(k.maximum(sumSquared, k.epsilon()))


def conv_blocK(filters):
    return Sequential([
        Conv2D(filters=filters, kernel_size=3, strides=2),
        BatchNormalization(),
        LeakyReLU(0.2)
    ])


def mbb_model(inputs=Input((512, 128))):
    # unet = UNet(inputs=inputs)
    # x = unet(inputs)
    # x = Reshape((512, 128))(x)
    x = MBBlock(bands=4)(inputs)
    x = MBBlock(bands=8)(x)
    x = MBBlock(bands=16)(x)
    x = MBBlock(bands=32)(x)
    x = MBBlock(bands=64)(x)
    # x = keras.layers.Activation('tanh')(x)
    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model


class MBBlock(keras.layers.Layer):
    def __init__(self, bands=4, initial_bands=128, axis=2, kernel_size=3, strides=1,
                 padding="SAME", **kwargs):
        super(MBBlock, self).__init__(**kwargs)
        self.bands = bands
        self.kernel_size = kernel_size
        self.filter_size = initial_bands // self.bands
        self.axis = axis
        self.padding = padding
        self.strides = strides

        self.layers = [

            Conv1D(filters=self.filter_size, kernel_size=self.kernel_size, strides=self.strides,
                   padding=self.padding, kernel_initializer="he_normal"),

            # BatchNormalization(),
            tfa.layers.InstanceNormalization(),

            LeakyReLU(0.2),

            Conv1D(filters=self.filter_size, kernel_size=self.kernel_size, strides=self.strides,
                   padding=self.padding, kernel_initializer="he_normal"),

            tfa.layers.InstanceNormalization(),
            # BatchNormalization(),

            LeakyReLU(0.2),
        ]

    def __call__(self, inputs):
        x = inputs

        splits = tf.split(value=x, num_or_size_splits=self.bands, axis=self.axis)

        for i in range(self.bands):
            for layer in self.layers:
                splits[i] = layer(splits[i])

        cat = tf.concat(splits, axis=self.axis)
        # x = tf.add(x, cat)

        return cat

    def get_config(self):
        config = super(MBBlock, self).get_config()
        config.update({
            'bands': self.bands,
            'axis': self.axis,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
        })


if __name__ == '__main__':
    mbb_model = mbb_model()
