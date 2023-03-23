from hparams import hparams
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, ReLU, Concatenate
from keras.layers import BatchNormalization, Dropout
from keras import Input, Model


def conv_block(inputs, filters=16, kernel_size=5, strides=2, padding="same"):
    x = Conv2D(filters=filters, kernel_size=kernel_size,
               strides=strides, padding=padding)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x


def deconv_block(inputs, filters=16, kernel_size=5, strides=2,
                 padding="same", concat=True, concat_input=None,
                 concat_axis=3):
    if concat:
        x = Concatenate(axis=concat_axis)([inputs, concat_input])
        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                            strides=strides, padding=padding)(x)
    else:
        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                            strides=strides, padding=padding)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def UNet(inputs=Input((512, 128, 1))):
    conv1 = conv_block(inputs, filters=16, kernel_size=5, strides=2)

    conv2 = conv_block(conv1, filters=32, kernel_size=5, strides=2)

    conv3 = conv_block(conv2, filters=64, kernel_size=5, strides=2)

    conv4 = conv_block(conv3, filters=128, kernel_size=5, strides=2)

    conv5 = conv_block(conv4, filters=256, kernel_size=5, strides=2)

    conv6 = conv_block(conv5, filters=512, kernel_size=5, strides=2)

    deconv1 = deconv_block(conv6, filters=256, kernel_size=5, strides=2, concat=False)

    deconv2 = deconv_block(deconv1, filters=128, kernel_size=5, strides=2,
                           concat=True, concat_axis=3, concat_input=conv5)

    deconv3 = deconv_block(deconv2, filters=64, kernel_size=5, strides=2,
                           concat=True, concat_axis=3, concat_input=conv4)

    deconv4 = deconv_block(deconv3, filters=32, kernel_size=5, strides=2,
                           concat=True, concat_axis=3, concat_input=conv3)

    deconv5 = deconv_block(deconv4, filters=16, kernel_size=5, strides=2,
                           concat=True, concat_axis=3, concat_input=conv2)
    deconv6 = Concatenate()([deconv5, conv1])
    deconv6 = Conv2DTranspose(filters=1, kernel_size=5, strides=2, padding="same")(deconv6)
    deconv6 = ReLU()(deconv6)

    model = Model(inputs=inputs, outputs=deconv6)

    model.summary()

    return model


if __name__ == '__main__':
    model = UNet()
