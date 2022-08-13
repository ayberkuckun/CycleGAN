import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras.layers import Conv2D, ReLU, Add, Lambda, Conv2DTranspose, LeakyReLU, \
    ZeroPadding2D, Activation
from tensorflow.keras import initializers

variance = np.sqrt(0.02)


class GeneratorModel(tf.keras.Model):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.n_res_blocks = args.n_res_blocks

        self.IN1 = tfa.layers.InstanceNormalization(center=True, scale=True,
                                                    beta_initializer=initializers.RandomNormal(stddev=variance),
                                                    gamma_initializer=initializers.RandomNormal(stddev=variance))
        self.IN2 = tfa.layers.InstanceNormalization(center=True, scale=True,
                                                    beta_initializer=initializers.RandomNormal(stddev=variance),
                                                    gamma_initializer=initializers.RandomNormal(stddev=variance))
        self.IN3 = tfa.layers.InstanceNormalization(center=True, scale=True,
                                                    beta_initializer=initializers.RandomNormal(stddev=variance),
                                                    gamma_initializer=initializers.RandomNormal(stddev=variance))
        self.IN4 = tfa.layers.InstanceNormalization(center=True, scale=True,
                                                    beta_initializer=initializers.RandomNormal(stddev=variance),
                                                    gamma_initializer=initializers.RandomNormal(stddev=variance))
        self.IN5 = tfa.layers.InstanceNormalization(center=True, scale=True,
                                                    beta_initializer=initializers.RandomNormal(stddev=variance),
                                                    gamma_initializer=initializers.RandomNormal(stddev=variance))

        self.c7s1_64 = Conv2D(filters=64, kernel_size=7, strides=1, padding='valid',
                              kernel_initializer=initializers.RandomNormal(stddev=variance), use_bias=False)

        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.relu3 = ReLU()
        self.relu4 = ReLU()
        self.relu5 = ReLU()
        # self.relu6 = ReLU()
        self.tanh = Activation('tanh', dtype='float32')

        self.d128 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same',
                           kernel_initializer=initializers.RandomNormal(stddev=variance), use_bias=False)
        self.d256 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same',
                           kernel_initializer=initializers.RandomNormal(stddev=variance), use_bias=False)

        self.reflection_padding1 = Lambda(lambda x: tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT'))
        self.reflection_padding2 = Lambda(lambda x: tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT'))

        self.u128 = Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same',
                                    kernel_initializer=initializers.RandomNormal(stddev=variance), use_bias=False)
        self.u64 = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same',
                                   kernel_initializer=initializers.RandomNormal(stddev=variance), use_bias=False)

        self.c7s1_3 = Conv2D(filters=3, kernel_size=7, strides=1, padding='valid',
                             kernel_initializer=initializers.RandomNormal(stddev=variance))

        self.resblocks = [self.resblock(3, 256) for _ in range(args.n_res_blocks)]

        self.build(input_shape=[None, 256, 256, 3])

    def resblock(self, kernelsize, filters):
        rp1 = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT'))

        conv1 = Conv2D(filters, kernelsize, padding='valid',
                       kernel_initializer=initializers.RandomNormal(stddev=variance), use_bias=False)
        in1 = tfa.layers.InstanceNormalization(center=True, scale=True,
                                               beta_initializer=initializers.RandomNormal(stddev=variance),
                                               gamma_initializer=initializers.RandomNormal(stddev=variance))
        relu1 = ReLU()

        rp2 = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT'))
        conv2 = Conv2D(filters, kernelsize, padding='valid',
                       kernel_initializer=initializers.RandomNormal(stddev=variance), use_bias=False)
        in2 = tfa.layers.InstanceNormalization(center=True, scale=True,
                                               beta_initializer=initializers.RandomNormal(stddev=variance),
                                               gamma_initializer=initializers.RandomNormal(stddev=variance))

        # relu2 = ReLU()  # official pytorch implementation does not do this, reference paper does

        return [rp1, conv1, in1, relu1, rp2, conv2, in2]

    def call(self, inputs):
        x = self.reflection_padding1(inputs)
        x = self.c7s1_64(x)
        x = self.IN1(x)
        x = self.relu1(x)

        # down sampling 1
        x = self.d128(x)
        x = self.IN2(x)
        x = self.relu2(x)

        # down sampling 2
        x = self.d256(x)
        x = self.IN3(x)
        x = self.relu3(x)

        for i in range(self.args.n_res_blocks):
            [rp1, conv1, in1, relu1, rp2, conv2, in2] = self.resblocks[i]
            fx = rp1(x)
            fx = conv1(fx)
            fx = in1(fx)
            fx = relu1(fx)
            fx = rp2(fx)
            fx = conv2(fx)
            fx = in2(fx)
            x = Add()([x, fx])
            # x = relu2(x)

        x = self.u128(x)
        x = self.IN4(x)
        x = self.relu4(x)

        x = self.u64(x)
        x = self.IN5(x)
        x = self.relu5(x)

        x = self.reflection_padding2(x)
        x = self.c7s1_3(x)
        x = self.tanh(x)

        return x

    def build_graph(self):
        x = tf.keras.layers.Input(shape=(256, 256, 3), name='Input')
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class DiscriminatorModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.zp1 = ZeroPadding2D(padding=1)
        self.c64 = Conv2D(filters=64, kernel_size=4, strides=2, padding='valid', name='Conv64',
                          kernel_initializer=initializers.RandomNormal(stddev=variance))
        self.LeReLU1 = LeakyReLU(alpha=0.2, name='LeakyRelu1')

        self.zp2 = ZeroPadding2D(padding=1)
        self.c128 = Conv2D(filters=128, kernel_size=4, strides=2, use_bias=False, padding='valid', name='Conv128',
                           kernel_initializer=initializers.RandomNormal(stddev=variance))
        self.IN1 = tfa.layers.InstanceNormalization(name='InsNorm1', center=True, scale=True,
                                                    beta_initializer=initializers.RandomNormal(stddev=variance),
                                                    gamma_initializer=initializers.RandomNormal(stddev=variance))
        self.LeReLU2 = LeakyReLU(alpha=0.2, name='LeakyRelu2')

        self.zp3 = ZeroPadding2D(padding=1)
        self.c256 = Conv2D(filters=256, kernel_size=4, strides=2, use_bias=False, padding='valid', name='Conv256',
                           kernel_initializer=initializers.RandomNormal(stddev=variance))
        self.IN2 = tfa.layers.InstanceNormalization(name='InsNorm2', center=True, scale=True,
                                                    beta_initializer=initializers.RandomNormal(stddev=variance),
                                                    gamma_initializer=initializers.RandomNormal(stddev=variance))
        self.LeReLU3 = LeakyReLU(alpha=0.2, name='LeakyRelu3')

        # stride=1 is not explicitly mentioned in the paper, but this results in 70x70 patches
        self.zp4 = ZeroPadding2D(padding=1)
        self.c512 = Conv2D(filters=512, kernel_size=4, strides=1, use_bias=False, padding='valid', name='Conv512',
                           kernel_initializer=initializers.RandomNormal(stddev=variance))
        self.IN3 = tfa.layers.InstanceNormalization(name='InsNorm3', center=True, scale=True,
                                                    beta_initializer=initializers.RandomNormal(stddev=variance),
                                                    gamma_initializer=initializers.RandomNormal(stddev=variance))
        self.LeReLU4 = LeakyReLU(alpha=0.2, name='LeakyRelu4')

        self.zp5 = ZeroPadding2D(padding=1)
        self.c_out = Conv2D(filters=1, kernel_size=4, strides=1, padding='valid', name='Conv1',
                            kernel_initializer=initializers.RandomNormal(stddev=variance), dtype='float32')

        self.build(input_shape=[None, 256, 256, 3])

    def call(self, inputs):
        x = self.zp1(inputs)
        x = self.c64(x)
        x = self.LeReLU1(x)

        x = self.zp2(x)
        x = self.c128(x)
        x = self.IN1(x)
        x = self.LeReLU2(x)

        x = self.zp3(x)
        x = self.c256(x)
        x = self.IN2(x)
        x = self.LeReLU3(x)

        x = self.zp4(x)
        x = self.c512(x)
        x = self.IN3(x)
        x = self.LeReLU4(x)

        x = self.zp5(x)
        x = self.c_out(x)

        return x

    def build_graph(self):
        x = tf.keras.layers.Input(shape=(256, 256, 3), name='Input')
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
