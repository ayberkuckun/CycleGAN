import tensorflow as tf


LAMBDA = 10
mse = tf.keras.losses.MeanSquaredError()


def discriminator_loss_fc(disc_x_real, disc_x_fake):
    # maximize L_{GAN}
    return mse(tf.ones_like(disc_x_real), disc_x_real) + mse(tf.zeros_like(disc_x_fake), disc_x_fake)


def cycle_loss_fc(real_image, cycled_image):
    return LAMBDA * tf.reduce_mean(tf.abs(real_image - cycled_image))


def generator_loss_fc(disc_fake):
    # minimize L_{GAN}
    return mse(tf.ones_like(disc_fake), disc_fake)


def identity_loss_fc(real, same):
    return LAMBDA * 0.5 * tf.reduce_mean(tf.abs(real - same))
