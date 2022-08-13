import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

import argument_helper
import dataset_helper
import preprocessing_helper
import setting_helper
import wandb
import wandb_helper
from network_helper import GeneratorModel, DiscriminatorModel

# Settings for efficent training.
seed = 42
setting_helper.set_settings(seed)

# Start wandb for experiment tracking.
wandb_helper.start(id=None)

# Get Arguments.
args = argument_helper.get_args()

# Get dataset.
dataset = "horse2zebra"
train_X_raw, train_Y_raw, test_X_raw, test_Y_raw = dataset_helper.get_dataset(dataset, seed)

train_X, train_Y = preprocessing_helper.preprocess_dataset_train(train_X_raw, train_Y_raw, seed)
test_X, test_Y = preprocessing_helper.preprocess_dataset_test(test_X_raw, test_Y_raw, seed)

X = next(iter(train_X))
Y = next(iter(train_Y))

dataset_len = min(len(train_X), len(train_Y))

EPOCH = 200
use_identity_loss = False
LAMBDA = 10
lr_init = 2e-4
steps_total = EPOCH * dataset_len
decay_start_step = 100 * dataset_len


class LinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_init, steps_total, decay_start_step):
        super(LinearDecay, self).__init__()
        self.lr_init = lr_init
        self.steps_total = steps_total
        self.decay_start_step = decay_start_step
        self.current_lr = tf.Variable(initial_value=lr_init, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_lr.assign(tf.cond(
            step >= self.decay_start_step,
            true_fn=lambda: self.lr_init - (step - self.decay_start_step) * (
                    self.lr_init / (self.steps_total - self.decay_start_step)),
            false_fn=lambda: self.lr_init
        ))

        return self.current_lr


# Weights are initialized from a Gaussian distribution N (0, 0.02).

# Define generators and discriminators
generator_g = GeneratorModel(args)
generator_f = GeneratorModel(args)

discriminator_x = DiscriminatorModel()
discriminator_y = DiscriminatorModel()

# Define optimizers
# optimizer_generator_g = Adam(learning_rate=lr_init, beta_1=0.5, beta_2=0.999)
# optimizer_generator_f = Adam(learning_rate=lr_init, beta_1=0.5, beta_2=0.999)
#
# optimizer_discriminator_x = Adam(learning_rate=lr_init, beta_1=0.5, beta_2=0.999)
# optimizer_discriminator_y = Adam(learning_rate=lr_init, beta_1=0.5, beta_2=0.999)

optimizer_generator_g = Adam(learning_rate=LinearDecay(lr_init, steps_total, decay_start_step), beta_1=0.5,
                             beta_2=0.999)
optimizer_generator_f = Adam(learning_rate=LinearDecay(lr_init, steps_total, decay_start_step), beta_1=0.5,
                             beta_2=0.999)

optimizer_discriminator_x = Adam(learning_rate=LinearDecay(lr_init, steps_total, decay_start_step), beta_1=0.5,
                                 beta_2=0.999)
optimizer_discriminator_y = Adam(learning_rate=LinearDecay(lr_init, steps_total, decay_start_step), beta_1=0.5,
                                 beta_2=0.999)

optimizer_generator_g = tf.keras.mixed_precision.LossScaleOptimizer(optimizer_generator_g)
optimizer_generator_f = tf.keras.mixed_precision.LossScaleOptimizer(optimizer_generator_f)

optimizer_discriminator_x = tf.keras.mixed_precision.LossScaleOptimizer(optimizer_discriminator_x)
optimizer_discriminator_y = tf.keras.mixed_precision.LossScaleOptimizer(optimizer_discriminator_y)

optimizers_list = [optimizer_generator_g, optimizer_generator_f, optimizer_discriminator_x, optimizer_discriminator_y]

mse = tf.keras.losses.MeanSquaredError()


def discriminator_loss_fc(disc_x_real, disc_x_fake):
    # maximize L_{GAN}
    return mse(tf.ones_like(disc_x_real), disc_x_real) + mse(tf.zeros_like(disc_x_fake), disc_x_fake)


def cycle_loss_fc(real_image, cycled_image):
    return LAMBDA * tf.reduce_mean(tf.abs(real_image - cycled_image))


def generator_loss_fc(disc_y_fake):
    # minimize L_{GAN}
    return mse(tf.ones_like(disc_y_fake), disc_y_fake)  # L_{GAN}


def identity_loss_fc(real, same):
    return LAMBDA * 0.5 * tf.reduce_mean(tf.abs(real - same))


class FakeImagePool:
    def __init__(self, size=50):
        self.size = size
        self.pool = []

    def __call__(self, fake_images):
        out_items = []

        for f_im in fake_images:
            if len(self.pool) < self.size:
                self.pool.append(f_im)
                out_items.append(f_im)
            else:
                if np.random.rand() > 0.5:
                    replace_idx = np.random.randint(0, len(self.pool))
                    out_items.append(self.pool[replace_idx])
                    self.pool[replace_idx] = f_im
                else:
                    out_items.append(f_im)
        return tf.stack(out_items, axis=0)


fake_x_pool = FakeImagePool()
fake_y_pool = FakeImagePool()


@tf.function(jit_compile=True)
def generator_step(x_real, y_real):
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch([generator_g.trainable_variables, generator_f.trainable_variables])

        # Generator G translates X -> Y
        # Generator F translates Y -> X

        y_fake = generator_g(x_real, training=True)
        x_fake = generator_f(y_real, training=True)

        x_cycled = generator_f(y_fake, training=True)
        y_cycled = generator_g(x_fake, training=True)

        # Calcualte identity loss
        if use_identity_loss:
            # x_same and y_same are used for identity loss.
            x_same = generator_f(x_real, training=True)
            y_same = generator_g(y_real, training=True)

            identity_loss = identity_loss_fc(x_real, x_same) + identity_loss_fc(y_real, y_same)

        else:
            identity_loss = 0

        # Calculate cyclic loss
        cyclic_loss = cycle_loss_fc(x_real, x_cycled) + cycle_loss_fc(y_real, y_cycled)

        # Calculate generator loss
        disc_x_fake_for_gen = discriminator_x(x_fake, training=True)
        disc_y_fake_for_gen = discriminator_y(y_fake, training=True)

        gen_g_loss = generator_loss_fc(disc_y_fake_for_gen)
        gen_f_loss = generator_loss_fc(disc_x_fake_for_gen)

        # Total generator loss = adversarial loss + cycle loss + identity loss
        total_gen_g_loss = gen_g_loss + cyclic_loss + identity_loss
        total_gen_f_loss = gen_f_loss + cyclic_loss + identity_loss

        total_gen_g_scaled_loss = optimizer_generator_g.get_scaled_loss(total_gen_g_loss)
        total_gen_f_scaled_loss = optimizer_generator_f.get_scaled_loss(total_gen_f_loss)

    generator_g_scaled_gradients = tape.gradient(total_gen_g_scaled_loss,
                                                 generator_g.trainable_variables)
    generator_f_scaled_gradients = tape.gradient(total_gen_f_scaled_loss,
                                                 generator_f.trainable_variables)

    generator_g_gradients = optimizer_generator_g.get_unscaled_gradients(generator_g_scaled_gradients)
    generator_f_gradients = optimizer_generator_f.get_unscaled_gradients(generator_f_scaled_gradients)

    optimizer_generator_g.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))
    optimizer_generator_f.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))

    return {
        "gen_G_loss": gen_g_loss,
        "gen_F_loss": gen_f_loss,
        "cyclic_loss": cyclic_loss,
        "identity_loss": identity_loss,
        "total_gen_G_loss": total_gen_g_loss,
        "total_gen_F_loss": total_gen_f_loss,
        "x_fake": x_fake,
        "y_fake": y_fake
    }


@tf.function(jit_compile=True)
def discriminator_step(x_real, x_fake, y_real, y_fake):
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch([discriminator_x.trainable_variables, discriminator_y.trainable_variables])

        # Discriminator X discriminates X domain images
        # Discriminator Y discriminates Y domain images

        disc_x_real = discriminator_x(x_real, training=True)
        disc_y_real = discriminator_y(y_real, training=True)

        disc_x_fake = discriminator_x(x_fake, training=True)
        disc_y_fake = discriminator_y(y_fake, training=True)

        # Calculate descriminator loss
        disc_x_loss = discriminator_loss_fc(disc_x_real, disc_x_fake) / 2  # divide by 2 not recommended.
        disc_y_loss = discriminator_loss_fc(disc_y_real, disc_y_fake) / 2  # divide by 2 not recommended.

        disc_x_scaled_loss = optimizer_discriminator_x.get_scaled_loss(disc_x_loss)
        disc_y_scaled_loss = optimizer_discriminator_y.get_scaled_loss(disc_y_loss)

    # Calculate the gradients for generator and discriminator

    discriminator_x_scaled_gradients = tape.gradient(disc_x_scaled_loss,
                                                     discriminator_x.trainable_variables)

    discriminator_y_scaled_gradients = tape.gradient(disc_y_scaled_loss,
                                                     discriminator_y.trainable_variables)

    discriminator_x_gradients = optimizer_discriminator_x.get_unscaled_gradients(discriminator_x_scaled_gradients)
    discriminator_y_gradients = optimizer_discriminator_y.get_unscaled_gradients(discriminator_y_scaled_gradients)

    optimizer_discriminator_x.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))
    optimizer_discriminator_y.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables))

    return {
        "disc_X_loss": disc_x_loss,
        "disc_Y_loss": disc_y_loss
    }


# Define training loop
def train_step(x_real, y_real):
    generator_out_dict = generator_step(x_real, y_real)

    x_fake_for_disc = fake_x_pool(generator_out_dict["x_fake"])
    y_fake_for_disc = fake_y_pool(generator_out_dict["y_fake"])

    discriminator_out_dict = discriminator_step(x_real, x_fake_for_disc, y_real, y_fake_for_disc)

    return {
        "gen_G_loss": generator_out_dict["gen_G_loss"],
        "gen_F_loss": generator_out_dict["gen_F_loss"],
        "cyclic_loss": generator_out_dict["cyclic_loss"],
        "identity_loss": generator_out_dict["identity_loss"],
        "total_gen_G_loss": generator_out_dict["total_gen_G_loss"],
        "total_gen_F_loss": generator_out_dict["total_gen_F_loss"],
        "disc_X_loss": discriminator_out_dict["disc_X_loss"],
        "disc_Y_loss": discriminator_out_dict["disc_Y_loss"]
    }


dataset = "cycle_gan/horse2zebra"
checkpoint_path = os.path.join(os.getcwd(), f"checkpoints/train/{dataset}")
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           optimizer_generator_g=optimizers_list[0],
                           optimizer_generator_f=optimizers_list[1],
                           optimizer_discriminator_x=optimizers_list[2],
                           optimizer_discriminator_y=optimizers_list[3])

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)

# reader = tf.train.load_checkpoint(checkpoint_path)
# shape_from_key = reader.get_variable_to_shape_map()
# dtype_from_key = reader.get_variable_to_dtype_map()
#
# a = sorted(shape_from_key.keys())

use_checkpoint = False
if use_checkpoint and ckpt_manager.latest_checkpoint:
    status = ckpt.restore(ckpt_manager.latest_checkpoint)
    status.assert_consumed()
    status.assert_existing_objects_matched()
    print(f'Latest checkpoint restored!! \n {ckpt_manager.latest_checkpoint}')
    wandb.config.update({"epochs": EPOCH}, allow_val_change=True)

else:
    wandb.config.epochs = EPOCH
    wandb.config.batch_size = 1


def generate_images(model, test_input):
    prediction = model(test_input, training=False)
    prediction = tf.cast(prediction, tf.float32)

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


def generate_images2(model, test_input, file_name, epoch):
    prediction = model(test_input, training=False)
    prediction = tf.cast(prediction, tf.float32)
    test_input = test_input

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0].numpy(), prediction[0].numpy()]
    title = ['Input Image', 'Predicted Image']

    print_first_image = epoch == 0

    if print_first_image:
        plt.imsave(file_name + "_original_image" + ".png", display_list[0] * 0.5 + 0.5)
        plt.imsave(file_name + "_epoch_" + str(epoch) + ".png", display_list[1] * 0.5 + 0.5)
    else:
        plt.imsave(file_name + "_epoch_" + str(epoch) + ".png", display_list[1] * 0.5 + 0.5)


lr = lr_init

# current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
# train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
# test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
# train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# tf.profiler.experimental.server.start(6009)
# tf.profiler.experimental.start(train_log_dir)

image_path = os.path.join(os.getcwd(), f"images/train/{dataset}/")
if not os.path.isdir(image_path):
    os.makedirs(image_path)

for epoch in range(0, EPOCH + 1):
    # with tf.profiler.experimental.Trace('train', step_num=epoch, _r=1):
    for i, data in enumerate(tqdm(tf.data.Dataset.zip((train_X, train_Y)))):
        x_real, y_real = data

        loss_dict = train_step(x_real, y_real)

        if i % 5 == 0:
            wandb.log(loss_dict)

            # if i % 100 == 0:
    generate_images2(generator_g, X, image_path + "summer", epoch)
    generate_images2(generator_f, Y, image_path + "winter", epoch)

    # with train_summary_writer.as_default():
    #     tf.summary.scalar('gen_G_loss', loss_dict["gen_G_loss"], step=epoch)
    #     tf.summary.scalar('gen_F_loss', loss_dict["gen_F_loss"], step=epoch)
    #     tf.summary.scalar('cyclic_loss', loss_dict["cyclic_loss"], step=epoch)
    #     tf.summary.scalar('identity_loss', loss_dict["identity_loss"], step=epoch)
    #     tf.summary.scalar('total_gen_G_loss', loss_dict["total_gen_G_loss"], step=epoch)
    #     tf.summary.scalar('total_gen_F_loss', loss_dict["total_gen_F_loss"], step=epoch)
    #     tf.summary.scalar('disc_X_loss', loss_dict["disc_X_loss"], step=epoch)
    #     tf.summary.scalar('disc_Y_loss', loss_dict["disc_Y_loss"], step=epoch)

    if epoch % 10 == 0:
        ckpt_manager.save(checkpoint_number=epoch + 1)

    # if epoch >= 100:
    #     for optimizer in optimizers_list:
    #         lr = lr - (epoch - 100) * (lr_init / 100)
    #         tf.keras.backend.set_value(optimizer.learning_rate, lr)

# tf.profiler.experimental.stop()
