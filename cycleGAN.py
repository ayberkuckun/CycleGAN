import os

import tensorflow as tf
import wandb
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

import helpers.argument_helper
import helpers.dataset_helper
import helpers.preprocessing_helper
import helpers.setting_helper
import helpers.wandb_helper
from helpers.network_helper import GeneratorModel, DiscriminatorModel, LinearDecay
import helpers.image_helper
import helpers.training_helper

# Settings for efficent training.
seed = 42
helpers.setting_helper.set_settings(seed)

# Start wandb for experiment tracking.
helpers.wandb_helper.start(id=None)

# Get Arguments.
args = helpers.argument_helper.get_args()

# Get Dataset.
# dataset_name ="apple2orange"
# dataset_name = "horse2zebra"
dataset_name = "summer2winter_yosemite"

domain1 = "summer"
domain2 = "winter"

trial = 0

train_X_raw, train_Y_raw, test_X_raw, test_Y_raw = helpers.dataset_helper.get_dataset(dataset_name, seed)

# Process Dataset.
train_X, train_Y = helpers.preprocessing_helper.preprocess_dataset_train(train_X_raw, train_Y_raw, dataset_name, seed)
# test_X, test_Y = helpers.preprocessing_helper.preprocess_dataset_test(test_X_raw, test_Y_raw, dataset_name, seed)

X = next(iter(train_X))
Y = next(iter(train_Y))

dataset_len = min(len(train_X), len(train_Y))

EPOCH = 200
lr_init = 2e-4
steps_total = EPOCH * dataset_len
decay_start_step = 100 * dataset_len

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
models_list = [generator_g, generator_f, discriminator_x, discriminator_y]

checkpoint_path = os.path.join(os.getcwd(), f"checkpoints/train/{dataset_name}/trial_{str(trial)}")
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

# current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
# train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
# test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
# train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# tf.profiler.experimental.server.start(6009)
# tf.profiler.experimental.start(train_log_dir)

image_path = os.path.join(os.getcwd(), f"images/train/{dataset_name}/trial_{str(trial)}/")
if not os.path.isdir(image_path):
    os.makedirs(image_path)

for epoch in range(0, EPOCH + 1):
    # with tf.profiler.experimental.Trace('train', step_num=epoch, _r=1):
    for i, data in enumerate(tqdm(tf.data.Dataset.zip((train_X, train_Y)))):
        x_real, y_real = data

        loss_dict = helpers.training_helper.train_step(x_real, y_real, models_list, optimizers_list)

        if i % 5 == 0:
            wandb.log(loss_dict)

            # if i % 100 == 0:
    helpers.image_helper.generate_images2(generator_g, X, image_path + domain1, epoch)
    helpers.image_helper.generate_images2(generator_f, Y, image_path + domain2, epoch)

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

# tf.profiler.experimental.stop()
