import tensorflow as tf
import helpers.loss_helper
from helpers.network_helper import FakeImagePool


use_identity_loss = False

fake_x_pool = FakeImagePool()
fake_y_pool = FakeImagePool()


# Define training loop
def train_step(x_real, y_real, models_list, optimizers_list):
    generator_out_dict = generator_step(x_real, y_real, models_list, optimizers_list)

    x_fake_for_disc = fake_x_pool(generator_out_dict["x_fake"])
    y_fake_for_disc = fake_y_pool(generator_out_dict["y_fake"])

    discriminator_out_dict = discriminator_step(x_real, x_fake_for_disc, y_real, y_fake_for_disc, models_list, optimizers_list)

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


@tf.function(jit_compile=True)
def generator_step(x_real, y_real, models_list, optimizers_list):
    generator_g = models_list[0]
    generator_f = models_list[1]
    discriminator_x = models_list[2]
    discriminator_y = models_list[3]

    optimizer_generator_g = optimizers_list[0]
    optimizer_generator_f = optimizers_list[1]

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

            identity_loss = helpers.loss_helper.identity_loss_fc(x_real, x_same) + helpers.loss_helper.identity_loss_fc(y_real, y_same)

        else:
            identity_loss = 0

        # Calculate cyclic loss
        cyclic_loss = helpers.loss_helper.cycle_loss_fc(x_real, x_cycled) + helpers.loss_helper.cycle_loss_fc(y_real, y_cycled)

        # Calculate generator loss
        disc_x_fake_for_gen = discriminator_x(x_fake, training=True)
        disc_y_fake_for_gen = discriminator_y(y_fake, training=True)

        gen_g_loss = helpers.loss_helper.generator_loss_fc(disc_y_fake_for_gen)
        gen_f_loss = helpers.loss_helper.generator_loss_fc(disc_x_fake_for_gen)

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
def discriminator_step(x_real, x_fake, y_real, y_fake, models_list, optimizers_list):
    discriminator_x = models_list[2]
    discriminator_y = models_list[3]

    optimizer_discriminator_x = optimizers_list[2]
    optimizer_discriminator_y = optimizers_list[3]

    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch([discriminator_x.trainable_variables, discriminator_y.trainable_variables])

        # Discriminator X discriminates X domain images
        # Discriminator Y discriminates Y domain images

        disc_x_real = discriminator_x(x_real, training=True)
        disc_y_real = discriminator_y(y_real, training=True)

        disc_x_fake = discriminator_x(x_fake, training=True)
        disc_y_fake = discriminator_y(y_fake, training=True)

        # Calculate descriminator loss
        disc_x_loss = helpers.loss_helper.discriminator_loss_fc(disc_x_real, disc_x_fake) / 2  # divide by 2 not recommended.
        disc_y_loss = helpers.loss_helper.discriminator_loss_fc(disc_y_real, disc_y_fake) / 2  # divide by 2 not recommended.

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
