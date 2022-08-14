from matplotlib import pyplot as plt
import tensorflow as tf


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

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0].numpy(), prediction[0].numpy()]

    print_first_image = epoch == 0

    if print_first_image:
        plt.imsave(file_name + "_original_image" + ".png", display_list[0] * 0.5 + 0.5)
        plt.imsave(file_name + "_epoch_" + str(epoch) + ".png", display_list[1] * 0.5 + 0.5)
    else:
        plt.imsave(file_name + "_epoch_" + str(epoch) + ".png", display_list[1] * 0.5 + 0.5)
