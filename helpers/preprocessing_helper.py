import tensorflow as tf

BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
AUTOTUNE = tf.data.AUTOTUNE


def preprocess_dataset_train(data, label, dataset_name, seed):
    if dataset_name == "summer2winter_yosemite":
        processed_data = data.cache().map(
            preprocess_image_train_2, num_parallel_calls=AUTOTUNE).cache().shuffle(
            len(data), seed=seed).batch(BATCH_SIZE).prefetch(BATCH_SIZE)

        processed_label = label.cache().map(
            preprocess_image_train_2, num_parallel_calls=AUTOTUNE).cache().shuffle(
            len(label), seed=seed).batch(BATCH_SIZE).prefetch(BATCH_SIZE)
    else:
        processed_data = data.cache().map(
            preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
            len(data), seed=seed).batch(BATCH_SIZE).prefetch(BATCH_SIZE)

        processed_label = label.cache().map(
            preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
            len(label), seed=seed).batch(BATCH_SIZE).prefetch(BATCH_SIZE)

    return processed_data, processed_label


def preprocess_dataset_test(data, label, dataset_name, seed):
    if dataset_name == "summer2winter_yosemite":
        processed_data = data.cache().map(
            preprocess_image_test_2, num_parallel_calls=AUTOTUNE).cache().shuffle(
            len(data), seed=seed).batch(BATCH_SIZE).prefetch(BATCH_SIZE)

        processed_label = label.cache().map(
            preprocess_image_test_2, num_parallel_calls=AUTOTUNE).cache().shuffle(
            len(label), seed=seed).batch(BATCH_SIZE).prefetch(BATCH_SIZE)
    else:
        processed_data = data.cache().map(
            preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
            len(data), seed=seed).batch(BATCH_SIZE).prefetch(BATCH_SIZE)

        processed_label = label.cache().map(
            preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
            len(label), seed=seed).batch(BATCH_SIZE).prefetch(BATCH_SIZE)

    return processed_data, processed_label


def random_crop(image):
    cropped_image = tf.image.random_crop(
        image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image


# normalizing the images to [-1, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def random_jitter(image):
    # resizing to 286 x 286 x 3
    image = tf.image.resize(image, [286, 286],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # randomly cropping to 256 x 256 x 3
    image = random_crop(image)

    # random mirroring
    # image = tf.image.random_flip_left_right(image)

    return image


def preprocess_image_train(image, label):
    image = random_jitter(image)
    image = normalize(image)
    return image


def preprocess_image_train_2(image):
    image = random_jitter(image)
    image = normalize(image)
    return image


def preprocess_image_test(image, label):
    image = normalize(image)
    return image


def preprocess_image_test_2(image):
    image = normalize(image)
    return image
