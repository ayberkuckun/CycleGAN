import tensorflow as tf
import tensorflow_datasets as tfds


def get_dataset(dataset, seed):
    if dataset == 'apple2orange':
        ds, _ = tfds.load('cycle_gan/apple2orange', as_supervised=True, shuffle_files=True, with_info=True)

        train_X, train_Y = ds["trainA"], ds["trainB"]
        test_X, test_Y = ds['testA'], ds['testB']

        return train_X, train_Y, test_X, test_Y

    elif dataset == 'horse2zebra':
        ds, _ = tfds.load('cycle_gan/horse2zebra', as_supervised=True, shuffle_files=True, with_info=True)

        train_X, train_Y = ds["trainA"], ds["trainB"]
        test_X, test_Y = ds['testA'], ds['testB']

        return train_X, train_Y, test_X, test_Y

    elif dataset == 'summer2winter_yosemite':
        # ds, ds_info = tfds.load('cycle_gan/summer2winter_yosemite', as_supervised=True, shuffle_files=True,
        #                         with_info=True)

        train_X = tf.keras.utils.image_dataset_from_directory(
            "datasets/summer2winter_yosemite/train/trainA/",
            labels=None,
            image_size=(256, 256),
            batch_size=None,
            shuffle=True,
            seed=seed
        )

        train_Y = tf.keras.utils.image_dataset_from_directory(
            "datasets/summer2winter_yosemite/train/trainB/",
            labels=None,
            image_size=(256, 256),
            batch_size=None,
            shuffle=True,
            seed=seed
        )

        test_X = tf.keras.utils.image_dataset_from_directory(
            "datasets/summer2winter_yosemite/test/testA/",
            labels=None,
            image_size=(256, 256),
            batch_size=None,
        )

        test_Y = tf.keras.utils.image_dataset_from_directory(
            "datasets/summer2winter_yosemite/test/testB/",
            labels=None,
            image_size=(256, 256),
            batch_size=None,
        )

        return train_X, train_Y, test_X, test_Y

    else:
        raise ValueError('Dataset not found')
