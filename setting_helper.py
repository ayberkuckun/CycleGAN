import os

import tensorflow as tf


def set_settings(seed):
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = '1'
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=C:/Users/altan/miniconda3/envs/tf/Library/bin'

    tf.keras.utils.set_random_seed(seed)
    # tf.config.experimental.enable_op_determinism()
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # options = tf.data.Options()
    # options.experimental_optimization.apply_default_optimizations = True
    # options.experimental_optimization.map_parallelization = True
    # options.experimental_optimization.parallel_batch = True
