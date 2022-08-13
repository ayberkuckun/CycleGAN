import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Options for training the CycleGan')
    parser.add_argument('--input_size', help='h and w dimensions of the input image', default=256)
    parser.add_argument('--n_res_blocks', help='number of residual blocks in G', default=9)
    parser.add_argument('--batch_size', help='batch size', default=1)
    parser.add_argument('--pool_size', help='pool size to store fake images of G', default=50)
    parser.add_argument('--data_set', help='name of the dataset', default='apple2orange')
    args = parser.parse_args(args=[])

    return args
