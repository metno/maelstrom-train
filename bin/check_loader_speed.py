import os
import sys
import argparse

import maelstrom
import numpy as np
import time
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('file', help='')

    args = parser.parse_args()
    # tf.config.threading.set_intra_op_parallelism_threads(1)
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.set_soft_device_placement(True)

    config = maelstrom.load_yaml(args.file)
    # config["loader"]["debug"] = True
    loader = maelstrom.loader.get(config["loader"])

    dataset = loader.get_dataset()

    s_time = time.time()
    count = 0
    print("Starting")
    for predictors, observations in dataset:
        print(f"{count+1}/{loader.num_patches}", time.time() - s_time)
        # print(np.nanmean(predictors))
        count += 1
    print(time.time() - s_time, predictors.shape, count)
    s_time = time.time()


if __name__ == "__main__":
    main()
