import argparse
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import maelstrom

""" This script tests the performance of the data loader
"""

# num_threads = 1
# tf.config.threading.set_intra_op_parallelism_threads(num_threads)
# tf.config.threading.set_inter_op_parallelism_threads(num_threads)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", help="Read data from these files", nargs="*")
    parser.add_argument("-j", default=1, type=get_num_parallel_calls, help="Number of parallel calls (number or AUTO)", dest="num_parallel_calls")
    parser.add_argument("--config", type=maelstrom.load_yaml, help="Use loader from this config file")
    args = parser.parse_args()

    if args.config is not None:
        loader = maelstrom.loader.Loader.from_config(args.config["loader"])
    else:
        if len(args.files) == 0:
            filenames = ["data/air_temperature/5TB/2021030*T*Z.nc"]
        else:
            filenames = args.files

        loader = maelstrom.loader.Loader(filenames=filenames,
                patch_size=256, predict_diff=True, prefetch=1, batch_size=1,
                num_parallel_calls=args.num_parallel_calls,
                # fake=True,
                limit_leadtimes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                normalization="data/air_temperature/normalization.yml",
                extra_features=[{"type": "leadtime"}],
                quick_metadata=True)
    dataset = loader.get_dataset()

    print(loader)

    # Load all the data
    s_time = time.time()
    count = 0
    for k in dataset:
        count += 1
        # if count % loader.num_patches_per_file == 0:
        #     print(f"Done {count}: ", time.time() - s_time)

    # Print timing statistics
    for k, v in loader.timing.items():
        print(k, v)
    print("TOTAL TIME:", time.time() - s_time)


def get_num_parallel_calls(num_parallel_calls):
    if num_parallel_calls == "AUTOTUNE":
        return tf.data.AUTOTUNE
    else:
        return int(num_parallel_calls)


if __name__ == "__main__":
    main()
