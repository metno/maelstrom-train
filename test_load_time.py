import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import time
import maelstrom
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

""" This script tests the performance of the data loader
"""

# num_threads = 1
# tf.config.threading.set_intra_op_parallelism_threads(num_threads)
# tf.config.threading.set_inter_op_parallelism_threads(num_threads)

num_parallel_calls = 1 # tf.data.AUTOTUNE

if 1:
    loader = maelstrom.loader.Loader(filenames=["data/air_temperature/5TB/2021030*T*Z.nc"],
            patch_size=256, predict_diff=True, prefetch=1, batch_size=1,
            num_parallel_calls=num_parallel_calls,
            # fake=True,
            limit_leadtimes=[0, 1, 2, 3, 4, 5],
            normalization="data/air_temperature/normalization.yml",
            extra_features=[{"type": "leadtime"}],
            quick_metadata=True)
else:
    config = maelstrom.load_yaml("etc/exp1.yml")
    loader = maelstrom.loader.get(config["loader"])
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
