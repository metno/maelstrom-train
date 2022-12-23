import maelstrom.new_loader
import numpy as np
import time
import sys
import tensorflow as tf

config = {
          # "filenames": [ "data/air_temperature/5TB/2020030*T*Z.nc"],
          "filenames": [ "data/air_temperature/5TB/20200*T*Z.nc"],
          # "filenames": [ "test/files/air_temperature/5GB/*T*Z.nc"],
          "normalization": "test/files/normalization.yml",
          "predict_diff": True,
          # "limit_predictors": ["air_temperature_2m", "precipitation_amount", "altitude"],
          "patch_size": 256,
          # "debug": True,
          "limit_leadtimes": [0, 1, 2],
          # "extra_features": [{"type": "x"}, {"type": "x"}, {"type": "x"}],
          # "prefetch": 1,
          }
loader = maelstrom.new_loader.Loader(**config)

"""
filename = "data/air_temperature/5TB/20200302T09Z.nc"
s_time = time.time()
loader.parse_file(filename)
print(time.time() - s_time)

sys.exit()
"""

# dataset = loader.get_dataset(tf.data.AUTOTUNE)
dataset = loader.get_dataset(num_parallel_calls=12)

count = 0
for sample in dataset:
    # print(count)
    maelstrom.util.print_memory_usage()
    sys.exit()
    count += 1

s_time = time.time()
count = 0
first_time = None
N = loader.num_patches_per_file
print(N)
for sample in dataset:
    # print(sample[0].shape)
    file_index = count // N
    if count % N == 0:
        if file_index == 0:
            first_time = time.time()
        print("## Start file:", file_index, time.time() - loader.s_time)

    # print(count)
    # print("##    ", count, time.time() - loader.s_time, (time.time() - loader.s_time) / (count + 1) * N)
    if (count + 1) % N == 0:
        # Last one for the file
        print("   End file:", file_index)
        print("   Curr accum time:", time.time() - loader.s_time)
        print("   Time per file:", (time.time() - loader.s_time) / (file_index + 1))
        if file_index > 0:
            print("   Time per file (skip first):", (time.time() - first_time) / (file_index))
        if file_index == 0:
            first_time = time.time()
        # maelstrom.util.print_memory_usage("%d" % count)
        # print("##    ", count // N, time.time() - first_time, (time.time() - first_time) / (count - 1) * N)
        pass
    if count == 2:
        second_time = time.time()

    count += 1
print(loader.logger)
total = 0
for k,v in loader.logger.results.items():
    total += v
print("Logger total", total)
total_time = time.time() - s_time
print("Total time", total_time)
print("Time per file", total_time / loader.num_files)
print("Number of samples", count)
