import maelstrom.new_loader
import numpy as np
import time
import sys

config = {
          "filenames": [ "data/air_temperature/5TB/20200301T*Z.nc"],
          # "filenames": [ "test/files/air_temperature/5GB/*T*Z.nc"],
          "normalization": "test/files/normalization.yml",
          "predict_diff": True,
          # "limit_predictors": ["air_temperature_2m", "precipitation_amount", "altitude"],
          "patch_size": 256,
          # "debug": True,
          "limit_leadtimes": [0, 1, 2],
          # "extra_features": [{"type": "x"}],
          }
loader = maelstrom.new_loader.Loader(**config)

"""
filename = "data/air_temperature/5TB/20200302T09Z.nc"
s_time = time.time()
loader.parse_file(filename)
print(time.time() - s_time)

sys.exit()
"""

dataset = loader.get_dataset(1)

s_time = time.time()
count = 0
for sample in dataset:
    if count % 63 == 0:
        print("## File", count // 63, time.time() - loader.s_time)
    count += 1
    # print(np.nanmean(sample), sample.shape)
    # print(time.time() - s_time, len(sample), sample[0].shape)
    if count > 1:
        # print(count, time.time() - second_time, (time.time() - second_time) / (count - 1))
        pass
    print("##    ", count, time.time() - loader.s_time, (time.time() - loader.s_time) / (count))
    # maelstrom.util.print_memory_usage("%d" % count)
    if count == 2:
        second_time = time.time()
print(loader.logger)
total = 0
for k,v in loader.logger.results.items():
    total += v
print("Logger total", total)
print("Total time", time.time() - s_time)
print("Number of samples", count)
