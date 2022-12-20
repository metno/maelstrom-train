import maelstrom.new_loader
import numpy as np
import time
import sys

config = {
          "filenames": [ "data_b/air_temperature/5TB/20200301T*Z.nc"],
          # "filenames": [ "test/files/air_temperature/5GB/*T*Z.nc"],
          "normalization": "test/files/normalization.yml",
          "predict_diff": True,
          "limit_predictors": ["air_temperature_2m", "precipitation_amount", "altitude"],
          "patch_size": 256,
          "limit_leadtimes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
          "extra_features": [{"type": "x"}],
          }
loader = maelstrom.new_loader.Loader(**config)
filename = "data_b/air_temperature/5TB/20200301T03Z.nc"
s_time = time.time()
loader.parse_file(filename)
print(time.time() - s_time)
s_time = time.time()
loader.parse_file_netcdf(filename)
print(time.time() - s_time)

sys.exit()

dataset = loader.get_dataset()

s_time = time.time()
count = 0
for sample in dataset:
    # print(np.nanmean(sample), sample.shape)
    # print(time.time() - s_time, len(sample), sample[0].shape)
    count += 1
print("Total time", time.time() - s_time)
print("Number of samples", count)
