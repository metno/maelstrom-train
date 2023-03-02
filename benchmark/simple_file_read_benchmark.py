import argparse
import netCDF4
import numpy as np
import time
import sys

parser = argparse.ArgumentParser("This program reads one AP1 file and computes the reading performance of the filesystem")
parser.add_argument("-f", default="/p/scratch/deepacf/maelstrom/maelstrom_data/ap1/air_temperature/5TB/20200301T03Z.nc", help="Read data from these files (e.g.  /p/scratch/deepacf/maelstrom/maelstrom_data/ap1/air_temperature/5TB/2020030*.nc)", dest="file")

args = parser.parse_args()

s_time = time.time()
total = 0
with netCDF4.Dataset(args.file) as file:
    vars = ["predictors", "static_predictors", "target_mean"]
    for var in vars:
        total += np.product(file.variables[var].shape) * 4
        q = file.variables[var][:]
total_time = time.time() - s_time
size = total / 1024**3
print(f"Time: {total_time:.2f} s")
print(f"Size: {size:.2f} GB")
print(f"Performance: {size / total_time:.2f} GB/s")
