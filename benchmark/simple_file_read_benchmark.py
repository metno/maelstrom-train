import argparse
import numpy as np
import time
import sys

parser = argparse.ArgumentParser("This program reads one AP1 file and computes the reading performance of the filesystem")
parser.add_argument("-f", default="/p/scratch/deepacf/maelstrom/maelstrom_data/ap1/air_temperature/5TB/20200301T03Z.nc", help="Read data from these files (e.g.  /p/scratch/deepacf/maelstrom/maelstrom_data/ap1/air_temperature/5TB/20200301T03Z.nc)", dest="file")
parser.add_argument("-m", default="open_mfdataset", help="Which data loading method to use?", choices=["open_mfdataset", "open_dataset_load", "open_dataset", "netcdf", "raw"], dest="method")

args = parser.parse_args()

s_time = time.time()
size = 0

if args.method == "open_mfdataset":
    # Using xr.open_mfdataset and extracting the first time step
    import xarray as xr
    with xr.open_mfdataset([args.file], combine="nested", concat_dim="time") as ds:
        vars = ["predictors", "static_predictors", "target_mean"]
        for var in vars:
            q = ds[var][0, ...].load()
            size += np.product(q.shape)*4
elif args.method == "open_dataset_load":
    # Using xr.open_dataset and .load()
    import xarray as xr
    with xr.open_dataset(args.file) as ds:
        ds.load()
        for da in ds:
            size += ds[da].size*4
elif args.method == "open_dataset":
    # Using xr.open_dataset and extracting each variable, one by one
    import xarray as xr
    with xr.open_dataset(args.file) as ds:
        for da in ds:
            q = ds[da].values
            size += q.size*4
elif args.method == "netcdf":
    # Using NetCDF4 package
    import netCDF4
    with netCDF4.Dataset(args.file) as file:
        for var in file.variables:
            size += np.product(file.variables[var].shape) * 4
            q = file.variables[var][:]
elif args.method == "raw":
    file = open(args.file, 'rb')
    s_time = time.time();
    size = 0
    N = 10000000
    x=file.read(N)
    while x:
        x=file.read(N)
        size += N
else:
    raise Exception(f"Unknown method {method}")

total_time = time.time() - s_time
size = size / 1024**3
print(f"Time: {total_time:.2f} s")
print(f"Size: {size:.2f} GB")
print(f"Performance: {size / total_time:.2f} GB/s")
