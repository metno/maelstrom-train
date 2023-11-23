import argparse
import numpy as np
import time
import sys

def main():
    parser = argparse.ArgumentParser("This program reads one AP1 file and computes the reading performance of the filesystem")
    parser.add_argument("-f", default="/p/scratch/deepacf/maelstrom/maelstrom_data/ap1/air_temperature/5TB/20200301T03Z.nc", help="Read data from these files (e.g.  /p/scratch/deepacf/maelstrom/maelstrom_data/ap1/air_temperature/5TB/20200301T03Z.nc)", dest="files", nargs='*')
    parser.add_argument("-m", default="open_mfdataset", help="Which data loading method to use?", choices=["open_mfdataset", "open_dataset_load", "open_dataset", "netcdf", "raw"], dest="method")
    parser.add_argument("-s", help="Shuffle leadtimes", dest="shuffle_leadtimes", action="store_true")

    args = parser.parse_args()

    s_time = time.time()
    size = 0

    # vars = ["predictors", "static_predictors", "target_mean"]
    vars = ["predictors", "target_mean"]
    if args.method == "open_mfdataset":
        # Using xr.open_mfdataset and extracting the first time step
        import xarray as xr
        with xr.open_mfdataset(args.files, combine="nested", concat_dim="time") as ds:
            for t in range(len(args.files)):
                for var in vars:
                    if args.shuffle_leadtimes:
                        for i in range(ds[var].shape[1]):
                            print(i)
                            q = ds[var][t, i, ...].load()
                            size += np.product(q.shape)*4
                    else:
                        q = ds[var][t, :, ...].load()
                        size += np.product(q.shape)*4

    elif args.method == "open_mfdataset_load":
        # Using xr.open_mfdataset and .load()
        import xarray as xr
        if args.shuffle_leadtimes:
            raise ValueError("open_dataset_load not supported with -s")
        with xr.open_mdataset(args.files) as ds:
            ds.load()
            for da in ds:
                size += ds[da].size*4

    elif args.method == "open_dataset":
        # Using xr.open_dataset and extracting each variable, one by one
        import xarray as xr
        for filename in args.files:
            with xr.open_dataset(filename) as ds:
                for da in vars:
                    if args.shuffle_leadtimes:
                        for i in range(ds[da].shape[0]):
                            q = ds[da][i, ...].values
                            size += q.size*4
                    else:
                        q = ds[da][:].values
                        size += q.size*4

    elif args.method == "netcdf":
        # Using NetCDF4 package
        import netCDF4
        for filename in args.files:
            with netCDF4.Dataset(filename) as file:
                for var in vars:
                    if args.shuffle_leadtimes:
                        for i in range(file.variables[var].shape[0]):
                            q = file.variables[var][i, ...]
                            size += np.product(q.shape) * 4
                    else:
                        q = file.variables[var][:]
                        size += np.product(q.shape) * 4

    elif args.method == "raw":
        if args.shuffle_leadtimes:
            raise ValueError("raw not supported with -s")
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


if __name__ == "__main__":
    main()
