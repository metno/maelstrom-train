import argparse
import numpy as np
import time
import sys
import xarray as xr
import netCDF4

def main():
    parser = argparse.ArgumentParser("This program reads one AP1 file and computes the reading performance of the filesystem")
    parser.add_argument("-f", default=["/p/scratch/deepacf/maelstrom/maelstrom_data/ap1/air_temperature/5TB/20200301T03Z.nc"], help="Read data from these files (e.g.  /p/scratch/deepacf/maelstrom/maelstrom_data/ap1/air_temperature/5TB/20200301T03Z.nc)", dest="files", nargs='*')
    parser.add_argument("-m", default="open_mfdataset", help="Which data loading method to use?", choices=["open_mfdataset", "open_dataset_load", "open_dataset", "netcdf", "raw"], dest="method")
    parser.add_argument("-s", help="Read each leadtime at a time", dest="read_single_leadtime", action="store_true")
    parser.add_argument("-v", default=["predictors", "target_mean"], help="Which variables to read?", dest="variables", nargs='*')

    args = parser.parse_args()

    s_time = time.time()
    size = 0

    if args.method == "open_mfdataset":
        # Using xr.open_mfdataset and extracting the first time step
        chunks = {}
        if args.read_single_leadtime:
            chunks = {"leadtime": 1}
        with xr.open_mfdataset(args.files, combine="nested", concat_dim="time", chunks=chunks) as ds:
            for t in range(len(args.files)):
                # print(t)
                for var in args.variables:
                    if args.read_single_leadtime:
                        for i in range(ds[var].shape[1]):
                            # print(i)
                            q = ds[var][t, i, ...].load()
                            size += np.product(q.shape)*4
                    else:
                        q = ds[var][t, :, ...].load()
                        size += np.product(q.shape)*4

    elif args.method == "open_dataset":
        # Using xr.open_dataset and extracting each variable, one by one
        for filename in args.files:
            with xr.open_dataset(filename) as ds:
                for da in args.variables:
                    if args.read_single_leadtime:
                        for i in range(ds[da].shape[0]):
                            q = ds[da][i, ...].load()
                            size += q.size*4
                    else:
                        q = ds[da][:].load()
                        size += q.size*4

    elif args.method == "netcdf":
        # Using NetCDF4 package
        for filename in args.files:
            with netCDF4.Dataset(filename) as file:
                for var in args.variables:
                    if args.read_single_leadtime:
                        for i in range(file.variables[var].shape[0]):
                            q = file.variables[var][i, ...]
                            size += np.product(q.shape) * 4
                    else:
                        q = file.variables[var][:]
                        size += np.product(q.shape) * 4

    elif args.method == "raw":
        """Read the raw bytes out of the file. Note that this reads all data
        in the file (i.e. ignoring -v)
        """
        if args.read_single_leadtime:
            raise ValueError("raw not supported with -s")

        for filename in args.files:
            file = open(filename, 'rb')
            s_time = time.time();
            buffer_size = 10000000

            x=file.read(buffer_size)
            size = buffer_size
            while x:
                x=file.read(buffer_size)
                size += buffer_size
    else:
        raise Exception(f"Unknown method {method}")

    total_time = time.time() - s_time
    size = size / 1024**3
    print(f"Time: {total_time:.2f} s")
    print(f"Size: {size:.2f} GB")
    print(f"Performance: {size / total_time:.2f} GB/s")


if __name__ == "__main__":
    main()
