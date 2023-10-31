#!/usr/bin/env python3
import argparse
import matplotlib.pylab as mpl
import netCDF4
import numpy as np
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('file', help='')
    parser.add_argument('-o', dest='ofilename')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    colors = ["r", "g", "b", "y", "k", "purple", "orange", "magenta", "cyan"]
    style = ["o", "s", "v"]
    name = "air_temperature_2m"
    name = "precipitation_amount"
    name = "x_wind_10m"
    num_levels = 6

    with netCDF4.Dataset(args.file) as file:
        print(file.variables["predictor"][:])
        variables = file.variables["predictor"][:]
        variables = ["air_temperature_2m", "precipitation_amount", "cloud_area_fraction", "x_wind_10m", "y_wind_10m"]
        cmap = "turbo"
        num_variables = len(variables)
        for i in range(num_variables):
            name = variables[i]
            Ip = np.where(file.variables["predictor"][:] == name)[0][0]
            Ilt = 1
            var = file.variables["predictors"]
            if var.shape[1] > 300:
                values = var[Ilt, 300:1200, 300:1200, Ip]
            else:
                values = var[Ilt, :, :, Ip]
            Y, X = values.shape
            values = values[0:(Y//32)*32, 0:(X//32)*32]
            # trim
            Nx = num_variables
            Ny = 3
            curr_max = np.copy(values)
            curr_mean = np.copy(values)
            for j in range(num_levels - 1):
                Y, X = curr_mean.shape
                curr_max = curr_max.reshape(Y//2, 2, X//2, 2).max((1, 3))
                curr_mean = curr_mean.reshape(Y//2, 2, X//2, 2).mean((1, 3))
            # mpl.subplot(Ny, Nx, 3*i + 1)
            mpl.subplot(Ny, Nx, i + 1)
            mpl.pcolormesh(values, cmap=cmap)
            # mpl.axis('off')
            mpl.xticks([])
            mpl.yticks([])
            mpl.grid()
            mpl.title(f"{name}")

            mpl.subplot(Ny, Nx, i + Nx + 1)
            mpl.pcolormesh(curr_max, cmap=cmap)
            # mpl.axis('off')
            mpl.xticks([])
            mpl.yticks([])
            mpl.grid()
            # mpl.title(f"Max pooling")

            mpl.subplot(Ny, Nx, i + 2*Nx + 1)
            mpl.pcolormesh(curr_mean, cmap=cmap)
            # mpl.axis('off')
            mpl.xticks([])
            mpl.yticks([])
            mpl.grid()
            # mpl.title(f"Mean pooling")
        # mpl.ylabel("Loss")
        # mpl.xlabel("Epoch")
        # mpl.legend()

    # mpl.xlabel("Epoch")
    if args.ofilename is not None:
        mpl.gcf().set_size_inches(2.5*Nx, 2.5*Ny)
        mpl.savefig(args.ofilename, dpi=300)
    else:
        mpl.show()


if __name__ == "__main__":
    main()
