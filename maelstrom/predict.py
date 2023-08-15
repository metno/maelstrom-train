import argparse
import copy
import datetime
import glob
import json
import os
import sys
import time
import xarray as xr
import netCDF4

import numpy as np
import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras


import maelstrom


def main():
    # fmt: off
    parser = argparse.ArgumentParser("This program performs inference")
    parser.add_argument( "folder", help="Run using this trained model folder")
    parser.add_argument( "file", help="Run using this file")
    parser.add_argument( "-j", type=int, help="Number of threads to train with", dest="num_threads",)
    parser.add_argument( "--hardware", default="cpu", help="Run on GPUS?", choices=["cpu", "gpu", "multigpu"],)
    # parser.add_argument( "-m", help="Run these models", dest="subset_models", nargs="*", required=True,)
    parser.add_argument( "-o", help="Output folder", dest="output_folder",)
    args = parser.parse_args()
    # fmt: on

    config_filename = args.folder + "/config.yml"
    config = maelstrom.load_yaml(config_filename)

    # Set up loader
    loader_config = config["loader"]
    loader_config["filenames"] = [args.file]
    # Disable patching, because it makes it hard to organize the output
    if "patch_size" in loader_config:
        del loader_config["patch_size"]
    loader_config["batch_size"] = 1
    loader = maelstrom.loader.Loader.from_config(loader_config)
    dataset = loader.get_dataset()

    # Set up model
    input_shape = loader.sample_predictor_shape
    quantiles = config["output"]["quantiles"]
    num_outputs = len(quantiles)
    model = maelstrom.models.get(input_shape, num_outputs, **config["model"])
    loss = maelstrom.loss.get(config["loss"], quantiles)
    # optimizer = maelstrom.optimizer.get(**config["training"]["optimizer"])
    # model.compile(
    #     optimizer=optimizer,
    #     loss=loss,
    #     experimental_run_tf_function=False,
    # )
    input_checkpoint_filepath = args.folder + "/checkpoint"
    model.load_weights(input_checkpoint_filepath).expect_partial()

    # TODO: Figure out why there is a leadtime dimension when we are running with_leadtime: False

    # Run predict
    results = model.predict(dataset)

    # for batch, (bfcst, btargets) in enumerate(dataset):
    #     results = model.predict_on_batch(bfcst)
    #     print("Mean", batch, np.mean(results), np.mean(btargets), np.mean(bfcst))

    # model.evaluate(dataset)

    # Adjust for diff
    raw_predictor = list()
    obs = list()
    I = loader.raw_predictor_index
    if I is None:
        raise Exception()
    for curr_fcst, curr_obs in dataset:
        # print(loss(curr_fcst, curr_obs))
        raw_predictor += [curr_fcst[..., I]]
        obs += [curr_obs[..., 0]]
    raw_predictor = np.concatenate(raw_predictor, 0)
    obs = np.concatenate(obs, 0)[:, 0, ...].astype(np.float32)

    # Add predictor to each quantile level
    if loader.predict_diff:
        for q in range(results.shape[-1]):
            results[..., q] += raw_predictor
        obs += raw_predictor

    # Save results to file
    # TODO: Create a gridded file output
    ofilename = "test.nc"
    T, _, Y, X, P = results.shape
    """
    coords = {"x": np.arange(X).astype(np.float32), "y": np.arange(Y).astype(np.float32), "time": loader.times[0] + loader.leadtimes}
    coords["level"] = [0.1,0.5,0.9]
    vars = dict()
    vars["air_temperature_2m"] = (("time", "y", "x", "level"), results[:, 0, :, :, :], {"units":
        "K"})
    dataset = xr.Dataset(vars, coords=coords)
    dataset["x"].attrs["units"] = "m"
    dataset["y"].attrs["units"] = "m"
    dataset["time"].attrs["units"] = "seconds since 1970-01-01T00:00:00 +00:00"
    dataset.to_netcdf(ofilename)
    """

    with netCDF4.Dataset(ofilename, "w") as file:
        file.createDimension("time")
        file.createDimension("x", X)
        file.createDimension("y", Y)
        file.createDimension("level", P)

        var = file.createVariable("time", "f8", ("time",))
        var[:] = loader.times[0] + loader.leadtimes
        var.units = "seconds since 1970-01-01T00:00:00 +00:00"

        var = file.createVariable("y", "f4", ("y",))
        var[:] = np.arange(Y).astype(np.float32)
        var.units = "m"

        var = file.createVariable("x", "f4", ("x",))
        var[:] = np.arange(X).astype(np.float32)
        var.units = "m"

        var = file.createVariable("level", "f4", ("level",))
        var[:] = np.array(quantiles).astype(np.float32)
        var.units = "1"

        var = file.createVariable("observation", "f4", ("time", "y", "x"))
        var[:] = obs
        var.units = "K"

        output = np.moveaxis(results, 4, 2)[:, 0, ...].astype(np.float32)
        T, Q, Y, X = output.shape
        fcst = output[:, Q // 2, ...]

        var = file.createVariable("air_temperature_2m", "f4", ("time", "level", "y", "x"))
        var[:] = output
        var.units = "K"

        var = file.createVariable("raw", "f4", ("time", "y", "x"))
        var[:] = raw_predictor
        var.units = "K"

        var = file.createVariable("error", "f4", ("time", "y", "x"))
        var[:] = obs - fcst
        var.units = "K"
        print(np.nanmean(var[:]))

        if len(quantiles) >= 2:
            var = file.createVariable("spread", "f4", ("time", "y", "x"))
            var[:] = (output[:, -1, :, :] - output[:, 0, :, :]).astype(np.float32)
            var.units = "K"

