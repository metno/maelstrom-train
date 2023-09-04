import os
import sys
import numpy as np
import matplotlib.pylab as mpl
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras

import maelstrom
import maelstrom.__main__


def main():
    parser = argparse.ArgumentParser(description='This script plots the state of a network at a given layer')
    parser.add_argument('folder', help='Folder with maelstrom-train results')
    parser.add_argument('-f', default="data/air_temperature/5GB/20200301T03Z.nc", help='Use this input NetCDF file for simulation', dest="file")
    parser.add_argument('-l', help='Layer name. If not provided, show a list of available layers.', dest='layer_name')
    parser.add_argument('-i', default=0, type=int, help='Index into dataset', dest='index')
    parser.add_argument('-o', help='Output image filename', dest='ofilename')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    config_filename = f"{args.folder}/config.yml"
    config = maelstrom.load_yaml(config_filename)

    for key in ["loader_validation", "loader_test"]:
        if key in config:
            del config[key]
    config["loader"]["filenames"] = [args.file]
    # config["loader"]["limit_leadtimes"] = [0]
    if "patch_size" in config["loader"]:
        del config["loader"]["patch_size"]
    quantiles = config["output"]["quantiles"]
    num_outputs = len(quantiles)
    loader, loader_val, loader_test = maelstrom.__main__.get_loaders(config, False)

    if "name" in config["model"]:
        model_name = config["model"]["name"]
    else:
        model_name = config["model"]["type"].lower()
        config["models"] = [config["model"]]
    model, model_config = maelstrom.__main__.get_model(
        loader,
        num_outputs,
        config["models"],
        model_name,
        False,
    )
    input_checkpoint_filepath = args.folder + "/checkpoint"
    model.load_weights(input_checkpoint_filepath).expect_partial()

    dataset = loader.get_dataset()

    layer_names = [layer.name for layer in model.layers]
    if args.layer_name is None:
        print(layer_names)
        return

    if args.layer_name not in layer_names:
        print(layer_names)
        raise ValueError("Layer does not exist")

    fields = dict()  # Name, 2D field to plot

    for i, (predictors, targets) in enumerate(dataset):
        if i < args.index:
            continue
        intermediate_model = keras.Model(inputs=model.input, outputs=model.get_layer(args.layer_name).output)
        q = intermediate_model(predictors)
        import matplotlib.pylab as mpl
        Q = q.shape[-1]
        N = Q + 1
        X = int(np.ceil(np.sqrt(N)))
        Y = int(np.ceil(N / X))
        count = 0
        mpl.subplot(X, Y, 1)
        cmap = "RdBu_r"
        mpl.pcolormesh(targets[0, 0, :, :, 0], cmap=cmap)
        mpl.title("Target")
        count += 1
        for i in range(Q):
            values = q[0, 0, :, :, i]
            if values.shape[0] >= 32:
                values = values[6:-6, 6:-6]
            mpl.subplot(X, Y, i + 1 + count)
            print(np.min(values), np.max(values))
            mpl.pcolormesh(values, cmap=cmap)
        mpl.gcf().suptitle(f"Layer: {args.layer_name}")
        if args.ofilename is not None:
            mpl.gcf().set_size_inches(10, 10)
            mpl.savefig(args.ofilename, dpi=300)
        else:
            mpl.show()
        # maelstrom.plot.plot_model(model, predictors, targets)
        break

if __name__ == "__main__":
    main()
