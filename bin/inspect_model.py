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
    parser.add_argument('-l', help='Layer names. If not provided, show a list of available layers.', dest='layer_names', nargs="*")
    parser.add_argument('-i', default=0, type=int, help='Index into dataset', dest='index')
    parser.add_argument('-o', help='Output image filename', dest='ofilename')
    parser.add_argument('-s', default=[], help='What to show', dest='show', choices=['input', 'target', 'output'], nargs='*')
    parser.add_argument('-fs', default=[20, 10], type=int, help='Image size in inches (width, height)', dest="image_size", nargs=2)
    parser.add_argument('-mf', type=int, help="Max number of features", dest="max_features")
    parser.add_argument('-sp', type=int, help="Subplot numbers", dest="subplot_size", nargs=2)
    parser.add_argument('-rl', help="One row for each layer?", dest="row_per_layer", action="store_true")

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
    # config["loader"]["x_range"] = [215, 1751]
    # config["loader"]["y_range"] = [50, 2098]
    # config["loader"]["x_range"] = [215, 215 + 512*2]
    # config["loader"]["y_range"] = [50, 50 + 512*2]
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
    weights = model.load_weights(input_checkpoint_filepath).expect_partial()

    dataset = loader.get_dataset()

    layer_names = [layer.name for layer in model.layers]
    if len(args.layer_names) == 0:
        print(layer_names)
        return

    for layer_name in args.layer_names:
        if layer_name not in layer_names:
            print(layer_names)
            raise ValueError(f"Layer {layer_name} does not exist")

    fields = dict()  # Name, 2D field to plot
    cmap = "RdBu_r"

    for i, (predictors, targets) in enumerate(dataset):
        if i < args.index:
            continue

        def trim(ar):
            ar = ar[0, 0, :, :]
            # if ar.shape[0] >= 32:
            #     ar = ar[6:-6, 6:-6]
            return ar
        predictors = predictors.numpy()
        # print(predictors.shape)
        # for p in range(predictors.shape[-1]):
        #     print(loader.predictor_names[p], np.mean(predictors[..., p]))
        # predictors[..., :] = 0

        aspect = args.image_size[0] / args.image_size[1]

        # Add targets
        if "input" in args.show:
            for p in range(predictors.shape[-1]):
                fields[loader.predictor_names[p]] = trim(predictors[..., p])
        if "output" in args.show:
            q = model(predictors)
            for p in range(q.shape[-1]):
                fields["Output %d" % p] = trim(q[..., p])
        if "target" in args.show:
            fields["Target"] = trim(targets[..., 0])

        for layer_name in args.layer_names:
            intermediate_model = keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
            q = intermediate_model(predictors)
            Q = q.shape[-1]
            for p in range(Q):
                if args.max_features is not None and p >= args.max_features:
                    continue
                values = trim(q[..., p])
                if len(args.layers) > 1:
                    name = f"{layer_name}: {p:d}"
                else:
                    name = f"Feature {p:d}"
                fields[name] = values

        N = len(fields)
        print(args.subplot_size)
        if args.subplot_size is not None:
            X, Y = args.subplot_size
            if X * Y < N:
                raise RuntimeError(f"Subplot size {X, Y} cannot contain {N} subplots")
        else:
            X = int(np.ceil(np.sqrt(N * aspect)))
            Y = int(np.ceil(N / X))
        # print(N, X, Y)
        for i, (name, field) in enumerate(fields.items()):
            mpl.subplot(Y, X, i + 1)
            vmin = np.percentile(field, 10)
            vmax = np.percentile(field, 90)
            mpl.pcolormesh(field, cmap=cmap, vmin=vmin, vmax=vmax)
            print(name, np.min(field), np.max(field))
            mpl.gca().axis('off')
            mpl.gca().set_aspect(1)
            mpl.title(name)

        # mpl.gcf().suptitle(f"Layer: {args.layer_name}")
        if args.ofilename is not None:
            mpl.gcf().set_size_inches(*args.image_size)
            mpl.savefig(args.ofilename, bbox_inches='tight', dpi=300)
        else:
            mpl.show()
        # maelstrom.plot.plot_model(model, predictors, targets)
        break

if __name__ == "__main__":
    main()
