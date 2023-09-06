import os
import sys
import numpy as np
import argparse
import maelstrom
import maelstrom.__main__


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('folder', help='')
    parser.add_argument('-f', default="data/air_temperature/5GB/20200301T03Z.nc", help='Use this input NetCDF file for simulation', dest="file")

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
    model.load_weights(input_checkpoint_filepath).expect_partial()

    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            print(weights)

if __name__ == "__main__":
    main()
