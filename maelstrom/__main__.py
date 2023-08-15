import argparse
import copy
import datetime
import glob
import json
import os
import sys
import time
import math

import numpy as np
import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras

try:
    from deep500.utils import timer_tf as timer
    do_deep500 = True
except Exception as e:
    print("Cannot load deep500")
    do_deep500 = False
do_deep500 = False

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


tf.random.set_seed(1000)
np.random.seed(1000)


import maelstrom

with_horovod = maelstromm.check_horovod()
if with_horovod:
    # Import it 
    print("Running with horovod")
    import horovod.tensorflow as hvd

def main():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument( "--config", type=maelstrom.load_yaml, help="Configuration file containing output paths, etc", required=True, nargs="*",)
    parser.add_argument( "-j", type=int, help="Number of threads to train with", dest="num_threads",)
    parser.add_argument( "--hardware", default="gpu", help="What hardware to run on?", choices=["cpu", "gpu"],)
    parser.add_argument( "-m", help="Only run these models", dest="subset_models", nargs="*", required=True,)
    parser.add_argument( "-w", help="Print weights of the model", dest="print_weights", action="store_true",)
    parser.add_argument( "-o", default="results/%N_%T", help="Output folder", dest="output_folder",)
    parser.add_argument( "--seed", type=int, help="Random seed",)
    parser.add_argument( "--load_weights", help="Initialize with model weights found in this directory")
    parser.add_argument( "--train", default=1, type=int, help="Enable training?", dest="do_train", choices=[0, 1])
    parser.add_argument( "--test", default=1, type=int, help="Enable testing?", dest="do_test", choices=[0, 1])
    args = parser.parse_args()
    # fmt: on

    if not args.do_train and not args.load_weights:
        raise ValueError("--load_weights required with --train=0")

    if args.seed is not None:
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)

    # Deal with GPUs
    main_process = True
    if args.hardware == "cpu":
        # Force CPU usage, even if GPUs are available
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        print("Num GPUs Available: ", len(gpus))
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if with_horovod:
            hvd.init()
            main_process = hvd.rank() == 0

            if main_process:
                print("Num GPUs Available: ", len(gpus))
            if len(gpus) > 1:
                tf.config.experimental.set_visible_devices(
                    gpus[hvd.local_rank()], "GPU"
                )
            print("Current rank", hvd.local_rank())

    config = maelstrom.merge_configs(args.config)

    loader, loader_val, loader_test = get_loaders(config)

    if config["loss"]["type"] in ["mae_prob"] and loader.num_targets == 1:
        raise Exception(
            f"Cannot use loss={config['loss']['type']} when loader only has 1 output"
        )

    if main_process:
        print("Available devices")
        print(tf.config.list_physical_devices())

        print("Training loader")
        print(loader)
        if loader_val != loader:
            print("Validation loader")
            print(loader_val)
        if loader_test not in [loader, loader_val]:
            print("Test loader")
            print(loader_test)
        print(loader == loader_val)
        print(loader_val == loader_test)
        print(loader == loader_test)

    if 0 and "tensorflow" in config and args.num_threads != 1:
        tf.config.threading.set_intra_op_parallelism_threads(
            config["tensorflow"]["num_threads"]
        )
        tf.config.threading.set_inter_op_parallelism_threads(
            config["tensorflow"]["num_threads"]
        )
    if 0 and args.num_threads is not None and args.num_threads > 1:
        tf.config.threading.set_intra_op_parallelism_threads(args.num_threads)
        tf.config.threading.set_inter_op_parallelism_threads(args.num_threads)

    # Model training
    epochs = config["training"]["num_epochs"]

    if with_horovod:
        # print("SHARDING", hvd.size(), hvd.local_rank())
        # dataset = dataset.shard(num_shards=hvd.size(), index=hvd.local_rank())
        # print(type(dataset))
        dataset = loader.get_dataset(True, repeat=epochs, shard_size=hvd.size(), shard_index=hvd.local_rank())
    else:
        dataset = loader.get_dataset(True, repeat=epochs)

    do_validation = loader_val is not None and loader_val != loader
    dataset_val = dataset
    if do_validation:
        # Don't use batches for the test dataset
        # Otherwise the model.predict doesn't give a full output for some reason...
        dataset_val = loader_val.get_dataset()

    quantiles = config["output"]["quantiles"]
    num_outputs = len(quantiles)

    loss = maelstrom.loss.get(config["loss"], quantiles)
    metrics = []
    if loss not in [maelstrom.loss.mae, maelstrom.loss.mae_prob]:
        metrics = [
            maelstrom.loss.mae
        ]  # [maelstrom.loss.meanfcst, maelstrom.loss.meanobs]

    models = get_models(
        loader,
        num_outputs,
        config["models"],
        args.subset_models,
        with_horovod,
    )

    if len(models) == 0:
        raise Exception("No models selected")

    for model_name, model_config in models.items():
        start_time = time.time()
        dt = datetime.datetime.utcfromtimestamp(int(start_time))
        date, hour = maelstrom.util.unixtime_to_date(start_time)
        minute = dt.minute
        second = dt.second
        curr_time = f"{date:08d}T{hour:02d}{minute:02d}{second:02d}Z"
        output_folder = args.output_folder.replace("%T", curr_time).replace(
            "%N", model_name
        )

        model = model_config["model"]
        optimizer = maelstrom.optimizer.get(**config["training"]["optimizer"])
        if with_horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, backward_passes_per_step=1,
                    average_aggregated_gradients=True)
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            experimental_run_tf_function=False,
        )

        if main_process:
            print(model.summary())

            config_logger = maelstrom.logger.Logger(f"{output_folder}/config.yml")
            config_logger.add(None, config)
            config_logger.write()

            logger = maelstrom.logger.Logger(f"{output_folder}/log.txt")

        model_description = model.description()
        model_description.update(model_config["settings"])
        num_trainable_weights = int(np.sum([K.count_params(w) for w in model.trainable_weights]))
        num_non_trainable_weights = int(np.sum([K.count_params(w) for w in model.non_trainable_weights]))
        model_description["Num traininable parameters"] = num_trainable_weights
        model_description["Num non-trainable parameters"] = num_non_trainable_weights
        if main_process:
            logger.add("Model", model_description)
            logger.add("Dataset", loader.description())
            logger.add("Timing", "Start time", int(start_time))
            # logger.add("Scores")
            for section in config.keys():
                if section not in ["models"]:
                    logger.add("Config", section.capitalize(), config[section])

            timing_callback = maelstrom.callback.Timing(logger)
        callbacks = list()
        if args.do_train:
            validation_frequency = get_validation_frequency(config, loader)
            if with_horovod:
                callbacks += [hvd.keras.callbacks.BroadcastGlobalVariablesCallback(0)]
                callbacks += [hvd.keras.callbacks.MetricAverageCallback()]
                validation_frequency = math.ceil(validation_frequency / hvd.size())
            if main_process:
                print(f"Validation frequency: {validation_frequency}")
                callbacks += [
                    maelstrom.callback.Convergence(f"{output_folder}/{model_name}_loss.txt", True, True, True)
                ]
                callbacks += [timing_callback]

                checkpoint_metric = 'loss'
                if do_validation and main_process:
                    """
                    callbacks += [
                        maelstrom.callback.Validation(
                            f"{output_folder}/{model_name}_val.txt",
                            model,
                            dataset_val,
                            validation_frequency,
                            logger,
                        )
                    ]
                    """
                    checkpoint_metric = 'val_loss'

                # Note that the ModelCheckpoint callback must be added after the validation
                # callback, otherwise val_loss will not be recorded when the checkpoint callback is
                # run.
                if main_process:
                    checkpoint_filepath = f"{output_folder}/checkpoint"

                    # NOTE: save_freq must be "epoch", otherwise this callback gets run before the
                    # end of the epoch, which is when the validation is computed.
                    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath=checkpoint_filepath,
                        save_weights_only=True,
                        save_freq="epoch", # validation_frequency,
                        monitor=checkpoint_metric,
                        verbose=1,
                        mode='min',
                        save_best_only=True
                    )
                    callbacks += [model_checkpoint_callback]
                if 0 and num_trainable_weights < 1e5:
                    callbacks += [
                        maelstrom.callback.WeightsCallback(
                            model, filename=f"{output_folder}/weights.nc"
                        )
                    ]
                else:
                    print(f"Too many trainable weights {num_trainable_weights}, not writing them out")
                if "early_stopping" in config["training"]:
                    callbacks += [
                        tf.keras.callbacks.EarlyStopping(
                            monitor="loss", **config["training"]["early_stopping"]
                        )
                    ]

        s_time = time.time()

        # This is the keras way of running validation. However, now we do validation via a
        # validation callback above instead.
        kwargs = {}
        if with_horovod:
            keras_epochs = int(epochs * loader.num_batches // hvd.size() // (validation_frequency))
            kwargs["steps_per_epoch"] = validation_frequency
            kwargs["verbose"] = main_process
        else:
            keras_epochs = int(epochs * loader.num_batches // (validation_frequency))
            kwargs["steps_per_epoch"] = validation_frequency

        if do_validation:
            kwargs = {"validation_data": dataset_val}

        if main_process:
            logger.add("Timing", "Training", "Start_time", int(time.time()))
        curr_args = {k: v for k, v in config["training"]["trainer"].items()}
        curr_args["model"] = model
        curr_args["optimizer"] = optimizer
        curr_args["loss"] = loss
        trainer = maelstrom.trainer.get(**curr_args)

        if main_process and do_deep500:
            tmr = timer.CPUGPUTimer()
            callbacks += [timer.TimerCallback(tmr, gpu=True)]
        # Note: We could add a check for num_trainable_parameters > 0
        # num_trainable_parameters = np.sum([K.count_params(w) for w in model.trainable_weights])
        # and skip training (this could be the raw model for example). However, we might still want
        # to run validation using the raw model, therefore we will still try to train it
        if args.load_weights is not None and main_process:
            input_checkpoint_filepath = args.load_weights + "/checkpoint"
            print("Loading weights from {input_checkpoint_filepath}")
            model.load_weights(input_checkpoint_filepath)
        if args.do_train:
            print("\n### Training ###")
            maelstrom.util.print_memory_usage()
            # history = trainer.fit(dataset, epochs=epochs, callbacks=callbacks, **kwargs)
            # NOTE: When keras run with a generator with unknown length, we need to tell keras how
            # long it is. DO this by specifying steps_per_epoch, and then making dataset repeat
            # itself.
            ss_time = time.time()
            if main_process:
                print(epochs, keras_epochs, validation_frequency)
                print("Callbacks:")
                for callback in callbacks:
                    print("   ", callback)
            history = trainer.fit(dataset, epochs=keras_epochs, callbacks=callbacks,
                    **kwargs)
            print(f"Training time: {time.time() - ss_time}")
            if main_process and do_deep500:
                tmr.complete_all()
                tmr.print_all_time_stats()
            # TODO: Enable this
            # if main_process:
            #     model.load_weights(checkpoint_filepath)
        else:
            history = None

        if main_process:
            logger.add("Timing", "Training", "end_time", int(time.time()))

            if args.do_test:
                print(f"\n### Testing ###")
                maelstrom.util.print_memory_usage()
                s_time = time.time()
                eval_results = testing(
                    config["evaluators"],
                    loader_test,
                    quantiles,
                    trainer,
                    output_folder,
                    model_name,
                )
                logger.add("Timing", "Testing", "total_time", time.time() - s_time)
                for k, v in eval_results.items():
                    logger.add("Scores", k, v)
                print(eval_results)
                maelstrom.util.print_memory_usage()

            # Write loader statistics
            for name, curr_time in loader.timing.items():
                logger.add("Timing", "Loader", name, curr_time)
            logger.add("Timing", "Loader", "num_files_read", loader.count_reads)

            if loader_val is not None and loader_val != loader:
                for name, curr_time in loader_val.timing.items():
                    logger.add("Timing", "Validation loader", name, curr_time)
                logger.add(
                    "Timing",
                    "Validation loader",
                    "num_files_read",
                    loader_val.count_reads,
                )

            if loader_test != loader:
                for name, curr_time in loader_test.timing.items():
                    logger.add("Timing", "Test loader", name, curr_time)
                logger.add(
                    "Timing", "Test loader", "num_files_read", loader_test.count_reads
                )

            if args.print_weights:
                for layer in model.layers:
                    print(layer.get_weights())

            # Add final information to logger
            if history is not None:
                for key in history.history.keys():
                    logger.add("Scores", key, history.history[key][-1])
            logger.add("Timing", "End time", int(time.time()))
            logger.add("Timing", "Total", time.time() - s_time)
            logger.write()


def get_loaders(config):
    """Initializes the loaders needed

    Args:
        config (dict): Dictionary with configuration

    Returns:
        loader: Loader for training
        loader_val: Loader for validation (can be none, if no validation is present in config)
        loader_test: Loader for testing: Defaults to loader_val, if no testing is set up
    """

    if "loader" not in config:
        raise ValueError("'loader' section not specified in YAML")

    loader_config = config["loader"]
    loader = maelstrom.loader.Loader.from_config(loader_config)
    loader_test = loader
    loader_val = None

    if "loader_validation" in config:
        loader_val = maelstrom.loader.Loader.from_config(config["loader_validation"])
        loader_test = loader_val

    if "loader_test" in config:
        loader_test = maelstrom.loader.Loader.from_config(config["loader_test"])

    return loader, loader_val, loader_test


def get_models(loader, num_outputs, configs, subset_models=None, multi=False):
    input_shape = [1] + loader.predictor_shape[1:]
    if multi:
        gpus = tf.config.list_logical_devices("GPU")
        strategy = tf.distribute.MirroredStrategy(gpus[0:2])
    models = dict()
    for config in configs:
        if "name" not in config:
            name = config["type"]
        else:
            name = config["name"]
        if "disabled" in config:
            continue
        args = {k: v for k, v in config.items() if k not in ["name", "disabled"]}

        if subset_models is not None:
            models_lower_name = [n.lower() for n in subset_models]
            new_models = list()
            if name.lower() not in models_lower_name:
                continue

        if args["type"].lower() == "selectpredictor":
            args["indices"] = list()
            for predictor_name in args["predictor_names"]:
                args["indices"] += [loader.predictor_names.index(predictor_name)]
            del args["predictor_names"]

        if multi:
            with strategy.scope():
                model = maelstrom.models.get(input_shape, num_outputs, **args)
        else:
            model = maelstrom.models.get(input_shape, num_outputs, **args)
        models[name] = {"model": model, "settings": args}
    return models


def testing(config, loader, quantiles, trainer, output_folder, model_name):
    """Runs model testing and returns final test loss

    Args:
        config: List of testing configurations
        loader (maelstrom.loader): Data loader
        quantiles (list): What quantiles do the output represent
        trainer (maelstrom.trainer): Thing that does the training
        output_folder (str): Where to write testing results to

    Returns:
        dict: Dictionary of testing loss value
    """

    s_time = time.time()
    results = dict()
    model = trainer.model
    loss = trainer.loss

    evaluators = get_evaluators(
        config, loader, model, loss, quantiles, output_folder, model_name
    )

    # Which input predictor is the raw forecast?
    if loader.predict_diff:
        Ip = loader.predictor_names.index("air_temperature_2m")
    total_loss = 0
    count = 0

    if 1:
        """Using keras.predict and prefetching

        The advantage is that prefetching will work automatically
        However, prediction is slow when patching is on

        This code doesn't currently work, since for a given batch we don't know what forecast
        reference time we are at. Ideally, the data should be batched across patches.
        """
        dataset = loader.get_dataset()
        forecast_reference_times = loader.times
        samples_per_file = loader.num_samples_per_file * loader.num_patches_per_sample
        num_files = loader.num_files

        ss_time = time.time()
        output = None

        # This for loop iterates over each sample, not each file
        for batch, (fcst, targets) in enumerate(dataset):
            assert fcst.shape[0] == 1
            targets = np.copy(targets)

            first_batch_in_file = batch % samples_per_file == 0
            last_batch_in_file = (batch + 1) % samples_per_file == 0
            batch_in_file = batch % samples_per_file

            if first_batch_in_file:
                file_index = batch // samples_per_file
                forecast_reference_time = forecast_reference_times[
                    batch // samples_per_file
                ]
                date, hour = maelstrom.util.unixtime_to_date(forecast_reference_time)
                print(f"Processing {date:08d}T{hour:02d}Z ({file_index+1}/{num_files})")

            curr_output = trainer.predict_on_batch(fcst)
            if output is None:
                new_shape = [samples_per_file] + list(curr_output.shape[1:])
                output = np.nan * np.zeros(new_shape, np.float32)

            num_outputs = curr_output.shape[-1]

            if loader.predict_diff:
                targets[..., 0] += fcst[..., Ip]
                for p in range(num_outputs):
                    curr_output[..., p] += fcst[..., Ip]
            output[batch_in_file, ...] = curr_output[0, ...]

            curr_loss = float(loss(targets, curr_output))
            total_loss += curr_loss

            if last_batch_in_file:
                # Only run the evaluation when we have finished all batches in the file
                for evaluator in evaluators:
                    evaluator.evaluate(forecast_reference_time, output, targets)
                print("   Time: %.2f" % (time.time() - ss_time))
                maelstrom.util.print_memory_usage("   ")
                ss_time = time.time()
                output = None

            count += 1

        total_loss /= count

        for evaluator in evaluators:
            evaluator.close()

        results["test_loss"] = total_loss
    elif 1:
        """Calling predict on each sample"""
        num = len(loader)
        for i in range(num):
            ss_time = time.time()
            sss_time = time.time()
            forecast_reference_time = loader.times[i]
            date, hour = maelstrom.util.unixtime_to_date(forecast_reference_time)
            print(f"Processing {date:08d}T{hour:02d}Z ({i+1}/{num})")
            fcst, targets = loader[i]
            targets = np.copy(targets)
            maelstrom.util.print_memory_usage("  ")
            print("   Data extraction: %.2f s" % (time.time() - sss_time))
            sss_time = time.time()

            output = None

            # Run prediction on one sample at a time, for memory reasons
            sss_time = time.time()
            for s in range(fcst.shape[0]):
                # NOTE: Use predict_on_batch, since predict has a memory leak when the function is
                # iterated across: https://github.com/tensorflow/tensorflow/issues/44711
                curr_output = trainer.predict_on_batch(
                    tf.expand_dims(fcst[s, ...], axis=0)
                )
                if output is None:
                    new_shape = [fcst.shape[0]] + list(curr_output.shape[1:])
                    output = np.nan * np.zeros(new_shape, np.float32)

                output[s, ...] = curr_output[0, ...]
                num_outputs = curr_output.shape[-1]

                curr_loss = float(loss(targets[s, ...], curr_output))
                total_loss += curr_loss
                count += 1
                if loader.predict_diff:
                    targets[s, ..., 0] += fcst[s, ..., Ip]
                    for p in range(num_outputs):
                        output[s, ..., p] += fcst[s, ..., Ip]
            print("   Predicting: %.2f s" % (time.time() - sss_time))

            sss_time = time.time()
            for evaluator in evaluators:
                evaluator.evaluate(forecast_reference_time, output, targets)
            print("   Evaluating: %.2f" % (time.time() - sss_time))
            print("   Total: %.2f" % (time.time() - ss_time))

        total_loss /= count

        for evaluator in evaluators:
            evaluator.close()

        results["test_loss"] = total_loss
    else:
        callbacks = [maelstrom.callback.Testing()]

        # Which input predictor is the raw forecast?
        Ip = loader.predictor_names.index("air_temperature_2m")

        num = len(loader)
        dataset = loader.get_dataset(1, 1)

        trainer.predict(dataset, callbacks=callbacks)
    print("Total time", time.time() - s_time)
    return results


def get_evaluators(config, loader, model, loss, quantiles, output_folder, model_name):
    if config is None:
        return []

    evaluators = list()
    leadtimes = loader.leadtimes
    for eval_config in config:
        eval_type = eval_config["type"].lower()
        if eval_type == "verif":
            sampling = 1
            if "sampling" in eval_config:
                sampling = eval_config["sampling"]
            points = loader.get_grid_resampled(sampling).to_points()
            attributes = loader.description()
            attributes.update(model.description())
            for k, v in attributes.items():
                attributes[k] = str(v)
            # The verif file should ideally be called the model name. That makes it more efficient
            # to use with verif when comparing multiple runs
            filename = f"{output_folder}/{model_name}.nc"
            evaluator = maelstrom.evaluator.Verif(
                filename, leadtimes, points, quantiles, attributes
            )
        elif eval_type == "aggregator":
            filename = f"{output_folder}/{model_name}_test.txt"
            evaluator = maelstrom.evaluator.Aggregator(filename, leadtimes, loss)
        else:
            raise ValueError(f"Unknown validation type {eval_type}")
        evaluators += [evaluator]
    return evaluators


def get_validation_frequency(config, loader):
    validation_frequency = loader.num_patches_per_file
    if "validation_frequency" in config["training"]:
        words = config["training"]["validation_frequency"].split(" ")
        if len(words) != 2:
            raise ValueError(
                "validation_frequency must be in the form <value> <unit>"
            )
        freq, freq_units = words
        freq = int(freq)
        if freq_units == "epoch":
            # if freq == 1:
            #     validation_frequency = None
            # else:
            validation_frequency = int(
                loader.num_samples * freq /
                config["loader"]["batch_size"]
            )
        elif freq_units == "file":
            validation_frequency = loader.num_samples_per_file * freq
        elif freq_units == "batch":
            validation_frequency = freq
        else:
            raise ValueError(
                f"Unknown validation frequency units '{validation_frequency}'"
            )
    return validation_frequency


if __name__ == "__main__":
    main()
