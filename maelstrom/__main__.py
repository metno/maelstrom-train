import argparse
import copy
import datetime
import glob
import json
import numpy as np
import os
import sys
import time
import math
import tqdm
import socket

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras

tf.random.set_seed(1000)
np.random.seed(1000)


import maelstrom

with_horovod = maelstrom.check_horovod()
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
    parser.add_argument( "-m", help="Run this model", dest="model", required=True)
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

    if with_horovod:
        num_processes = hvd.size()
    else:
        num_processes = 1

    config = maelstrom.merge_configs(args.config)

    loader, loader_val, loader_test = get_loaders(config, with_horovod)

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
    dataset = loader.get_dataset(True, repeat=epochs)

    do_validation = loader_val is not None and loader_val != loader
    dataset_val = dataset
    if do_validation:
        # Don't use batches for the test dataset
        # Otherwise the model.predict doesn't give a full output for some reason...
        dataset_val = loader_val.get_dataset()
        # Check that loaders have the same predictors
        if not loader.check_compatibility(loader_val):
            raise Exception("Loaders do not have the same predictors in the same order")

    quantiles = config["output"]["quantiles"]
    num_outputs = len(quantiles)

    validation_frequency = loader.get_frequency(config["training"]["validation_frequency"], with_horovod)
    if main_process:
        print(f"validation frequency: {validation_frequency} batches")

    loss = maelstrom.loss.get(config["loss"], quantiles)
    metrics = []
    if loss not in [maelstrom.loss.mae, maelstrom.loss.mae_prob]:
        # within_loss = maelstrom.loss.get({"type": "within"}, quantiles)
        metrics = [
            # within_loss
            # maelstrom.loss.mae
        ]  # [maelstrom.loss.meanfcst, maelstrom.loss.meanobs]

    model_name = args.model
    model, model_config = get_model(
        loader,
        num_outputs,
        config["models"],
        args.model,
        with_horovod,
    )

    start_time = time.time()
    dt = datetime.datetime.utcfromtimestamp(int(start_time))
    date, hour = maelstrom.util.unixtime_to_date(start_time)
    minute = dt.minute
    second = dt.second
    curr_time = f"{date:08d}T{hour:02d}{minute:02d}{second:02d}Z"
    output_folder = args.output_folder.replace("%T", curr_time).replace(
        "%N", model_name
    )

    optimizer = maelstrom.optimizer.get(validation_frequency, **config["training"]["optimizer"])
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
        config_to_save = copy.copy(config)
        config_to_save["model"] = model_config
        del config_to_save["models"]
        config_logger.add(None, config_to_save)
        config_logger.write()

        logger = maelstrom.logger.Logger(f"{output_folder}/log.txt")

    model_description = model.description()
    model_description.update(model_config)
    num_trainable_weights = int(np.sum([K.count_params(w) for w in model.trainable_weights]))
    num_non_trainable_weights = int(np.sum([K.count_params(w) for w in model.non_trainable_weights]))
    model_description["Num trainable parameters"] = num_trainable_weights
    model_description["Num non-trainable parameters"] = num_non_trainable_weights
    if main_process:
        logger.add("Model", model_description)
        logger.add("Model", "Config", model_config)
        logger.add("Dataset", loader.description())
        logger.add("Timing", "Start time", int(start_time))
        # logger.add("Scores")
        for section in config.keys():
            if section not in ["models"]:
                logger.add("Config", section.capitalize(), config[section])

    # Set up callbacks
    callbacks = list()
    if args.do_train:
        if with_horovod:
            callbacks += [hvd.keras.callbacks.BroadcastGlobalVariablesCallback(0)]
            callbacks += [hvd.keras.callbacks.MetricAverageCallback()]
        if main_process:
            callbacks += maelstrom.callback.get_from_config(config, logger, model, output_folder)

    s_time = time.time()

    # This is the keras way of running validation. However, now we do validation via a
    # validation callback above instead.
    kwargs = {}
    if do_validation:
        kwargs["validation_data"] = dataset_val
        keras_epochs = int(epochs * loader.num_batches // (validation_frequency))
        if with_horovod:
            kwargs["verbose"] = main_process
        kwargs["steps_per_epoch"] = validation_frequency
    else:
        keras_epochs = epochs
        kwargs["steps_per_epoch"] = loader.num_batches

    # if do_validation:
    #     keras_epochs = int(epochs * loader.num_batches // (validation_frequency))
    # kwargs["verbose"] = main_process
    if main_process:
        print("keras_epochs:", keras_epochs)

    if main_process:
        logger.add("Timing", "Training", "Start_time", int(time.time()))
    curr_args = {k: v for k, v in config["training"]["trainer"].items()}
    curr_args["model"] = model
    curr_args["optimizer"] = optimizer
    curr_args["loss"] = loss
    trainer = maelstrom.trainer.get(**curr_args)

    # Note: We could add a check for num_trainable_parameters > 0
    # num_trainable_parameters = np.sum([K.count_params(w) for w in model.trainable_weights])
    # and skip training (this could be the raw model for example). However, we might still want
    # to run validation using the raw model, therefore we will still try to train it
    if args.load_weights is not None and main_process:
        input_checkpoint_filepath = args.load_weights + "/checkpoint"
        print(f"Loading weights from {input_checkpoint_filepath}")
        model.load_weights(input_checkpoint_filepath).expect_partial()

    if main_process:
        print("\nRun configuration:")
        print(f"   Training size: {loader.size_gb * num_processes:.2f} GB")
        if do_validation:
            print(f"   Validation size: {loader_val.size_gb * num_processes:.2f} GB")
        print(f"   Number of processes: {num_processes}")
        print(f"   Batch size: {loader.batch_size}")
        print(f"   Patch size: {loader.patch_size}")
        print(f"   Epochs: {epochs}")
        print(f"   Validation frequency: {validation_frequency} batches")
        hostname = socket.gethostname().split('.')[0]
        print(f"   Hostname: {hostname}")
        # TODO: Put dataset information and run information

        print("\nModel configuration:")
        print(f"   Model name: {model_name}")
        for k,v in model_config.items():
            print(f"   {k}: {v}")
        print(f"   Num trainable parameters: {num_trainable_weights}")

    if args.do_train:
        if main_process:
            print("\n### Training ###")
        maelstrom.util.print_memory_usage()
        if num_trainable_weights > 0:
            history = trainer.fit(dataset, epochs=keras_epochs, callbacks=callbacks, **kwargs)
            training_results = history.history
        else:
            # Horovod does not like training models without traininable parameters
            eval_callbacks = [hvd.keras.callbacks.MetricAverageCallback()]
            training_results = dict()
            training_results["loss"] = [trainer.evaluate(dataset, callbacks=eval_callbacks)]
            if do_validation:
                training_results["val_loss"] = [trainer.evaluate(dataset_val, callbacks=eval_callbacks)]

        if main_process:
            print("\nTraining results")
            training_time = time.time() - start_time
            print(f"   Training time: {training_time:.2f} s")
            print(f"   Training time per epoch: {training_time / epochs:.2f} s")
            performance = loader.size_gb * num_processes / (training_time / epochs)
            print(f"   Training performance: {performance:.2f} GB/s")

            loss = training_results["loss"]
            print(f"   Last loss: {loss[-1]:.4f}")
            print(f"   Best loss: {np.min(loss):.4f}")

            if "val_loss" in training_results:
                val_loss = training_results["val_loss"]
                print(f"   Last val loss: {val_loss[-1]:.4f}")
                print(f"   Best val loss: {np.min(val_loss):.4f}")

            for key in training_results.keys():
                logger.add("Scores", key, training_results[key][-1])
        # TODO: Enable this
        # if main_process:
        #     model.load_weights(checkpoint_filepath)

    if main_process:
        logger.add("Timing", "Training", "end_time", int(time.time()))

    if args.do_test:
        if not loader.check_compatibility(loader_test):
            raise Exception("Loaders do not have the same predictors in the same order")
        if main_process:
            print(f"\n### Testing ###")
            maelstrom.util.print_memory_usage()
        s_time = time.time()
        # history = trainer.evaluate(loader_test.get_dataset())
        test_loss = testing(
            config["evaluators"],
            loader_test,
            quantiles,
            trainer,
            output_folder,
            model_name,
            with_horovod,
        )
        # eval_results = history.history
        test_time = time.time() - s_time
        if main_process:
            logger.add("Timing", "Testing", "total_time", test_time)
            # for k, v in eval_results.items():
            #     logger.add("Scores", k, v)
            # test_loss = eval_results["test_loss"]
            logger.add("Scores", "test_loss", test_loss)
            print("\nTesting results")
            print(f"   Test size: {loader_test.size_gb * num_processes:.2f} GB")
            print(f"   Test time: {test_time:.2f} s")
            print(f"   Test loss: {test_loss:.4f}")
            performance = loader_test.size_gb * num_processes / test_time
            print(f"   Test performance: {performance:.2f} GB/s")

    if main_process:
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
        total_runtime = time.time() - start_time
        logger.add("Timing", "End time", int(time.time()))
        logger.add("Timing", "Total", total_runtime)
        logger.write()
        print("\nOverall information")
        print(f"   Total runtime: {total_runtime:.2f} s")
        maelstrom.util.print_gpu_usage("   Final GPU memory: ")
        maelstrom.util.print_cpu_usage("   Final CPU memory: ")

    if with_horovod:
        horovod.tensorflow.shutdown()


def get_loaders(config, with_horovod):
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
    loader = maelstrom.loader.Loader.from_config(loader_config, with_horovod)
    loader_test = loader
    loader_val = None

    if "loader_validation" in config:
        loader_val = maelstrom.loader.Loader.from_config(config["loader_validation"], with_horovod)
        loader_test = loader_val

    if "loader_test" in config:
        loader_test = maelstrom.loader.Loader.from_config(config["loader_test"], with_horovod)

    return loader, loader_val, loader_test


def get_model(loader, num_outputs, configs, model, multi=False):
    input_shape = loader.sample_predictor_shape
    if multi:
        gpus = tf.config.list_logical_devices("GPU")
        strategy = tf.distribute.MirroredStrategy(gpus[0:2])

    for config in configs:
        if "name" not in config:
            name = config["type"]
        else:
            name = config["name"]

        if name.lower() == model.lower():
            args = {k: v for k, v in config.items() if k not in ["name", "disabled"]}

            if args["type"].lower() in ["selectpredictor", "elevcorr"]:
                args["indices"] = list()
                for predictor_name in args["predictor_names"]:
                    args["indices"] += [loader.predictor_names.index(predictor_name)]
                del args["predictor_names"]

                if "predictor_name_altitude" in args:
                    args["index_altitude"] = loader.predictor_names.index(args["predictor_name_altitude"])
                    del args["predictor_name_altitude"]
                if "predictor_name_model_altitude" in args:
                    args["index_model_altitude"] = loader.predictor_names.index(args["predictor_name_model_altitude"])
                    del args["predictor_name_model_altitude"]

            if multi:
                with strategy.scope():
                    model = maelstrom.models.get(input_shape, num_outputs, **args)
            else:
                model = maelstrom.models.get(input_shape, num_outputs, **args)
            return model, args
    raise ValueError(f"Model {model} not defined in configuration file")


def testing(config, loader, quantiles, trainer, output_folder, model_name, with_horovod):
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
        config, loader, model, loss, quantiles, output_folder, model_name, with_horovod
    )

    # Which input predictor is the raw forecast?
    if loader.predict_diff:
        Ip = loader.predictor_names.index("air_temperature_2m")
    total_loss = 0
    count = 0

    if len(evaluators) == 0:
        """ Use keras build in predict. The disadvantage is that we can't evaluate results per
        leadtime """
        history = trainer.evaluate(loader.get_dataset())
        if maelstrom.util.is_list(history):
            total_loss = history[0]
        else:
            total_loss = history
    elif 1:
        """Using keras.predict and prefetching

        The advantage is that prefetching will work automatically
        However, prediction is slow when patching is on

        This code doesn't currently work, since for a given batch we don't know what forecast
        reference time we are at. Ideally, the data should be batched across patches.
        """
        dataset = loader.get_dataset()
        forecast_reference_times = loader.times
        samples_per_file = loader.num_samples_per_file
        num_files = loader.num_files

        ss_time = time.time()

        progbar = tf.keras.utils.Progbar(loader.num_batches)  # , stateful_metrics=['val_loss'])
        for batch, (bfcst, btargets) in enumerate(dataset):
            btargets = np.copy(btargets)
            bpred = trainer.predict_on_batch(bfcst)
            # print("Mean", batch, np.mean(bpred), np.mean(btargets), np.mean(bfcst))

            num_outputs = bpred.shape[-1]

            # Undo the prediction difference
            if loader.predict_diff:
                btargets[..., 0] += bfcst[..., Ip]
                for p in range(num_outputs):
                    bpred[..., p] += bfcst[..., Ip]

            curr_loss = float(loss(btargets, bpred))
            total_loss += curr_loss
            count += 1
            progbar.update(batch, [("loss", curr_loss)])

            for sample in range(bpred.shape[0]):
                forecast_reference_time = loader.get_time_from_batch(batch, sample)
                date, hour = maelstrom.util.unixtime_to_date(forecast_reference_time)
                for ileadtime in range(bpred.shape[1]):
                    leadtime = loader.get_leadtime_from_batch(batch, sample, ileadtime)

                    fcst = bfcst[sample, ileadtime, ...]
                    pred = bpred[sample, ileadtime, ...]
                    targets = btargets[sample, ileadtime, ...]

                    # print(f"Processing {date:08d}T{hour:02d}Z ({file_index+1}/{num_files})")

                    for evaluator in evaluators:
                        evaluator.evaluate(forecast_reference_time, leadtime, pred, targets)
            # print("   Time: %.2f" % (time.time() - ss_time))
            # maelstrom.util.print_memory_usage("   ")
            ss_time = time.time()

        total_loss /= count

        if with_horovod:
            total_loss_list = hvd.allgather_object(total_loss)
            total_loss = np.mean(total_loss_list)

            for evaluator in evaluators:
                evaluator.sync()

        for evaluator in evaluators:
            if not with_horovod or hvd.local_rank() == 0:
                evaluator.write()
            evaluator.close()

        # results["test_loss"] = total_loss
    elif 0:
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
    # return results
    return total_loss


def get_evaluators(config, loader, model, loss, quantiles, output_folder, model_name, with_horovod=False):
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


def get_validation_frequency(config, loader, with_horovod):
    validation_frequency = loader.num_batches
    if "validation_frequency" in config["training"]:
        words = config["training"]["validation_frequency"].split(" ")
        if len(words) != 2:
            raise ValueError(
                "validation_frequency must be in the form <value> <unit>"
            )
        freq, freq_units = words
        freq = int(freq)
        if freq_units == "epoch":
            validation_frequency = loader.num_batches * freq
        elif freq_units == "file":
            validation_frequency = loader.num_batches_per_file * freq
            # The convept of "file" needs to be handled differently when horovod processes several
            # files in parallel. The most intuitive would be that if we want validation every 36
            # files, and 4 are run in parallel, we should validated every 9 files relative to one
            # process's data loader.
            if with_horovod:
                validation_frequency //= hvd.size()
        elif freq_units == "batch":
            validation_frequency = freq
        else:
            raise ValueError(
                f"Unknown validation frequency units '{validation_frequency}'"
            )
    return validation_frequency


if __name__ == "__main__":
    main()
