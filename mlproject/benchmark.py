import argparse
import collections
import glob
import math
import numpy as np
import os
import psutil
import resource
import time
import xarray as xr
import yaml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

import mlflow

""" This script tests the performance of the Application 1 data loader

Each file the data loader reads has predictors with dimensions (leadtime, y, x, predictor). This
tensor is of the size (59, 2321, 1796, 8). This needs to be processed such that the output is
(leadtime, y_patch, x_patch, predictor), where y_patch and x_patch typically are 256.
"""

def check_horovod():
    """Check if we should run with horovod based on environment variables

    Returns:
        bool: True if we should run with horovod, False otherwise
    """
    # Program is run with horovodrun
    with_horovod = "HOROVOD_RANK" in os.environ

    if not with_horovod:
        # Program is run with srun
        with_horovod = "SLURM_STEP_NUM_TASKS" in os.environ and int(os.environ["SLURM_STEP_NUM_TASKS"]) > 1

    return with_horovod

with_horovod = check_horovod()
if with_horovod:
    # Import it 
    print("Running with horovod")
    import horovod.tensorflow as hvd

# TODO: When repeating a dataset and the dataset size  doesn't divide evenly by the batch size, we
# run out of data. We should really strive to make the benchmark divide evenly, otherwise it is hard
# to analyse the results for the training.

def main():
    parser = argparse.ArgumentParser("Program that test the MAELSTROM AP1 data pipeline")
    parser.add_argument("files", help="Read data from these files (e.g. /p/scratch/deepacf/maelstrom/maelstrom_data/ap1/air_temperature/5TB/2020030*.nc)", nargs="+")
    parser.add_argument("-j", default=tf.data.AUTOTUNE, type=parse_num_parallel_calls, help="Number of parallel calls. If not specified, use tf.data.AUTOTUNE.", dest="num_parallel_calls")
    parser.add_argument('-p', default=None, type=int, help="Patch size in pixels", dest="patch_size")
    parser.add_argument('-t', help="Rebatch all leadtimes in one sample", dest="with_leadtime", action="store_true")
    parser.add_argument('-b', default=1, type=int, help="Batch size (default 1)", dest="batch_size")
    parser.add_argument('-c', help='Cache data to this filename', dest="filename_cache")
    parser.add_argument('-e', default=1, type=int, help='Number of epochs', dest="epochs")
    parser.add_argument('-m', default="train", help='Mode. One of load, train', dest="mode", choices=["load", "train", "infer"])
    parser.add_argument('-val', help='Filenames used for validation', dest="validation_files", nargs="*")
    parser.add_argument('--debug', help='Turn on debugging information', action="store_true")
    parser.add_argument('--cpu', help='Limit execution to CPU', dest='cpu_only', action='store_true')
    parser.add_argument('--norm', default="/p/scratch/deepacf/maelstrom/maelstrom_data/ap1/air_temperature/normalization.yml", help='File with normalization information', dest='normalization')
    parser.add_argument('-s', help='Shuffle leadtimes (read one leadtime from each file)', dest='shuffle_leadtimes', action="store_true")
    parser.add_argument('-f', help='Generate fake data (thus there is no reading from the filesystem', dest='fake_data', action='store_true')
    args = parser.parse_args()

    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    filenames = list()
    for f in args.files:
        filenames += glob.glob(f)


    main_process = True
    num_processes = 1
    set_gpu_memory_growth()
    if with_horovod:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        hvd.init()
        print(hvd.rank(), hvd.size())
        print("Num GPUs Available: ", len(gpus))
        if len(gpus) == 0:
            raise Exception("No GPUs available")
        if len(gpus) > 1:
        # if hvd.size() == len(gpus):
            # Probably using horovodrun (not srun)
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")
        main_process = hvd.rank() == 0
        num_processes = hvd.size()
        if len(filenames) % num_processes != 0 and main_process:
            print(f"Warning number of files ({len(filenames)}) not divisible by {num_processes}")
    else:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        print("Num GPUs Available: ", len(gpus))

    # Let the Loader do the sharding, because then we can shard using different filenames
    loader = Ap1Loader(filenames, args.patch_size, args.batch_size, args.normalization, args.with_leadtime,
            with_horovod, args.epochs, args.filename_cache, args.shuffle_leadtimes, args.fake_data, args.debug)
    if main_process:
        print(loader)
    dataset = loader.get_dataset(args.num_parallel_calls)

    # Load all the data
    s_time = time.time()
    count = 0
    loader.start_time = s_time
    time_last_file = s_time
    saving_time = None
    time_first_sample = None
    loader_size_gb = loader.size_gb * num_processes
    loader_num_files = loader.num_files * num_processes

    input_shape = loader.batch_predictor_shape[1:]
    num_outputs = 3
    model = Unet(input_shape, num_outputs, levels=6)
    learning_rate = 1e-3
    loss = quantile_score
    optimizer = keras.optimizers.Adam(learning_rate)
    callbacks = []
    if with_horovod:
        optimizer = hvd.DistributedOptimizer(optimizer, backward_passes_per_step=1,
                average_aggregated_gradients=True)
        callbacks += [hvd.keras.callbacks.BroadcastGlobalVariablesCallback(0)]
        callbacks += [hvd.keras.callbacks.MetricAverageCallback()]
    if main_process:
        timing_callback = TimingCallback()
        callbacks += [timing_callback, TrainingCallback()]
    model.compile(optimizer=optimizer, loss=loss)
    if main_process and args.mode in ["train", "infer"]:
        model.summary()

    if args.mode == "train":
        ss_time = time.time()
        kwargs = dict()
        if args.validation_files is not None:

            val_filenames = list()
            for f in args.validation_files:
                val_filenames += glob.glob(f)

            # Do sharding on the dataset, instead of in the loader, since we might not have enough
            # files to support sharding into the number of processes
            val_loader = Ap1Loader(val_filenames, args.patch_size, args.batch_size, args.normalization, args.with_leadtime,
                    False, 1, args.filename_cache, args.shuffle_leadtimes, args.fake_data, args.debug)
            val_dataset = val_loader.get_dataset(args.num_parallel_calls)
            if with_horovod:
                val_dataset = val_dataset.shard(hvd.size(), hvd.rank())
            kwargs["validation_data"] = val_dataset
        history = model.fit(dataset, epochs=args.epochs, steps_per_epoch=loader.num_batches,
                callbacks=callbacks, verbose=main_process, **kwargs)
        training_time = time.time() - ss_time
        ss_time = time.time()

        create_directory("results")
        model.save("results/model")
        saving_time = time.time() - ss_time
        time_first_sample = None
    elif args.mode == "load":
        for k in dataset:
            curr_time = time.time()
            if count == 0 and main_process:
                # The first sample is avaiable
                agg_time = curr_time - s_time
                time_first_sample = agg_time
                print(f"First sample ready:")
                print(f"   Tensor shape: {k[0].shape}")
                print(f"   Time: {agg_time:.2f}")
                print_gpu_usage("   GPU memory: ")
                print_cpu_usage("   CPU memory: ")
            count += 1

            if count % loader.num_batches_per_file == 0 and main_process:
                # We have processed a complete file
                curr_file_index = count // loader.num_batches_per_file
                print(f"Done {curr_file_index} files")

                this_time = time.time() - time_last_file
                this_size_gb = loader.size_gb / loader_num_files
                print(f"   Curr time: {this_time:.2f} s")
                print(f"   Curr performance: {this_size_gb / this_time:.2f} GB/s")
                if with_horovod:
                    print(f"   Agg curr performance: {this_size_gb / this_time * num_processes:.2f} GB/s")

                agg_time = curr_time - s_time
                agg_size_gb = loader.size_gb / loader.num_batches * count
                print(f"   Acc time: {agg_time:.2f} ")
                mlflow.log_metrics({"Acc time": agg_time}, step=count)
                print(f"   Avg time per file: {agg_time/curr_file_index:.2f} s")
                mlflow.log_metrics({"Average time per file in seconds": agg_time/curr_file_index}, step=count)
                print(f"   Avg performance: {agg_size_gb / agg_time:.2f} GB/s")
                mlflow.log_metrics({"Average performance in GB/s": agg_size_gb / agg_time}, step=count)
                if with_horovod:
                    print(f"   Agg avg performance: {agg_size_gb / agg_time * num_processes:.2f} GB/s")

                print_gpu_usage("   GPU memory: ")
                print_cpu_usage("   CPU memory: ")
                time_last_file = curr_time
    elif args.mode == "infer":
        ss_time = time.time()
        time_first_sample = None
        inference_time = 0
        if 1:
            # Run predict_on_batch on each batch
            count = 0
            for predictors, targets in dataset:
                count += 1
                curr_time = time.time()
                if count % loader.num_batches_per_file == 0 and main_process:
                    curr_file_index = count // loader.num_batches_per_file
                    print(f"Done {curr_file_index} files")

                    agg_time = curr_time - s_time
                    agg_size_gb = loader.size_gb / loader.num_batches * count
                    print(f"   Acc time: {agg_time:.2f} ")
                    mlflow.log_metrics({"Acc time": agg_time}, step=count)
                    print(f"   Avg time per file: {agg_time/curr_file_index:.2f} s")
                    mlflow.log_metrics({"Average time per file in seconds": agg_time / curr_file_index}, step=count)
                    print(f"   Avg performance: {agg_size_gb / agg_time:.2f} GB/s")
                    mlflow.log_metrics({"Average performance in GB/s": agg_size_gb / agg_time}, step=count)
                    if with_horovod:
                        print(f"   Agg avg performance: {agg_size_gb / agg_time * num_processes:.2f} GB/s")

                sss_time = time.time()

                if time_first_sample is None:
                    time_first_sample = time.time() - ss_time
                q = model.predict_on_batch(predictors)
                inference_time += time.time() - sss_time
            print(f"Batches {count}")
            total_time = time.time() - ss_time
            data_loading_overhead = total_time - inference_time
        else:
            # Run predict on the whole dataset. This seams to cause memory to run out when running
            # on large datasets.
            q = model.predict(dataset, verbose=1)
            inference_time = time.time() - ss_time
            data_loading_overhead = None
        # print(time.time() - ss_time)

    if with_horovod:
        hvd.join()
    total_time = time.time() - s_time
    if main_process:
        print("")
        print("Benchmark configuration:")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Patch size: {args.patch_size}")
        print(f"   All leadtimes: {args.with_leadtime}")
        print(f"   Num parallel_calls: {args.num_parallel_calls}")
        if with_horovod:
            print(f"   Horovod proceses: {num_processes}")
        if args.mode == "train":
            print(f"   Num epochs: {args.epochs}")
        print("")
        print("Benchmark results:")
        print(f"   Total time: {total_time:.2f} s")
        print(f"   Number of files: {loader.num_files}")
        print(f"   Time per file: {total_time / loader.num_files:.2f}")
        if time_first_sample is not None:
            print(f"   Time to first sample: {time_first_sample:.2f} s")
        if with_horovod:
            print(f"   Data amount / process: {loader.size_gb:2f} GB")
            print(f"   Performance / process: {loader.size_gb / total_time:.2f} GB/s")
        print(f"   Data amount: {loader_size_gb:2f} GB")
        print(f"   Performance: {loader_size_gb / total_time:.2f} GB/s")
        print_gpu_usage("   GPU memory: ")
        print_cpu_usage("   CPU memory: ")

        print("")
        if args.mode == "train":
            print("Training performance:")
            # for key, value in history.history.items():
            #     print(key, value)
            times = timing_callback.get_epoch_times()
            print(f"   Total runtime: {total_time:.2f} s")
            print(f"   Total training time: {training_time:.2f} s")
            print(f"   Model saving time: {saving_time:.2f} s")
            print(f"   Average performance: {loader_size_gb / training_time * args.epochs:.2f} GB/s")
            print(f"   First epoch time: {times[0]:.2f} s")
            print(f"   Min epoch time: {np.min(times):.2f} s")
            print(f"   Performance min epoch: {loader_size_gb / np.min(times):.2f} GB/s")
            print(f"   Mean epoch time: {np.mean(times):.2f} s")
            print(f"   Performance mean epoch: {loader_size_gb / np.mean(times):.2f} GB/s")
            print(f"   Max epoch time: {np.max(times):.2f} s")
            print(f"   Performance max epoch: {loader_size_gb / np.max(times):.2f} GB/s")
            print(f"   Final loss: {history.history['loss'][-1]:.3f}")
            if args.validation_files is not None:
                print(f"   Final val loss: {history.history['val_loss'][-1]:.3f}")
            print(f"   Average time per batch: {total_time / loader.num_batches / num_processes / args.epochs:.2f} s")
            print_gpu_usage("   Final GPU memory: ")
            print_cpu_usage("   Final CPU memory: ")
            # for i, curr_time in enumerate():
            #     print("   Epoch {i} {curr_time}")
        elif args.mode == "load":
            print("Data loading performance:")
            print(f"   Total runtime: {total_time:.2f} s")
            print(f"   Average performance: {loader_size_gb / total_time * args.epochs:.2f} GB/s")
            print(f"   Average time per batch: {total_time / loader.num_batches / num_processes / args.epochs:.2f} s")
            print_gpu_usage("   Final GPU memory: ")
            print_cpu_usage("   Final CPU memory: ")
            for k,v in loader.timing.items():
                print(f"   {k}: {v:.2f}")
        elif args.mode == "infer":
            print("Inference performance:")
            print(f"   Total runtime: {total_time:.2f} s")
            print(f"   Inference time: {inference_time:.2f} s")
            if data_loading_overhead is not None:
                print(f"   Data loading overhead: {data_loading_overhead:.2f} s")
            print(f"   Average performance: {loader_size_gb / total_time * args.epochs:.2f} GB/s")
            print_gpu_usage("   Final GPU memory: ")
            print_cpu_usage("   Final CPU memory: ")


def map_decorator1_to_3(func):
    """Decorator to wrap a 1-argument function as a tf.py_function"""
    def wrapper(self, i):
        return tf.py_function(
                lambda i: func(self, i),
                inp=(i,),
                Tout=(tf.float32, tf.float32, tf.float32)
                )
    return wrapper

def map_decorator2_to_2(func):
    """Decorator to wrap a 2-argument function as a tf.py_function"""
    def wrapper(self, i, j):
        return tf.py_function(
                lambda i, j: func(self, i, j),
                inp=(i, j),
                Tout=(tf.float32, tf.float32)
                )
    return wrapper

def map_decorator3_to_2(func):
    """Decorator to wrap a 2-argument function as a tf.py_function"""
    def wrapper(self, i, j, k):
        return tf.py_function(
                lambda i, j, k: func(self, i, j, k),
                inp=(i, j, k),
                Tout=(tf.float32, tf.float32)
                )
    return wrapper

def map_decorator3_to_3(func):
    """Decorator to wrap a 2-argument function as a tf.py_function"""
    def wrapper(self, i, j, k):
        return tf.py_function(
                lambda i, j, k: func(self, i, j, k),
                inp=(i, j, k),
                Tout=(tf.float32, tf.float32, tf.float32)
                )
    return wrapper

"""
Data loader
"""
class Ap1Loader:
    def __init__(self, filenames, patch_size, batch_size, filename_normalization, with_leadtime=False,
            with_horovod=False, repeat=None, filename_cache=None, shuffle_leadtimes=False, create_fake_data=False, debug=True):
        self.with_horovod = with_horovod
        if self.with_horovod:
            if len(filenames) == 0:
                raise Exception(f"Too few files ({len(filenames)}) to divide into {hvd.size()} processes")
            start = hvd.rank() * math.ceil(len(filenames) // hvd.size())
            end = (hvd.rank() + 1 ) * math.ceil(len(filenames) // hvd.size())
            if end > len(filenames):
                end = len(filenames)
            self.filenames = [filenames[f] for f in range(start, end)]
            if len(self.filenames) == 0:
                raise Exception(f"Too few files ({len(filenames)}) to divide into {hvd.size()} processes")
        else:
            self.filenames = filenames
        self.filename_normalization = filename_normalization
        self.with_leadtime = with_leadtime
        self.repeat = repeat
        self.filename_cache = filename_cache
        self.debug = debug
        self.shuffle_leadtimes = shuffle_leadtimes

        # Where should data reside during the processing steps? Processing seems faster on CPU,
        # perhaps because the pipeline stages can run in parallel better than on the GPU?
        self.device = "CPU:0"

        # Load metadata
        with xr.open_dataset(filenames[0], decode_timedelta=False) as dataset:
            self.num_leadtimes = len(dataset.variables["leadtime"])
            self.predictor_shape = dataset.variables["predictors"].shape
            self.static_predictor_shape = dataset.variables["static_predictors"].shape
            self.target_shape = dataset.variables["target_mean"].shape
            self.patch_size = patch_size
            if self.patch_size is None:
                num_x_patches = 1
                num_y_patches = 1
            else:
                num_x_patches = self.predictor_shape[2] // self.patch_size
                num_y_patches = self.predictor_shape[1] // self.patch_size
                if self.patch_size > self.predictor_shape[1] or self.patch_size > self.predictor_shape[2]:
                    raise Exception("Cannot patch this grid. It is too small.")
            if self.with_leadtime:
                self.num_samples_per_file = num_x_patches * num_y_patches
            else:
                self.num_samples_per_file = num_x_patches * num_y_patches * self.num_leadtimes
            self.num_patches_per_file = num_x_patches * num_y_patches
            self.predictor_names = [i for i in dataset.variables["predictor"].values]
            self.predictor_names += [i for i in dataset.variables["static_predictor"].values]

        self.extra_predictor_names = ["x", "y", "leadtime"]
        self.num_extra_features = len(self.extra_predictor_names)
        self.predictor_names += self.extra_predictor_names

        # cache=False seems to have no effect
        self.data = xr.open_mfdataset(self.filenames, combine="nested", concat_dim="time") # , cache=False)

        # Used to store the time it takes for each processing step
        self.timing = collections.defaultdict(lambda: 0)

        # The name of the raw forecast predictor, used to subtract the target
        self.raw_predictor_index = self.predictor_names.index("air_temperature_2m")
        self.create_fake_data = create_fake_data

        # Cache the normalization coefficients
        self.normalize_add = None
        self.normalize_factor = None

        self.start_time = time.time()
        self.batch_size = batch_size

    def get_dataset(self, num_parallel_calls):
        """Returns a tf.data object

        Args:
            num_parallel_calls (int): Maximum number of threads that each pipeline stage can use
        """
        self.start_time = time.time()

        if self.shuffle_leadtimes:
            dataset = tf.data.Dataset.range(len(self.filenames) * self.num_leadtimes)
            num_read_threads = num_parallel_calls
        else:
            dataset = tf.data.Dataset.range(len(self.filenames))
            num_read_threads = 1

        dataset = dataset.shuffle(len(self.filenames))
        if self.repeat is not None:
            dataset = dataset.repeat(self.repeat)

        # Read data from NETCDF files
        # Outputs three tensors:
        #     predictors: 59, 2321, 1796, 8
        #     static_predictor: 2321, 1796, 6
        #     targets: 59, 2321, 1796, 1
        # Set number of parallel calls to 1, so that the pipeline doesn't get too far ahead on the
        # reading, causing the memory requirement to be large. The reading is not the bottleneck so
        # we don't need to read multiple files in parallel.
        dataset = dataset.map(self.read, num_read_threads)
        dataset = dataset.prefetch(1)

        # Broadcast static_predictors to leadtime dimension
        # Set parallel_calls to 1 here as well, to prevent the pipeline from getting too far ahead
        dataset = dataset.map(self.expand_static_predictors, num_read_threads) # num_parallel_calls)

        # Unbatch the leadtime dimension, so that each leadtime can be processed in parallel
        dataset = dataset.unbatch()
        dataset = dataset.batch(1)

        # Processing steps
        if 1:
            # Merge static_predictors into predictors and add a few more predictors
            dataset = dataset.map(self.feature_extraction, num_parallel_calls)
            # Predictor shape: 1, 2321, 1796, 16

            # Normalize the predictors
            dataset = dataset.map(self.normalize, num_parallel_calls)
            # Predictor shape: 1, 2321, 1796, 16

            # Split the y,x dimensions into patches of size 256x256
            dataset = dataset.map(self.patch, num_parallel_calls)
            # Predictor shape: 1, 63, 256, 256, 16

            # Sutract the raw forecast from the targets
            dataset = dataset.map(self.diff, num_parallel_calls)
            # Predictor shape: 1, 63, 256, 256, 16
        else:
            # Perform the 4 above steps in one function
            dataset = dataset.map(self.process, num_parallel_calls)
            # Predictor shape: 1, 63, 256, 256, 16

        if self.with_leadtime:
            # Collect all leadtimes into one tensor again
            dataset = dataset.unbatch()
            # Predictor shape: 63, 256, 256, 16
            dataset = dataset.batch(self.num_leadtimes)
            # Predictor shape: 59, 63, 256, 256, 16

            # Put patch dimension before leadtime
            dataset = dataset.map(self.reorder, num_parallel_calls)
            # Predictor shape: 63, 59, 256, 256, 16

            # Unbatch the patch dimension
            dataset = dataset.unbatch()
            # Predictor shape: 59, 256, 256, 14
        else:
            # Unbatch the leadtime dimension
            dataset = dataset.unbatch()
            # Predictor shape: 63, 256, 256, 14

            # Unbatch the patch dimension
            dataset = dataset.unbatch()
            # Predictor shape: 256, 256, 14

            # Batch so that the dataset has 4 dimensions
            dataset = dataset.batch(1)
            # Predictor shape: 1, 256, 256, 14

        dataset = dataset.batch(self.batch_size)

        if self.filename_cache is not None:
            dataset = dataset.cache(self.filename_cache)

        # Copy data to the GPU
        dataset = dataset.map(self.to_gpu, num_parallel_calls)
        dataset = dataset.prefetch(1)
        return dataset

    """
    Dataset operations
    """
    @map_decorator1_to_3
    def read(self, index):
        """Read data from file

        Args:
            index(int): File index to read data from

        Returns:
            predictors (tf.tensor): Predictors tensor (leadtime, y, x, predictor)
            static_predictors (tf.tensor): Satic predictor tensor (y, x, static_predictor)
            targets (tf.tensor): Targets tensor (leadtime, y, x, 1)
        """
        s_time = time.time()
        index = index.numpy()
        if self.shuffle_leadtimes:
            file_index = index // self.num_leadtimes
            leadtime_index = index % self.num_leadtimes
            print(index, file_index, leadtime_index)
        else:
            file_index = None
            leadtime_index = None

        self.print(f"Start reading index={index}")

        with tf.device(self.device):
            if not self.create_fake_data:
                if not self.shuffle_leadtimes:
                    predictors = self.data["predictors"][index, ...]
                    static_predictors = self.data["static_predictors"][index, ...]
                    targets = self.data["target_mean"][index, ...]
                else:
                    predictors = self.data["predictors"][index, leadtime_index, ...]
                    static_predictors = self.data["static_predictors"][index, leadtime_index, ...]
                    targets = self.data["target_mean"][index, leadtime_index, ...]
                    predictors = np.expand_dims(predictors, -1)
                    static_predictors = np.expand_dims(static_predictors, -1)
                    targets = np.expand_dims(targets, -1)

                targets = np.expand_dims(targets, -1)

                # Force explicit conversion here, so that we can account the time it takes
                # Otherwise the conversion happens when the function returns
                predictors = tf.convert_to_tensor(predictors)
                static_predictors = tf.convert_to_tensor(static_predictors)
                targets = tf.convert_to_tensor(targets)
            else:
                if not self.shuffle_leadtimes:
                    predictors = tf.random.uniform(self.predictor_shape)
                    targets = tf.expand_dims(tf.random.uniform(self.target_shape), 3)
                    static_predictors = tf.random.uniform(self.static_predictor_shape)
                else:
                    # TODO:
                    predictors = tf.random.uniform(self.predictor_shape)
                    targets = tf.expand_dims(tf.random.uniform(self.target_shape), 3)
                    static_predictors = tf.random.uniform(self.static_predictor_shape)
        e_time = time.time()
        self.timing["read"] += e_time - s_time
        # print(predictors.shape, static_predictors.shape, targets.shape)
        return predictors, static_predictors, targets

    @map_decorator3_to_3
    def expand_static_predictors(self, predictors, static_predictors, targets):
        """Copies static predictors to leadtime dimension"""
        s_time = time.time()
        self.print("Start processing")
        with tf.device(self.device):
            shape = [predictors.shape[0], 1, 1, 1]
            static_predictors = tf.expand_dims(static_predictors, 0)
            static_predictors = tf.tile(static_predictors, shape)
        self.timing["expand"] += time.time() - s_time
        return predictors, static_predictors, targets


    @map_decorator3_to_2
    def feature_extraction(self, predictors, static_predictors, targets):
        """Merges predictors, static_predictors, and two features"""
        s_time = time.time()
        # self.print("Start feature extraction")
        features = [predictors, static_predictors]
        with tf.device(self.device):
            shape = list(predictors.shape[:-1]) + [1]
            for f in range(self.num_extra_features):
                feature = np.zeros(shape, np.float32)
                features += [feature]

            predictors = tf.concat(features, axis=-1)
        # self.print("Done feature extraction")
        self.timing["feature"] += time.time() - s_time
        return predictors, targets

    @map_decorator2_to_2
    def normalize(self, predictors, targets):
        s_time = time.time()
        with tf.device(self.device):
            if 0:
                # Poor performance because of poor cache locality
                new_predictors = list()
                for p in range(predictors.shape[-1]):
                    new_predictor = tf.expand_dims((predictors[..., p] + 1 ) / 2, -1)
                    new_predictors += [new_predictor]
                predictors = tf.concat(new_predictors, axis=-1)
            else:
                # Check for the existance of both vectors, since when this runs in parallel, the
                # first vector may be available before the other
                if self.normalize_add is None or self.normalize_factor is None:
                    coefficients = self.read_normalization()
                    a = coefficients[:, 0]
                    s = coefficients[:, 1]
                    shape = tf.concat((tf.shape(predictors)[0:-1], [1]), 0)

                    def expand_array(a, shape):
                        """Expands array a so that it has the shape"""
                        if len(a.shape) == 5:
                            # Use if unbatch has not been run
                            a = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(a, 0), 0), 0), 0)
                        else:
                            a = tf.expand_dims(tf.expand_dims(tf.expand_dims(a, 0), 0), 0)
                        a = tf.tile(a, shape)
                        return a

                    self.normalize_add = expand_array(a, shape)
                    self.normalize_factor = expand_array(s, shape)

                predictors = predictors - self.normalize_add
                predictors = predictors / self.normalize_factor

        self.timing["normalize"] += time.time() - s_time
        return predictors, targets

    def read_normalization(self):
        coefficients = np.zeros([self.num_predictors, 2], np.float32)
        coefficients[:, 1] = 1
        with open(self.filename_normalization) as file:
            data = yaml.load(file, Loader=yaml.SafeLoader)
            # Add normalization information for the extra features
            data["x"] = (1000, 100)
            data["y"] = (1000, 100)
            data["leadtime"] = (30, 10)

            for i, name in enumerate(self.predictor_names):
                if name in data:
                    coefficients[i, :] = data[name]
                else:
                    coefficients[i, :] = [0, 1]
        return coefficients


    @map_decorator2_to_2
    def patch(self, predictors, targets):
        """Decompose grid into patches

        Input: leadtime, y, x, predictor
        Output: leadtime, patch, y_patch, x_patch, predictor
        """
        s_time = time.time()
        # self.debug("Start patch", time.time() - self.s_time, predictors.shape)

        if self.patch_size is None:
            # A patch dimension is still needed when patching is not done
            with tf.device(self.device):
                p, t = tf.expand_dims(predictors, 1),  tf.expand_dims(targets, 1)
            return p, t

        def patch_tensor(a, ps):
            """ Patch a 4D array
            Args:
                a (tf.tensor): 4D (leadtime, y, x, predictor)
                ps (int): Patch size

            Returns:
                tf.tensor: 5D (patch, leadtime, ps, ps, predictor)
            """
            # This is magic, don't ask how it works...

            with tf.device(self.device):
                if len(a.shape) == 4:
                    LT = a.shape[0]
                    Y = a.shape[1]
                    X = a.shape[2]
                    P = a.shape[3]
                    num_patches_y = Y // ps
                    num_patches_x = X // ps

                    # Remove edge of domain to make it evenly divisible
                    a = a[:, :Y//ps * ps, :X//ps * ps, :]

                    a = tf.image.extract_patches(a, [1, ps, ps, 1], [1, ps, ps, 1], rates=[1, 1, 1, 1], padding="SAME")
                    a = tf.expand_dims(a, 0)
                    a = tf.reshape(a, [LT, num_patches_y * num_patches_x, ps, ps, P])
                    return a
                else:
                    Y = a.shape[0]
                    X = a.shape[1]
                    P = a.shape[2]
                    num_patches_y = Y // ps
                    num_patches_x = X // ps

                    # Remove edge of domain to make it evenly divisible
                    a = a[:Y//ps * ps, :X//ps * ps, :]
                    a = tf.expand_dims(a, 0)

                    a = tf.image.extract_patches(a, [1, ps, ps, 1], [1, ps, ps, 1], rates=[1, 1, 1, 1], padding="SAME")
                    a = tf.reshape(a, [num_patches_y * num_patches_x, ps, ps, P])
                    return a

        with tf.device(self.device):
            p = patch_tensor(predictors, self.patch_size)
            t = patch_tensor(targets, self.patch_size)

        # self.debug("Done patching", time.time() - self.s_time, p.shape)
        # print("Patching: ", p.shape, t.shape)
        self.timing["patch"] += time.time() - s_time
        return p, t

    @map_decorator2_to_2
    def diff(self, predictors, targets):
        """Subtract the raw_forecast from the target"""
        # self.print("Start diff")
        s_time = time.time()
        with tf.device(self.device):
            raw_predictor = tf.expand_dims(predictors[..., self.raw_predictor_index], -1)
            targets = tf.math.subtract(targets, raw_predictor)

        self.timing["diff"] += time.time() - s_time
        return predictors, targets

    @map_decorator3_to_2
    def process(self, predictors, static_predictors, targets):
        """Perform all processing steps in one go"""
        with tf.device(self.device):
            p, t = self.feature_extraction(predictors, static_predictors, targets)
            p, t = self.normalize(p, t)
            p, t = self.patch(p, t)
            p, t = self.diff(p, t)
        return p, t

    @map_decorator2_to_2
    def reorder(self, predictors, targets):
        """Move patch dimension to be the first dimension

        Input: leadtime, patch, y, x, predictor
        Output: patch, leadtime, y, x, predictor
        """
        if self.patch_size is None:
            return predictors, targets
        s_time = time.time()
        with tf.device(self.device):
            p = tf.transpose(predictors, [1, 0, 2, 3, 4])
            t = tf.transpose(targets, [1, 0, 2, 3, 4])
        self.timing["reorder"] += time.time() - s_time
        return p, t

    @map_decorator2_to_2
    def to_gpu(self, predictors, targets):
        s_time = time.time()
        p = tf.convert_to_tensor(predictors)
        t = tf.convert_to_tensor(targets)
        self.timing["to_gpu"] += time.time() - s_time
        return p, t

    def print(self, message):
        if self.debug:
            curr_time = time.time() - self.start_time
            print(f"{curr_time:.2f}: {message}")

    """
    Dataset properties
    """
    @property
    def size_gb(self):
        size_bytes = np.product(self.predictor_shape) * 4
        size_bytes += np.product(self.target_shape) * 4
        size_bytes += np.product(self.static_predictor_shape) * self.num_leadtimes * 4
        return size_bytes * len(self.filenames) / 1024 ** 3

    @property
    def num_files(self):
        return len(self.filenames)

    @property
    def num_samples(self):
        return self.num_samples_per_file * self.num_files

    @property
    def num_batches(self):
        return math.ceil(self.num_samples / self.batch_size)

    @property
    def num_batches_per_file(self):
        return math.ceil(self.num_samples_per_file / self.batch_size)

    @property
    def num_predictors(self):
        return len(self.predictor_names)

    @property
    def batch_predictor_shape(self):
        shape = [self.batch_size] + [i for i in self.predictor_shape]
        if not self.with_leadtime:
            shape[1] = 1
        if self.patch_size is not None:
            shape[2] = self.patch_size
            shape[3] = self.patch_size
        shape[-1] = self.predictor_shape[-1] + self.static_predictor_shape[-1] + self.num_extra_features
        return shape

    def __str__(self):
        s = "Dataset properties:\n"
        s += f"   Number of files: {len(self.filenames)}\n"
        s += f"   Dataset size: {self.size_gb:.2f} GB\n"
        s += f"   Predictor shape: {self.predictor_shape}\n"
        s += f"   Static predictor shape: {self.static_predictor_shape}\n"
        s += f"   Target shape: {self.target_shape}\n"
        s += f"   Batch tensor shape: {self.batch_predictor_shape}\n"
        s += f"   Patch size: {self.patch_size}\n"
        s += f"   Patches per file: {self.num_patches_per_file}\n"
        s += f"   Samples per file: {self.num_samples_per_file}\n"
        s += f"   Batches per file: {self.num_batches_per_file}\n"
        s += f"   Num batches: {self.num_batches}\n"
        s += f"   Batch size: {self.size_gb / self.num_batches * 1024:.2f} MB\n"
        return s

def parse_num_parallel_calls(string):
    if string == "autotune":
        return tf.data.AUTOTUNE
    else:
        return int(string)

def print_gpu_usage(message="", show_line=False):
    try:
        usage = tf.config.experimental.get_memory_info("GPU:0")
        output = message + ' - '.join([f"{k}: {v / 1024**3:.2f} GB" for k,v in usage.items()])
    except ValueError as e:
        output = message + ' None'

    if show_line:
        frameinfo = inspect.getouterframes(inspect.currentframe())[1]
        output += " (%s:%s)" % (frameinfo.filename, frameinfo.lineno)

    print(output)

def print_cpu_usage(message="", show_line=False):
    """Prints the current and maximum memory useage of this process
    Args:
        message (str): Prepend with this message
        show_line (bool): Add the file and line number making this call at the end of message
    """

    output = "current: %.2f GB - peak: %.2f GB" % (
        get_memory_usage() / 1024 ** 3,
        get_max_memory_usage() / 1024 ** 3,
    )
    output = message + output
    if show_line:
        frameinfo = inspect.getouterframes(inspect.currentframe())[1]
        output += " (%s:%s)" % (frameinfo.filename, frameinfo.lineno)

    print(output)

def get_memory_usage():
    p = psutil.Process(os.getpid())
    mem = p.memory_info().rss
    for child in p.children(recursive=True):
        mem += child.memory_info().rss
    return mem


def get_max_memory_usage():
    """In bytes"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1000


def set_gpu_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

def create_directory(filename):
    """Creates all sub directories necessary to be able to write filename"""
    dir = os.path.dirname(filename)
    if dir != "":
        os.makedirs(dir, exist_ok=True)


class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        self.s_time = time.time()
        self.start_times = dict()
        self.end_times = dict()

    def on_epoch_begin(self, epoch, logs = {}):
        print("Adding epoch", epoch)
        self.start_times[epoch] = time.time()

    def on_epoch_end(self,epoch,logs = {}):
        self.end_times[epoch] = time.time()
        self.times.append(time.time() - self.s_time)
        mlflow.log_metrics({"epoch time": self.times[epoch]}, step=epoch)

    def get_epoch_times(self):
        times = list()
        keys = list(self.start_times.keys())
        keys.sort()
        for key in keys:
            if key not in self.end_times:
                print(f"WARNING: Did not find epoch={key} in end times")
                continue
            times += [self.end_times[key] - self.start_times[key]]
        return times

    # def on_train_end(self,logs = {}):

class TrainingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.training_loss = dict()
        self.validation_loss= dict()

    def on_epoch_end(self, epoch, logs=None):
        self.training_loss["training loss"] = logs["loss"]
        if "val_loss" in logs:
            self.validation_loss["validation loss"] = logs["val_loss"]

        mlflow.log_metrics({**self.training_loss, **self.validation_loss}, step=epoch)

"""
ML model
"""
class Unet(keras.Model):
    def __init__(
        self,
        input_shape,
        num_outputs,
        features=16,
        levels=3,
        pool_size=2,
        conv_size=3,
        upsampling_type="conv_transpose",
    ):
        """U-net

        Args:
            features (int): Number of features in the first layer
            levels (int): Depth of the U-net
            pool_size (int): Pooling ratio (> 0)
            upsampling_type (str): One of "upsampling" or "conv_transpose"
            conv_size (int): Convolution size (> 0)
        """
        if upsampling_type not in ["upsampling", "conv_transpose"]:
            raise ValueError(f"Unknown upsampling type {upsampling_type}")

        # print(f"Initializing a U-Net with shape {input_shape}")

        self._num_outputs = num_outputs
        self._features = features
        self._levels = levels
        self._pool_size = pool_size
        self._conv_size = conv_size
        self._upsampling_type = upsampling_type

        # Build the model
        inputs = keras.layers.Input(input_shape)
        outputs = self.get_outputs(inputs)

        super().__init__(inputs, outputs)

    def get_outputs(self, inputs):
        outputs = inputs
        levels = list()

        features = self._features

        pool_size = [1, self._pool_size, self._pool_size]
        hood_size = [1, self._conv_size, self._conv_size]

        Conv = keras.layers.Conv3D
        if self._upsampling_type == "upsampling":
            UpConv = keras.layers.UpSampling3D
        elif self._upsampling_type == "conv_transpose":
            UpConv = keras.layers.Conv3DTranspose

        # Downsampling
        # conv -> conv -> max_pool
        for i in range(self._levels - 1):
            outputs = Conv(features, hood_size, activation="relu", padding="same")(
                outputs
            )
            outputs = Conv(features, hood_size, activation="relu", padding="same")(
                outputs
            )
            levels += [outputs]
            # print(i, outputs.shape)

            outputs = keras.layers.MaxPooling3D(pool_size=pool_size)(outputs)
            features *= 2

        # conv -> conv
        outputs = Conv(features, hood_size, activation="relu", padding="same")(
            outputs
        )
        outputs = Conv(features, hood_size, activation="relu", padding="same")(
            outputs
        )

        # upconv -> concat -> conv -> conv
        for i in range(self._levels - 2, -1, -1):
            features /= 2
            outputs = UpConv(features, hood_size, strides=pool_size, padding="same")(outputs)

            # print(levels[i].shape, outputs.shape)
            outputs = keras.layers.concatenate((levels[i], outputs), axis=-1)
            outputs = Conv(features, hood_size, activation="relu", padding="same")(
                outputs
            )
            outputs = Conv(features, hood_size, activation="relu", padding="same")(
                outputs
            )

        # Dense layer at the end
        outputs = keras.layers.Dense(self._num_outputs, activation="linear")(
            outputs
        )

        return outputs

"""
Loss function
"""
def quantile_score(y_true, y_pred):
    quantiles = [0.1, 0.5, 0.9]
    qtloss = 0
    for i, quantile in enumerate(quantiles):
        err = y_true[..., 0] - y_pred[..., i]
        qtloss += (quantile - tf.cast((err < 0), tf.float32)) * err
    return K.mean(qtloss) / len(quantiles)

if __name__ == "__main__":
    main()
