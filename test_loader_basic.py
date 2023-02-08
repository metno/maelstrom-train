import argparse
import collections
import glob
import numpy as np
import math
import os
import psutil
import resource
import time
import xarray as xr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

""" This script tests the performance of the Application 1 data loader
"""

def main():
    parser = argparse.ArgumentParser("Program that test the MAELSTROM AP1 data pipeline")
    parser.add_argument("files", help="Read data from these files", nargs="+")
    parser.add_argument("-j", default=1, type=get_num_parallel_calls, help="Number of parallel calls (number or AUTO)", dest="num_parallel_calls")
    parser.add_argument('-p', default=256, type=int, help="Patch size in pixels (default 256)", dest="patch_size")
    args = parser.parse_args()

    filenames = list()
    for f in args.files:
        filenames += glob.glob(f)

    loader = Ap1Loader(filenames, args.patch_size)
    print(loader)
    dataset = loader.get_dataset(args.num_parallel_calls)
    # dataset = loader.get_dataset(tf.data.AUTOTUNE)

    # Load all the data
    s_time = time.time()
    count = 0
    loader.start_time = s_time
    time_last_file = s_time
    for k in dataset:
        curr_time = time.time() 
        if count == 0:
            agg_time = curr_time - s_time
            print(f"First sample ready:")
            print(f"   Time: {agg_time:.2f}")
            print_gpu_usage("   GPU memory: ")
            print_cpu_usage("   CPU memory: ")
        # print(count)
        # print(k[0].device)
        count += 1
        if count % loader.num_samples_per_file == 0:
            curr_file_index = count // loader.num_samples_per_file
            print(f"Done {curr_file_index} files")

            agg_time = curr_time - s_time
            agg_size_gb = loader.size_gb / loader.num_samples * count
            print(f"   Agg time: {agg_time:.2f}")
            print(f"   Agg performance: {agg_size_gb / agg_time:.2f} GB/s")

            this_time = time.time() - time_last_file
            this_size_gb = loader.size_gb / loader.num_files
            print(f"   Curr time: {this_time:.2f}")
            print(f"   Curr performance: {this_size_gb / this_time:.2f} GB/s")

            print_gpu_usage("   GPU memory: ")
            print_cpu_usage("   CPU memory: ")
            time_last_file = curr_time

    total_time = time.time() - s_time
    print(f"Total time: {total_time:.2f} s")
    print(f"Performance: {loader.size_gb / total_time:.2f} GB/s")
    print("")
    print("Timing breakdown:")
    for k,v in loader.timing.items():
        print(f"   {k}: {v:.2f}")

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

class Ap1Loader:
    def __init__(self, filenames, patch_size=16):
        self.filenames = filenames
        self.device = "CPU:0"

        # Load metadata
        with xr.open_dataset(filenames[0], decode_timedelta=False) as dataset:
            self.num_leadtimes = len(dataset.variables["leadtime"])
            self.predictor_shape = dataset.variables["predictors"].shape
            self.static_predictor_shape = dataset.variables["static_predictors"].shape
            self.target_shape = dataset.variables["target_mean"].shape
            self.patch_size = patch_size
            num_x_patches = self.predictor_shape[2] // self.patch_size
            num_y_patches = self.predictor_shape[1] // self.patch_size
            if self.patch_size > self.predictor_shape[1] or self.patch_size > self.predictor_shape[2]:
                raise Exception("Cannot patch this grid. It is too small.")
            self.num_samples_per_file = num_x_patches * num_y_patches
            predictor_names = [i for i in dataset.variables["predictor"].values]
        self.data = xr.open_mfdataset(self.filenames, combine="nested", concat_dim="time")
        self.start_time = time.time()
        self.timing = collections.defaultdict(lambda: 0)
        self.raw_predictor_index = predictor_names.index("air_temperature_2m")
        self.create_fake_data = False

    def get_dataset(self, num_parallel_calls):
        """Returns a tf.data object"""
        self.start_time = time.time()

        dataset = tf.data.Dataset.range(len(self.filenames))
        # dataset = dataset.shuffle(10000)
        dataset = dataset.map(self.read, 1)
        dataset = dataset.map(self.expand_static_predictors, 1)
        dataset = dataset.prefetch(1)

        # Rebatch the leadtime
        dataset = dataset.unbatch()
        dataset = dataset.batch(math.ceil(self.num_leadtimes / num_parallel_calls))

        # Processing steps
        if 1:
            dataset = dataset.map(self.feature_extraction, num_parallel_calls)
            dataset = dataset.map(self.normalize, num_parallel_calls)
            dataset = dataset.map(self.patch, num_parallel_calls)
            dataset = dataset.map(self.diff, num_parallel_calls)
        else:
            dataset = dataset.map(self.process, num_parallel_calls)

        # Collect leadtimes
        dataset = dataset.unbatch()
        dataset = dataset.batch(self.num_leadtimes)

        # Put patch dimension first
        dataset = dataset.map(self.reorder, num_parallel_calls)

        # Unbatch the patch
        dataset = dataset.unbatch()

        dataset = dataset.map(self.to_gpu, num_parallel_calls)
        return dataset

    """
    Dataset operations
    """
    @map_decorator1_to_3
    def read(self, index):
        """Returns 3 tensors"""
        s_time = time.time()
        self.print(f"Start reading index={index}")
        with tf.device(self.device):
            if not self.create_fake_data:
                index = index.numpy()
                predictors = self.data["predictors"][index, ...]
                static_predictors = self.data["static_predictors"][index, ...]
                targets = self.data["target_mean"][index, ...]
                targets = np.expand_dims(targets, 3)
            else:
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
        s_time = time.time()
        # self.print("Start feature extraction")
        with tf.device(self.device):
            shape = list(predictors.shape[:-1]) + [1]
            feature1 = np.zeros(shape, np.float32)
            feature2 = np.zeros(shape, np.float32)

            predictors = tf.concat((predictors, static_predictors, feature1, feature2), axis=-1)
        # self.print("Done feature extraction")
        self.timing["feature"] += time.time() - s_time
        return predictors, targets

    @map_decorator2_to_2
    def normalize(self, predictors, targets):
        s_time = time.time()
        with tf.device(self.device):
            new_predictors = list()
            for p in range(predictors.shape[-1]):
                new_predictor = tf.expand_dims((predictors[..., p] + 1 ) / 2, -1)
                new_predictors += [new_predictor]
            predictors = tf.concat(new_predictors, axis=-1)
        self.timing["normalize"] += time.time() - s_time
        return predictors, targets

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
        s_time = time.time()
        # self.print("Start reorder")
        with tf.device(self.device):
            p = tf.transpose(predictors, [1, 0, 2, 3, 4])
            t = tf.transpose(targets, [1, 0, 2, 3, 4])
        self.print("Done processing")
        return p, t

    @map_decorator2_to_2
    def to_gpu(self, predictors, targets):
        p = tf.convert(predictors)
        t = tf.convert(targets)
        return p, t

    def print(self, message):
        curr_time = time.time() - self.start_time
        print(f"{curr_time:.2f}: {message}")

    """
    Dataset properties
    """
    @property
    def size_gb(self):
        size_bytes = np.product(self.predictor_shape) * 4 + np.product(self.target_shape) * 4
        return size_bytes * len(self.filenames) / 1024 ** 3

    @property
    def num_files(self):
        return len(self.filenames)

    @property
    def num_samples(self):
        return self.num_samples_per_file * self.num_files

    def __str__(self):
        s = "Dataset properties:\n"
        s += f"   Number of files: {len(self.filenames)}\n"
        s += f"   Predictor shape: {self.predictor_shape}\n"
        s += f"   Target shape: {self.target_shape}\n"
        s += f"   Dataset size: {self.size_gb:.2f} GB\n"
        return s


def get_num_parallel_calls(num_parallel_calls):
    if num_parallel_calls == "AUTOTUNE":
        return tf.data.AUTOTUNE
    else:
        return int(num_parallel_calls)

def print_gpu_usage(message="", show_line=False):
    usage = tf.config.experimental.get_memory_info("GPU:0")
    output = message + ' - '.join([f"{k}: {v / 1024**3:.2f} GB" for k,v in usage.items()])

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

    output = "current: %.2f GB - max: %.2f GB" % (
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

if __name__ == "__main__":
    main()
