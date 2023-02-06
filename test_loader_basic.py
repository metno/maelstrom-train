import argparse
import os
import resource
import time
import glob
import xarray as xr
import numpy as np
import psutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras

""" This script tests the performance of the data loader
"""

num_threads = 1
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", help="Read data from these files", nargs="*")
    parser.add_argument("-j", default=1, type=get_num_parallel_calls, help="Number of parallel calls (number or AUTO)", dest="num_parallel_calls")
    args = parser.parse_args()

    filenames = list()
    for f in args.files:
        filenames += glob.glob(f)

    loader = Loader(filenames)
    print(loader)
    dataset = loader.get_dataset(1)

    # Load all the data
    s_time = time.time()
    count = 0
    for k in dataset:
        # print(count)
        count += 1
        if count % loader.num_samples_per_file == 0:
            print(f"Done {count}: ", time.time() - s_time)

    total_time = time.time() - s_time
    print(f"TOTAL TIME: {total_time:.2f} s")
    print(f"GB/s: {loader.size_gb / total_time:.2f}")

def map_decorator1(func):
    """Decorator to wrap a 1-argument function as a tf.py_function"""
    def wrapper(self, i):
        return tf.py_function(
                lambda i: func(self, i),
                inp=(i,),
                Tout=(tf.float32, tf.float32)
                )
    return wrapper

def map_decorator2(func):
    """Decorator to wrap a 2-argument function as a tf.py_function"""
    def wrapper(self, i, j):
        return tf.py_function(
                lambda i, j: func(self, i, j),
                inp=(i, j),
                Tout=(tf.float32, tf.float32)
                )
    return wrapper

class Loader:
    def __init__(self, filenames, patch_size=16):
        self.filenames = filenames

        # Load metadata
        with xr.open_dataset(filenames[0], decode_timedelta=False) as dataset:
            self.num_leadtimes = len(dataset.variables["leadtime"])
            self.predictor_shape = dataset.variables["predictors"].shape
            self.target_shape = dataset.variables["target_mean"].shape
            self.patch_size = patch_size
            num_x_patches = self.predictor_shape[2] // self.patch_size
            num_y_patches = self.predictor_shape[2] // self.patch_size
            self.num_samples_per_file = num_x_patches * num_y_patches
        self.data = xr.open_mfdataset(self.filenames, combine="nested", concat_dim="time")

    def get_dataset(self, num_parallel_calls):
        """Returns a tf.data object"""
        dataset = tf.data.Dataset.range(len(self.filenames))
        dataset = dataset.map(self.read, num_parallel_calls)
        dataset = dataset.prefetch(1)

        # Unbatch the leadtime
        dataset = dataset.unbatch()

        # Processing steps
        dataset = dataset.map(self.feature_extraction, num_parallel_calls)
        dataset = dataset.map(self.patch, num_parallel_calls)

        # Collect leadtimes
        dataset = dataset.batch(self.num_leadtimes)

        # Put patch dimension first
        dataset = dataset.map(self.reorder, num_parallel_calls)

        # Unbatch the patch
        dataset = dataset.unbatch()
        return dataset

    @map_decorator1
    def read(self, index):
        s_time = time.time()
        index = index.numpy()
        print(f"Start reading {index}")
        # filename = self.filenames[index]
        # dataset = xr.open_dataset(filename, decode_timedelta=False)
        # predictors = dataset["predictors"]
        # targets = np.expand_dims(dataset["target_mean"], 3)
        # dataset.close()
        predictors = self.data["predictors"][index, ...]
        targets = self.data["target_mean"][index, ...]
        targets = np.expand_dims(targets, 3)
        e_time = time.time()
        print(predictors.shape, targets.shape)
        print(f"Done reading {index}: {e_time - s_time:.2f} s")
        print_memory_usage()
        return predictors, targets

    @map_decorator2
    def feature_extraction(self, predictors, targets):
        tensor = tf.concat((predictors, predictors), axis=-1)
        return predictors, targets

    @map_decorator2
    def patch(self, predictors, targets):
        """Decompose grid into patches

        Input: leadtime, y, x, predictor
        Output: leadtime, patch, y_patch, x_patch, predictor
        """
        s_time = time.time()
        # self.debug("Start patch", time.time() - self.s_time, predictors.shape)

        if self.patch_size is None:
            # A patch dimension is still needed when patching is not done
            with tf.device("CPU:0"):
                p, t = tf.expand_dims(predictors, 1),  tf.expand_dims(targets, 1)
            # self.debug(p.device)
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

        with tf.device("CPU:0"):
            p = patch_tensor(predictors, self.patch_size)
            t = patch_tensor(targets, self.patch_size)

        # self.debug("Done patching", time.time() - self.s_time, p.shape)
        return p, t

    @map_decorator2
    def reorder(self, predictors, targets):
        """Move patch dimension to be the first dimension

        Input: leadtime, patch, y, x, predictor
        Output: patch, leadtime, y, x, predictor
        """
        s_time = time.time()
        with tf.device("CPU:0"):
            p = tf.transpose(predictors, [1, 0, 2, 3, 4])
            t = tf.transpose(targets, [1, 0, 2, 3, 4])
        return p, t

    @property
    def size_gb(self):
        size_bytes = np.product(self.predictor_shape) * 4 + np.product(self.target_shape) * 4
        return size_bytes / 1024 ** 3

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

def print_memory_usage(message=None, show_line=False):
    """Prints the current and maximum memory useage of this process
    Args:
        message (str): Prepend with this message
        show_line (bool): Add the file and line number making this call at the end of message
    """

    output = "Memory usage (max): %.2f MB (%.2f MB)" % (
        get_memory_usage() / 1024 / 1024,
        get_max_memory_usage() / 1024 / 1024,
    )
    if message is not None:
        output = message + " " + output
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


if __name__ == "__main__":
    main()
