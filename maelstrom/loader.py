import collections
import glob
import json
import math
import netCDF4
import numpy as np
import tensorflow as tf
import time
import xarray as xr
import yaml

import maelstrom
from decorator import *

class Loader:
    """Data loader class

    Use get_dataset() to get a streaming tf.data object
    """

    def __init__(
        self,
        filenames,
        limit_leadtimes=None,
        limit_predictors=None,
        x_range=None,
        y_range=None,
        probabilistic_target=False,
        normalization=None,
        patch_size=None,
        predict_diff=False,
        batch_size=1,
        prefetch=None,
        cache=False,
        num_parallel_calls=None,
        extra_features=[],
        quick_metadata=True,
        debug=False,
        fake=False,
        to_gpu=True,
        with_leadtime=True,
    ):
        """Initialize data loader
        
        Args:
            filenames (list): List of netCDF files to load
            limit_leadtimes (list): Only retrieve these leadtimes
            limit_predictors (list): Only retrieve these predictor names
            x_range (list): Only retrieve these x-axis indices
            y_range (list): Only retrieve these y-axis indices
            probabilistic_target (bool): Load both target mean and uncertainty
            normalization (str): Filename with normalization data
            patch_size (int): Patch the data with a stencil of this width (pixels)
            predict_diff (bool): Change the prediction problem to estimate the forecast bias
            batch_size (int): Number of samples to use per batch
            prefetch (int): Number of batches to prefetch
            cache (bool): Cache data in memory (before moving to GPU)
            num_parallel_calls (int): Number of threads to use for each pipeline stage
            extra_features (dict): Configuration of extra features to generate
            quick_metadata (bool): Deduce date metadata from filename, instead of reading the file
            debug (bool): Turn on debugging information
            fake (bool): Generate fake data, instead of reading from file
            to_gpu (bool): Move final tensors to GPU in the data processing pipeline
            with_leadtime (bool): Generate data with the whole leadtime dimension
        """
        self.show_debug = debug
        self.filenames = list()
        self.extra_features = extra_features
        for f in filenames:
            self.filenames += glob.glob(f)
        print(self.filenames)
        self.limit_predictors = limit_predictors
        self.limit_leadtimes = limit_leadtimes
        self.x_range = x_range
        self.y_range = y_range
        self.patch_size = patch_size
        self.predict_diff = predict_diff
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.cache = cache
        self.load_metadata(self.filenames[0])
        self.timing = maelstrom.timer.Timer("test.txt")
        self.normalization = normalization
        self.fake = fake
        self.num_parallel_calls = num_parallel_calls

        self.load_coefficients()

        self.count_reads = 0
        self.count_start_processing = 0
        self.count_done_processing = 0

        self.dataset = xr.open_mfdataset(self.filenames, combine="nested", concat_dim="time")
        limit = self.get_dimension_limits(self.dataset)
        self.dataset = self.dataset.isel(**limit)

        # Initialize a timer so that we can track overall processing time
        self.device = "CPU:0"
        self.start_time = time.time()
        self.with_leadtime = with_leadtime

        # Cache the normalization coefficients
        self.normalize_add = None
        self.normalize_factor = None

    @staticmethod
    def from_config(config):
        """Returns a Loader object based on a configuration dictionary"""
        kwargs = {k:v for k,v in config.items() if k != "type"}
        range_variables = ["x_range", "y_range", "limit_leadtimes"]

        # Process value arguments
        for range_variable in range_variables:
            if range_variable in kwargs:
                curr = kwargs[range_variable]
                if isinstance(curr, str):
                    if curr.find(":") == -1:
                        raise ValueError(
                            f"Cannot interpret range string {curr}. Should be in the form start:end"
                        )
                    start, end = curr.split(":")
                    kwargs[range_variable] = range(int(start), int(end))
        return Loader(**kwargs)

    def get_dataset(self, randomize_order=False, num_parallel_calls=1, repeat=None, shard_size=None, shard_index=None):
        """Returns a tf.data object that streams data from disk

        Args:
            randomize_order (bool): Randomize the order that data is retrieved in
            num_parallel_calls (int): How many threads to process data with. Can also be
                tf.data.AUTOTUNE
            repeat (int): Repeat the dataset this many times

        Returns:
            tf.data: Dataset
        """
        if self.num_parallel_calls is not None:
            num_parallel_calls = self.num_parallel_calls

        if shard_size is None:
            dataset = tf.data.Dataset.range(self.num_files)
        else:
            start = shard_index * math.ceil(self.num_files // shard_size)
            end = (shard_index + 1) * math.ceil(self.num_files // shard_size)
            print("SHARD", shard_index, start, end)
            dataset = tf.data.Dataset.range(start, end)
        if randomize_order:
            dataset = dataset.shuffle(10000)

        if repeat is not None:
            dataset = dataset.repeat(repeat)

        # Read the data from file
        dataset = dataset.map(self.read, num_parallel_calls=1)
        dataset = dataset.map(self.expand_static_predictors, num_parallel_calls=1)

        # Rebatch the leadtime
        dataset = dataset.unbatch()
        if self.with_leadtime:
            # if num_parallel_calls != tf.data.AUTOTUNE:
            #     dataset = dataset.batch(math.ceil(self.num_leadtimes / num_parallel_calls))
            dataset = dataset.batch(1)
        else:
            dataset = dataset.batch(1)
        # Shape: (leadtime, y, x, predictor)

        dataset = dataset.map(self.print_start_processing)

        # Split leadtime into samples
        # if num_parallel_calls != tf.data.AUTOTUNE:
        #     dataset = dataset.unbatch()
        #     dataset = dataset.batch(math.ceil(self.num_leadtimes / num_parallel_calls))
        #     # Shape: (leadtime, y, x, predictor)

        # Perform various processing steps
        if 0:
            dataset = dataset.map(self.process, num_parallel_calls=num_parallel_calls)
        else:
            # Same steps but split into smaller steps
            dataset = dataset.map(self.extract_features, num_parallel_calls=num_parallel_calls)
            # Shape: (leadtime, y, x, predictor)
            dataset = dataset.map(self.patch, num_parallel_calls=num_parallel_calls)
            # Shape: (leadtime, patch, y_patch, x_patch, predictor)
            dataset = dataset.map(self.diff, num_parallel_calls=num_parallel_calls)
            # Shape: (leadtime, patch, y_patch, x_patch, predictor)
            dataset = dataset.map(self.normalize, num_parallel_calls=num_parallel_calls)
        # Shape: (leadtime, patch, y_patch, x_patch, predictor)

        # Collect leadtimes
        if self.with_leadtime:
            dataset = dataset.unbatch()
            dataset = dataset.batch(self.num_leadtimes)
            # Shape: (leadtime, patch, y_patch, x_patch, predictor)

            # Move patch into sample dimension
            dataset = dataset.map(self.reorder, num_parallel_calls=num_parallel_calls)
            # dataset = dataset.map(self.print_done_processing)
            # Shape: (patch, leadtime, y_patch, x_patch, predictor)

            # Split patches into samples
            dataset = dataset.unbatch()
        else:
            dataset = dataset.unbatch()
            dataset = dataset.unbatch()
            dataset = dataset.batch(1)
        # Shape: (leadtime, y_patch, x_patch, predictor)

        # Cache tensors on the CPU (we don't want caching on GPU)

        # Add sample dimension
        dataset = dataset.batch(self.batch_size)
        # Shape: (batch_size, leadtime, y_patch, x_patch, predictor)

        # Move tensor to GPU. We do this at the end to save GPU memory
        if self.to_gpu:
            dataset = dataset.map(self.to_gpu, num_parallel_calls=num_parallel_calls)

        if self.prefetch is not None:
            dataset = dataset.prefetch(self.prefetch)

        dataset = dataset.prefetch(1)

        # dataset = dataset.cache()
        if self.cache:
            dataset = dataset.cache()

        self.start_time = time.time()
        return dataset


    """
    Functions for getting dataset metadata
    """
    @property
    def predictor_shape(self):
        """The shape of a predictor for one sample"""
        if self.with_leadtime:
            return [
                self.num_leadtimes,
                self.num_y,
                self.num_x,
                self.num_predictors,
            ]
        else:
            return [
                1,
                self.num_y,
                self.num_x,
                self.num_predictors,
            ]

    @property
    def target_shape(self):
        if self.with_leadtime:
            return [
                self.num_leadtimes,
                self.num_y,
                self.num_x,
                self.num_targets,
            ]
        else:
            return [
                1,
                self.num_y,
                self.num_x,
                self.num_targets,
            ]

    @property
    def num_files(self):
        """Returns the total number of files in the dataset"""
        return len(self.filenames)

    @property
    def num_patches_per_file(self):
        """Returns the number of patches in each NetCDF file"""
        return self.num_x_patches * self.num_y_patches

    @property
    def num_patches(self):
        """Returns the total number of patches in the dataset"""
        return self.num_patches_per_file * self.num_files

    @property
    def num_leadtimes(self):
        """Returns the number of leadtimes in the dataset"""
        return len(self.leadtimes)

    @property
    def num_samples_per_file(self):
        if self.with_leadtime:
            return self.num_patches_per_file
        else:
            return self.num_patches_per_file * self.num_leadtimes

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
    def batch_predictor_shape(self):
        shape = [self.batch_size] + [i for i in self.predictor_shape]
        if not self.with_leadtime:
            shape[1] = 1
        shape[2] = self.patch_size
        shape[3] = self.patch_size
        shape[-1] = self.predictor_shape[-1] + self.static_predictor_shape[-1] + self.num_extra_features
        return shape

    @property
    def num_x(self):
        """Returns the number of x-axis points in the dataset"""
        if self.patch_size is not None:
            return self.patch_size
        return self.num_x_input

    @property
    def num_y(self):
        """Returns the number of y-axis points in the dataset"""
        if self.patch_size is not None:
            return self.patch_size
        return self.num_y_input

    @property
    def num_y_patches(self):
        """Returns the number of patches along the y-axis"""
        if self.patch_size is not None:
            return self.num_y_input // self.patch_size
        return 1

    @property
    def num_x_patches(self):
        """Returns the number of patches along the x-axis"""
        if self.patch_size is not None:
            return self.num_x_input // self.patch_size
        return 1

    """
    Functions used by the data processing pipeline
    """
    @map_decorator1_to_3
    def read(self, index):
        """Read data from NetCDF

        Args:
            index (int): Read data from this time index

        Returns:
            tf.Tensor: 4D array of predictors
            tf.Tensor: 4D array of static_predictors
            tf.Tensor: 4D array of observations
        """
        s_time = time.time()
        index = index.numpy()
        # self.print(f"Start reading index={index}")
        with tf.device(self.device):
            predictors = self.dataset["predictors"][index, ...].values
            static_predictors = self.dataset["static_predictors"][index, ...].values
            targets = self.dataset["target_mean"][index, ...].values
            targets = tf.expand_dims(targets, -1)

            # Force explicit conversion here, so that we can account the time it takes
            # Otherwise the conversion happens when the function returns
            predictors = tf.convert_to_tensor(predictors)
            static_predictors = tf.convert_to_tensor(static_predictors)
            targets = tf.convert_to_tensor(targets)

        e_time = time.time()
        self.timing.add("read", time.time() - s_time)
        return predictors, static_predictors, targets

    @map_decorator3_to_3
    def expand_static_predictors(self, predictors, static_predictors, targets):
        """Copies static predictors to leadtime dimension"""
        s_time = time.time()
        # self.print("Start processing")
        # print(predictors.shape, static_predictors.shape, targets.shape)
        with tf.device(self.device):
            shape = [predictors.shape[0], 1, 1, 1]
            static_predictors = tf.expand_dims(static_predictors, 0)
            static_predictors = tf.tile(static_predictors, shape)
        self.timing.add("expand", time.time() - s_time)
        return predictors, static_predictors, targets


    @map_decorator3_to_2
    def process(self, predictors, static_predictors, targets):
        """Perform all processing steps in one function.

        This is an alterative to calling each function separately.
        """
        s_time = time.time()
        p, t = self.extract_features(predictors, static_predictors, targets)
        p, t = self.patch(p, t)
        p, t = self.diff(p, t)
        p, t = self.normalize(p, t)
        # print("End process", time.time() - s_time)
        return p, t

    @map_decorator3_to_2
    def extract_features(self, predictors, static_predictors, targets):
        """Extract features and append to predictors

        Input: leadtime, y, x, predictor
        Output: leadtime, y, x, predictor
        """
        s_time = time.time()
        with tf.device("CPU:0"):
            p = [predictors, static_predictors]
            shape = predictors.shape
            for f, feature in enumerate(self.extra_features):
                feature_type = feature["type"]
                if feature_type == "x":
                    x = tf.range(shape[2], dtype=tf.float32)
                    curr = self.broadcast(x, shape, 2)
                elif feature_type == "y":
                    x = tf.range(shape[1], dtype=tf.float32)
                    curr = self.broadcast(x, shape, 1)
                elif feature_type == "leadtime":
                    x = tf.range(shape[0], dtype=tf.float32)
                    curr = self.broadcast(x, shape, 0)
                curr = tf.convert_to_tensor(curr)
                curr = tf.expand_dims(curr, -1)
                p += [curr]

            p = tf.concat(p, axis=3)
        self.timing.add("extract", time.time() - s_time)
        return p, targets

    @map_decorator2_to_2
    def patch(self, predictors, targets):
        """Decompose grid into patches

        Input: leadtime, y, x, predictor
        Output: leadtime, patch, y_patch, x_patch, predictor
        """
        s_time = time.time()
        self.debug("Start patch", time.time() - self.start_time, predictors.shape)

        if self.patch_size is None:
            # A patch dimension is still needed when patching is not done
            with tf.device("CPU:0"):
                p, t = tf.expand_dims(predictors, 1),  tf.expand_dims(targets, 1)
            self.debug(p.device)
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

        self.timing.add("patch", time.time() - s_time)
        self.debug("Done patching", time.time() - self.start_time, p.shape)
        return p, t

    @map_decorator2_to_2
    def diff(self, predictors, targets):
        """Subtract the raw forecast from predictors and targets

        Input: leadtime, patch, y, x, predictor
        Output: leadtime, patch, y, x, predictor
        """
        s_time = time.time()
        if self.raw_predictor_index is None:
            return predictors, targets
        Ip = self.raw_predictor_index
        v = tf.expand_dims(predictors[..., Ip], -1)
        t = tf.math.subtract(targets, v)
        self.timing.add("diff", time.time() - s_time)
        return predictors, t

    @map_decorator2_to_2
    def normalize(self, predictors, targets):
        """Normalize predictors

        Input: leadtime, patch, y, x, predictor
        Output: leadtime, patch, y, x, predictor
        """
        s_time = time.time()
        if self.coefficients is None:
            self.timing.add("normalize", time.time() - s_time)
            return predictors, targets

        self.debug("Normalize", predictors.shape, self.coefficients.shape)

        with tf.device("CPU:0"):
            if self.normalize_add is None or self.normalize_factor is None:
                a = self.coefficients[:, 0]
                s = self.coefficients[:, 1]
                shape = tf.concat((tf.shape(predictors)[0:-1], [1]), 0)

                def expand_array(a, shape):
                    """Expands array a so that it has the shape"""
                    if 1:
                        # Use if unbatch has not been run
                        a = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(a, 0), 0), 0), 0)
                    else:
                        a = tf.expand_dims(tf.expand_dims(tf.expand_dims(a, 0), 0), 0)
                    a = tf.tile(a, shape)
                    return a

                self.normalize_add = expand_array(a, shape)
                self.normalize_factor = expand_array(s, shape)

            p = tf.math.subtract(predictors, self.normalize_add)
            p = tf.math.divide(p, self.normalize_factor)

        self.timing.add("normalize", time.time() - s_time)
        return p, targets

    @map_decorator2_to_2
    def reorder(self, predictors, targets):
        """Move patch dimension to be the first dimension

        Input: leadtime, patch, y, x, predictor
        Output: patch, leadtime, y, x, predictor
        """
        self.debug("Reorder start", time.time() - self.start_time, predictors.shape)
        s_time = time.time()
        with tf.device("CPU:0"):
            p = tf.transpose(predictors, [1, 0, 2, 3, 4])
            t = tf.transpose(targets, [1, 0, 2, 3, 4])
        self.timing.add("reorder", time.time() - s_time)
        self.debug("Done reordering", time.time() - self.start_time, p.device)
        return p, t

    @map_decorator2_to_2
    def to_gpu(self, p, t):
        """Copy tensors to GPU"""
        s_time = time.time()
        # p, t = tf.identity(p), tf.identity(t)
        p = tf.convert_to_tensor(p)
        t = tf.convert_to_tensor(t)
        self.debug("Moving ", p.shape, "to device", p.device)
        self.timing.add("to_gpu", time.time() - s_time)
        # print("TO GPU", p.shape, t.shape)
        # print("Moved to gpu", time.time() - self.start_time, time.time() - s_time)
        return p, t

    @map_decorator2_to_2
    def print_shape(self, p, t):
        """Helper function to print out the shape of tensors"""
        print(p.shape, t.shape)
        return p, t

    @map_decorator3_to_3
    def print_start_processing(self, p, s, t):
        # print("%.4f" % (time.time() - self.start_time), "Start processing", self.count_start_processing)
        self.count_start_processing += 1
        return p, s, t

    @map_decorator2_to_2
    def print_done_processing(self, p, t):
        print("%.4f" % (time.time() - self.start_time), "Done processing", self.count_done_processing)
        self.count_done_processing += 1
        return p, t

    @map_decorator1_to_1
    def print_start_time(self, p):
        print("Start", p.numpy(), "%.4f" % (time.time() - self.start_time))
        return p

    @property
    def raw_predictor_index(self):
        """Returns the predictor index corresponding to the raw forecast"""
        raw_predictor_index = None
        if self.predict_diff:
            raw_predictor_index = self.predictor_names.index("air_temperature_2m")
            return raw_predictor_index

    @map_decorator2_to_2
    def convert(self, p, t):
        """Convert numpy arrays to tensors"""
        p, t = tf.convert_to_tensor(p), tf.convert_to_tensor(t)
        self.debug("Convert", p.device)
        return p, t

    """
    Various helper functions
    """
    @staticmethod
    def broadcast(tensor, final_shape, axis):
        if axis == 2:
            tensor = tf.expand_dims(tf.expand_dims(tensor, 0), 1)
            ret = tf.tile(tensor, [final_shape[0], final_shape[1], 1])
        elif axis == 1:
            tensor = tf.expand_dims(tf.expand_dims(tensor, 0), 2)
            ret = tf.tile(tensor, [final_shape[0], 1, final_shape[2]])
        else:
            tensor = tf.expand_dims(tf.expand_dims(tensor, 1), 2)
            ret = tf.tile(tensor, [1, final_shape[1], final_shape[2]])
        # new_shape = tf.transpose(final_shape, [axis, -1])
        # ret = tf.broadcast_to(tensor, new_shape)
        # ret = tf.transpose(ret, -1, axis)
        return ret

    def get_dimension_limits(self, dataset, leadtime_indices=None):
        """Returns a dictionary containing the indices that each dimension should be limited by"""
        limit = dict()
        if self.limit_predictors is not None:
            limit["predictor"] = [i for i in range(len(dataset.predictor)) if dataset.predictor[i] in self.limit_predictors]
            limit["static_predictor"] = [i for i in range(len(dataset.static_predictor)) if dataset.static_predictor[i] in self.limit_predictors]
        if leadtime_indices is None:
            if self.limit_leadtimes is not None:
                # limit["leadtime"] = [i for i in range(len(dataset.leadtime)) if dataset.leadtime[i] in self.limit_leadtimes]
                limit["leadtime"] = self.limit_leadtimes
        else:
            if self.limit_leadtimes is not None:
                limit["leadtime"] = [dataset.leadtime[i] for i in leadtime_indices]
            else:
                limit["leadtime"] = leadtime_indices
        if self.x_range is not None:
            limit["x"] = self.x_range
        if self.y_range is not None:
            limit["y"] = self.y_range
        return limit

    def load_metadata(self, filename):
        """Reads matadata from one file and stores relevant information in self"""
        dataset = xr.open_dataset(filename, decode_timedelta=False)
        limit = self.get_dimension_limits(dataset)
        dataset = dataset.isel(**limit)

        self.leadtimes = dataset.leadtime.to_numpy()
        # if self.limit_leadtimes is not None:
        #     self.leadtimes = [dataset.leadtime[i] for i in self.limit_leadtimes]
        self.num_x_input = len(dataset.x)
        self.num_y_input = len(dataset.y)
        if self.patch_size is not None:
            if self.patch_size > self.num_x_input:
                raise ValueError("Patch size too small")
            if self.patch_size > self.num_y_input:
                raise ValueError("Patch size too small")
        self.num_input_predictors = len(dataset.predictor) + len(dataset.static_predictor)
        self.num_predictors = self.num_input_predictors + len(self.extra_features)
        self.predictor_names_input = [p for p in dataset.predictor.to_numpy()] + [p for p in dataset.static_predictor.to_numpy()]
        self.predictor_names = self.predictor_names_input + [self.get_feature_name(p) for p in self.extra_features]
        if self.limit_predictors is not None:
            self.num_predictors = len(self.limit_predictors)

        self.num_targets = 1

        dataset.close()

    def description(self):
        d = dict()
        d["Predictor shape"] = ", ".join(["%d" % i for i in self.predictor_shape])
        d["Target shape"] = ", ".join(["%d" % i for i in self.target_shape])
        d["Num files"] = self.num_files
        d["Num leadtimes"] = self.num_leadtimes

        if self.patch_size is None:
            d["Num samples"] = self.num_files * self.num_patches_per_file
        else:
            d["Patches per file"] = self.num_patches_per_file
            d["Samples per file"] = self.num_samples_per_file
            d["Num patches"] = self.num_patches
            d["Num samples"] = self.num_samples
            d["Patch size"] = self.patch_size
            d["Num batches"] = self.num_batches
            d["Batches per file"] = self.num_batches_per_file
        d["Batch size"] = self.batch_size
        d["Num predictors"] = self.num_predictors
        d["Num targets"] = self.num_targets
        d["Batch size (MB)"] = self.get_data_size() / 1024**2 / self.num_samples * self.batch_size
        d["Total size (GB)"] = self.get_data_size() / 1024**3

        d["Predictors"] = list()
        if self.predictor_names is not None:
            for q in self.predictor_names:
                d["Predictors"] += [str(q)]
        return d

    def __str__(self):
        """Returns a string representation of the dataset"""
        return json.dumps(self.description(), indent=4)

    def load_coefficients(self):
        self.coefficients = None
        if self.normalization is not None:
            self.coefficients = np.zeros([self.num_predictors, 2], np.float32)
            self.coefficients[:, 1] = 1
            with open(self.normalization) as file:
                coefficients = yaml.load(file, Loader=yaml.SafeLoader)
                # Add normalization information for the extra features
                for k,v in self.get_extra_features_normalization(self.extra_features).items():
                    coefficients[k] = v

                for i, name in enumerate(self.predictor_names):
                    if name in coefficients:
                        self.coefficients[i, :] = coefficients[name]
                    elif name in self.extra_features:
                        sel.coefficients[i, :] = [0, 1]
                    else:
                        self.coefficients[i, :] = [0, 1]

    def get_extra_features_normalization(self, extra_features):
        normalization = dict()
        X = self.num_x_input
        Y = self.num_y_input
        for feature in extra_features:
            curr = [0, 1]
            feature_name = self.get_feature_name(feature)
            feature_type = feature["type"]
            if feature_type == "x":
                val = np.arange(X)
                curr = [np.mean(val), np.std(val)]
            elif feature_type == "y":
                val = np.arange(Y)
                curr = [np.mean(val), np.std(val)]
            elif feature_type == "leadtime":
                val = np.arange(self.num_leadtimes)
                curr = [np.mean(val), np.std(val)]

            normalization[feature_name] = curr
        return normalization

    def get_feature_name(self, feature):
        if "name" in feature:
            return feature["name"]
        else:
            return feature["type"]

    def get_data_size(self):
        """Returns the number of bytes needed to store the full dataset"""
        size_per_sample = (
            4
            * (np.product(self.predictor_shape) + np.product(self.target_shape))
        )
        return size_per_sample * self.num_samples

    def debug(self, *args):
        if self.show_debug:
            print(*args)

    def parse_file_netcdf(self, filename, leadtime_indices=None):
        """ Old way of reading data using NetCDF4 library

        Don't use, only provided for reference.

        Args:
            filename (str): Read data from this filename
            leadtimes (list): Only read these leadtimes. If None, read all

        Returns:
            np.array: 4D array of predictors
            np.array: 4D array of observations
        """
        s_time = time.time()
        self.debug("Loading", filename)
        with netCDF4.Dataset(filename) as dataset:

            dims = dataset.dimensions
            Ip = range(len(dims["predictor"]))

            # Figure out which dimensions should be limited
            limit = dict()
            Ip = slice(0, len(dims["predictor"]))
            Ips = slice(0, len(dims["static_predictor"]))
            if self.limit_predictors is not None:
                Ip = [i for i in range(len(dims["predictor"])) if dataset.variables["predictor"][i] in self.limit_predictors]
                Ips = [i for i in range(len(dims["static_predictor"])) if dataset.variables["static_predictor"][i] in self.limit_predictors]

            It = slice(0, len(dims["leadtime"]))
            if leadtime_indices is None:
                if self.limit_leadtimes is not None:
                    # limit["leadtime"] = [i for i in range(len(dataset.leadtime)) if dataset.leadtime[i] in self.limit_leadtimes]
                    It = self.limit_leadtimes
            else:
                if self.limit_leadtimes is not None:
                    It = [dataset.variabls["leadtime"][i] for i in leadtime_indices]
                else:
                    It = leadtime_indices

            predictors = dataset.variables["predictors"][It, :, :, Ip]
            # Merge static predictors
            if len(dataset.dimensions["static_predictor"]) > 0:
                static_predictors0 = dataset.variables["static_predictors"][It, :, :, Ips]
                static_predictors = np.zeros(list(predictors.shape[0:-1]) + [static_predictors0.shape[-1]], np.float32)
                for lt in range(predictors.shape[0]):
                    static_predictors[lt, ...] = static_predictors0
                predictors = np.concatenate((predictors, static_predictors), axis=3)

            targets = dataset.variables["target_mean"][It, :, :]
            targets = np.expand_dims(targets, 3)

        p, t = predictors, targets
        self.debug("Finished parsing", time.time() - s_time)
        # p, t = tf.convert_to_tensor(predictors), tf.convert_to_tensor(targets)
        # self.debug("Convert", time.time() - s_time)
        return p, t

    def print(self, message):
        curr_time = time.time() - self.start_time
        print(f"{curr_time:.2f}: {message}")

    @property
    def size_gb(self):
        return self.get_data_size() / 1024**3
