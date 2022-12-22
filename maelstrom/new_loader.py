import collections
import copy
import datetime
import glob
import gridpp
import json
import maelstrom
import multiprocessing
import netCDF4
import numbers
import numpy as np
import os
import re
import sys
import tensorflow as tf
import time
import tqdm
import xarray as xr
import yaml
import math


def get(config):
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

def map_decorator(func):
    def wrapper(self, i, j):
        # Use a tf.py_function to prevent auto-graph from compiling the method
        # f = func(self)
        return tf.py_function(
                lambda i, j: func(self, i, j),
                inp=(i, j),
                Tout=(tf.float32, tf.float32)
                )
    return wrapper

class Loader:
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
    ):
        self.show_debug = debug
        self.filenames = list()
        self.extra_features = extra_features
        for f in filenames:
            self.filenames += glob.glob(f)
        self.limit_predictors = limit_predictors
        self.limit_leadtimes = limit_leadtimes
        self.x_range = x_range
        self.y_range = y_range
        # if limit_leadtimes is not None:
        #     self.limit_leadtimes = [int(i * 3600) for i in self.limit_leadtimes]
        self.patch_size = patch_size
        self.predict_diff = predict_diff
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.cache = cache
        self.load_metadata(self.filenames[0])
        self.logger = maelstrom.timer.Timer("test.txt")
        self.normalization = normalization
        self.fake = fake
        self.num_parallel_calls = num_parallel_calls

        self.load_coefficients()
        print(self.coefficients)

        self.timing = collections.defaultdict(lambda: 0)
        self.s_time = time.time()

    def debug(self, *args):
        if self.show_debug:
            print(*args)

    def get_dataset(self, randomize_order=False, num_parallel_calls=1, repeat=None):
        """Notes on strategies for loading data

        It doesn't really make sense to parallelize the reading across leadtimes, since we still
        have to wait for all leadtimes to finish
        """

        if self.num_parallel_calls is not None:
            num_parallel_calls = self.num_parallel_calls

        # num_parallel_calls = tf.data.AUTOTUNE

        # Get a list of numbers
        if randomize_order:
            z = np.argsort(np.random.rand(self.num_files)).tolist()
        else:
            z = list(range(self.num_files))
        dataset = tf.data.Dataset.from_generator(lambda: z, tf.uint32)

        if repeat is not None:
            dataset = dataset.repeat(repeat)

        # Load the data from one file
        load_file = lambda i: tf.py_function(func=self.load_file, inp=[i], Tout=[tf.float32, tf.float32])
        dataset = dataset.map(load_file, num_parallel_calls=1)
        # dataset = dataset.shuffle(buffer_size=1)
        # dataset.take(1)

        # Split leadtime into samples
        if num_parallel_calls != tf.data.AUTOTUNE:
            dataset = dataset.unbatch()
            dataset = dataset.batch(math.ceil(self.num_leadtimes / num_parallel_calls))

        # Patch each leadtime
        if 0:
            dataset = dataset.map(self.process, num_parallel_calls=num_parallel_calls)
        else:
            dataset = dataset.map(self.extract_features, num_parallel_calls=num_parallel_calls)
            dataset = dataset.map(self.patch, num_parallel_calls=num_parallel_calls)
            dataset = dataset.map(self.diff, num_parallel_calls=num_parallel_calls)
            dataset = dataset.map(self.normalize, num_parallel_calls=num_parallel_calls)

        # Collect leadtimes
        if num_parallel_calls != tf.data.AUTOTUNE:
            dataset = dataset.unbatch()
            dataset = dataset.batch(self.num_leadtimes)

        # Move patch into sample dimension
        dataset = dataset.map(self.reorder, num_parallel_calls=num_parallel_calls)

        # Split patches into samples
        dataset = dataset.unbatch()

        # Cache tensors on the CPU
        if self.cache:
            dataset = dataset.cache()

        # Move tensor to GPU. We do this at the end to save GPU memory
        if self.to_gpu:
            dataset = dataset.map(self.to_gpu, num_parallel_calls=num_parallel_calls)

        # Add sample dimension
        dataset = dataset.batch(self.batch_size)

        if self.prefetch is not None:
            dataset = dataset.prefetch(self.prefetch)

        self.s_time = time.time()
        return dataset

    def __getitem__(self, idx):
        pass

    @property
    def predictor_shape(self):
        return [
            self.num_leadtimes,
            self.num_y,
            self.num_x,
            self.num_predictors,
        ]

    @property
    def target_shape(self):
        return [
            self.num_leadtimes,
            self.num_y,
            self.num_x,
            self.num_targets,
        ]

    @property
    def num_files(self):
        return len(self.filenames)

    @property
    def num_patches_per_file(self):
        return self.num_x_patches * self.num_y_patches

    @property
    def num_patches(self):
        return self.num_patches_per_file * self.num_files

    @property
    def predictor_diff_index(self):
        predictor_diff_index = None
        if self.predict_diff:
            predictor_diff_index = self.predictor_names.index("air_temperature_2m")
            return predictor_diff_index

    @property
    def num_leadtimes(self):
        return len(self.leadtimes)

    def get_dimension_limits(self, dataset, leadtime_indices=None):
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

        if self.patch_size is None:
            d["Num samples"] = self.num_files * self.num_patches_per_file
        else:
            d["Patches per file"] = self.num_patches_per_file
            d["Num patches"] = self.num_patches
            d["Patch size"] = self.patch_size
        d["Batch size"] = self.batch_size
        d["Num predictors"] = self.num_predictors
        d["Num targets"] = self.num_targets
        d["Patch size (MB)"] = self.get_data_size() / 1024**2 / self.num_patches
        d["Total size (GB)"] = self.get_data_size() / 1024**3

        d["Predictors"] = list()
        if self.predictor_names is not None:
            for q in self.predictor_names:
                d["Predictors"] += [str(q)]
        return d

    def __str__(self):
        """Returns a string representation of the dataset"""
        return json.dumps(self.description(), indent=4)

    @property
    def num_x(self):
        if self.patch_size is not None:
            return self.patch_size
        return self.num_x_input

    @property
    def num_y(self):
        if self.patch_size is not None:
            return self.patch_size
        return self.num_y_input

    @property
    def num_y_patches(self):
        if self.patch_size is not None:
            return self.num_y_input // self.patch_size
        return 1

    @property
    def num_x_patches(self):
        if self.patch_size is not None:
            return self.num_x_input // self.patch_size
        return 1

    @property
    def num_patches_per_file(self):
        return self.num_y_patches * self.num_x_patches

    @map_decorator
    def convert(self, p, t):
        p, t = tf.convert_to_tensor(p), tf.convert_to_tensor(t)
        self.debug("Convert", p.device)
        return p, t

    @map_decorator
    def to_gpu(self, p, t):
        s_time = time.time()
        p, t = tf.identity(p), tf.identity(t)
        self.debug("Moving ", p.shape, "to device", p.device)
        self.logger.add("to_gpu", time.time() - s_time)
        # print("Moved to gpu", time.time() - self.s_time, time.time() - s_time)
        return p, t

    @map_decorator
    def process(self, predictors, targets):
        p, t = self.extract_features(predictors, targets)
        p, t = self.patch(p, t)
        p, t = self.diff(p, t)
        p, t = self.normalize(p, t)
        return p, t

    def load_file(self, index, leadtime_indices=None):
        s_time = time.time()
        print("Loading", index.numpy(), self.filenames[index], time.time() - self.s_time)
        self.debug("Loading", index)
        if self.fake:
            p, t = self.generate_fake_data(index)
        else:
            p, t = self.parse_file(self.filenames[index])
        # print("Done loading", time.time() - self.s_time, time.time() - s_time)
        # maelstrom.util.print_memory_usage()
        return p, t

        # Parallelize leadtimes
        file_index = index.numpy() // self.num_leadtimes
        leadtime_indices = [index.numpy() % self.num_leadtimes]
        p, t = self.parse_file(self.filenames[file_index], leadtime_indices)
        return p, t

    def generate_fake_data(self, index):
        self.debug("Loading", index)
        s_time = time.time()
        shape = [self.num_leadtimes, self.num_y_input, self.num_x_input, self.num_input_predictors]
        p = np.zeros(shape, np.float32)
        t = np.zeros([self.num_leadtimes, self.num_y_input, self.num_x_input, 1], np.float32)
        self.logger.add("parse", time.time() - s_time)
        s_time = time.time()
        with tf.device("CPU:0"):
            p = tf.convert_to_tensor(p)
            t = tf.convert_to_tensor(t)
        self.logger.add("convert", time.time() - s_time)
        return p, t

    def parse_file(self, filename, leadtime_indices=None):
        """
        Args:
            filename (str): Read data from this filename
            leadtimes (list): Only read these leadtimes. If None, read all

        Returns:
            np.array: 4D array of predictors
            np.array: 4D array of observations
        """
        # maelstrom.util.print_memory_usage("1")
        s_time = time.time()
        self.debug("Loading", filename)
        dataset = xr.open_dataset(filename, decode_timedelta=False)
        Ip = range(len(dataset.predictor))

        # Figure out which dimensions should be limited
        limit = self.get_dimension_limits(dataset)

        dataset = dataset.isel(**limit)

        # predictors = np.zeros([59, 2321, 1796, 14], np.float32)
        # predictors[..., 0:8] = dataset["predictors"]
        predictors = dataset["predictors"]
        # Merge static predictors
        if len(dataset.static_predictor) > 0:
            static_predictors0 = dataset["static_predictors"]
            static_predictors = np.zeros(list(predictors.shape[0:-1]) + [static_predictors0.shape[-1]], np.float32)
            for lt in range(predictors.shape[0]):
                static_predictors[lt, ...] = static_predictors0
            predictors = np.concatenate((predictors, static_predictors), axis=3)
            # predictors[..., 8:14] = static_predictors

        targets = dataset["target_mean"]
        targets = np.expand_dims(targets, 3)

        dataset.close()

        # p, t = predictors, targets
        self.debug("Finished parsing", time.time() - s_time)
        self.logger.add("parse", time.time() - s_time)
        s_time = time.time()
        with tf.device("CPU:0"):
            t = tf.convert_to_tensor(targets)
            del targets
            # s = tf.convert_to_tensor(static_predictors)
            # del static_predictors
            p = tf.convert_to_tensor(predictors)
            del predictors
        self.debug("Convert", time.time() - s_time)
        self.logger.add("convert", time.time() - s_time)
        # maelstrom.util.print_memory_usage()
        return p, t

    def parse_file_netcdf(self, filename, leadtime_indices=None):
        """
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

    @map_decorator
    def split(self, predictors, targets):
        pass

    @map_decorator
    def reorder(self, predictors, targets):
        self.debug("Reorder start", time.time() - self.s_time, predictors.shape)
        s_time = time.time()
        with tf.device("CPU:0"):
            p = tf.transpose(predictors, [1, 0, 2, 3, 4])
            t = tf.transpose(targets, [1, 0, 2, 3, 4])
        self.logger.add("reorder", time.time() - s_time)
        self.debug("Done reordering", time.time() - self.s_time, p.device)
        return p, t

    @map_decorator
    def extract_features(self, predictors, targets):
        # Input: LT, Y, X, P
        s_time = time.time()
        with tf.device("CPU:0"):
            p = [predictors]
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
        self.logger.add("extract", time.time() - s_time)
        return p, targets

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

    @map_decorator
    def diff(self, predictors, targets):
        # Input: (..., )
        s_time = time.time()
        if self.predictor_diff_index is None:
            return predictors, targets
        Ip = self.predictor_diff_index
        v = tf.expand_dims(predictors[..., Ip], -1)
        t = tf.math.subtract(targets, v)
        self.logger.add("diff", time.time() - s_time)
        return predictors, t

    @map_decorator
    def normalize(self, predictors, targets):
        # Input: LT, S, Y, X, P
        s_time = time.time()
        if self.coefficients is None:
            self.logger.add("normalize", time.time() - s_time)
            return predictors, targets

        self.debug("Normalize", predictors.shape, self.coefficients.shape)

        with tf.device("CPU:0"):
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

            a = expand_array(a, shape)
            s = expand_array(s, shape)

            p = tf.math.subtract(predictors, a)
            p = tf.math.divide(p, s)

        self.logger.add("normalize", time.time() - s_time)
        return p, targets

    @map_decorator
    def patch(self, predictors, targets):
        s_time = time.time()
        self.debug("Start patch", time.time() - self.s_time, predictors.shape)
        # print("   Start patching", time.time() - self.s_time)
        # p = predictors
        # t = targets

        ps = self.patch_size
        if ps is None:
            # self.debug("Done patching", time.time() - self.s_time)
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
                # self.debug("   #1", time.time() - self.s_time)

                a = tf.image.extract_patches(a, [1, ps, ps, 1], [1, ps, ps, 1], rates=[1, 1, 1, 1], padding="SAME")
                # self.debug("   #2", time.time() - self.s_time)
                a = tf.expand_dims(a, 0)
                # self.debug("   #3", time.time() - self.s_time)
                a = tf.reshape(a, [LT, num_patches_y * num_patches_x, ps, ps, P])
                # self.debug("   #4", time.time() - self.s_time)
                # a = tf.transpose(a, [1, 0, 2, 3, 4])
                # self.debug("   #5", time.time() - self.s_time)
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
                # a = tf.transpose(a, [1, 0, 2, 3, 4])
                return a

        with tf.device("CPU:0"):
            p = patch_tensor(predictors, ps)
            t = patch_tensor(targets, ps)

        self.logger.add("patch", time.time() - s_time)
        self.debug("Done patching", time.time() - self.s_time, p.shape)
        return p, t

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
        size_per_patch = (
            4
            * (np.product(self.predictor_shape) + np.product(self.target_shape))
        )
        return size_per_patch * self.num_patches
