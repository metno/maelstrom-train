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
        cache_size=None,
        patch_size=None,
        predict_diff=False,
        batch_size=1,
        prefetch=None,
        num_parallel_calls=1,
        extra_features=[],
        quick_metadata=True,
        debug=False,
    ):
        self.filenames = list()
        for f in filenames:
            self.filenames += glob.glob(f)
        self.limit_predictors = limit_predictors
        self.limit_leadtimes = limit_leadtimes
        # if limit_leadtimes is not None:
        #     self.limit_leadtimes = [int(i * 3600) for i in self.limit_leadtimes]
        self.patch_size = patch_size
        self.read_metadata(self.filenames[0])
        self.logger = maelstrom.timer.Timer("test.txt")

    def get_dataset(self, num_parallel_calls=1):
        """Notes on strategies for loading data

        It doesn't really make sense to parallelize the reading across leadtimes, since we still
        have to wait for all leadtimes to finish
        """
        if False:
            # Parallelize leadtimes
            z = list(range(self.num_files * self.num_leadtimes))
        else:
            z = list(range(self.num_files))
        load_func = lambda i: tf.py_function(func=self.read_file, inp=[i], Tout=[tf.float32, tf.float32])
        # load_func = lambda i: tf.py_function(func=self.generate_fake_data, inp=[i], Tout=[tf.float32, tf.float32])
        split_func = lambda i, j: tf.py_function(func=self.split, inp=[i, j], Tout=[tf.float32, tf.float32])
        reorder_func = lambda i, j: tf.py_function(func=self.reorder, inp=[i, j], Tout=[tf.float32, tf.float32])
        patch_func = lambda i, j: tf.py_function(func=self.patch, inp=[i, j], Tout=[tf.float32, tf.float32])

        # Get a list of numbers
        dataset = tf.data.Dataset.from_generator(lambda: z, tf.uint32)

        # Load the data from one file
        dataset = dataset.map(load_func, num_parallel_calls=num_parallel_calls)

        # Split leadtime into samples
        dataset = dataset.unbatch()

        # Patch each leadtime
        dataset = dataset.map(patch_func, num_parallel_calls=num_parallel_calls)

        # Collect leadtimes
        dataset = dataset.batch(self.num_leadtimes)

        # Move patch into sample dimension
        dataset = dataset.map(reorder_func, num_parallel_calls=num_parallel_calls)

        # Split patches into samples
        dataset = dataset.unbatch()
        return dataset

    def __getitem__(self, idx):
        apss

    @property
    def num_leadtimes(self):
        pass

    @property
    def num_files(self):
        return len(self.filenames)

    @property
    def num_leadtimes(self):
        return len(self.leadtimes)

    def read_metadata(self, filename):
        dataset = xr.open_dataset(filename, decode_timedelta=False)
        self.leadtimes = dataset.leadtime
        if self.limit_leadtimes is not None:
            self.leadtimes = [dataset.leadtime[i] for i in self.limit_leadtimes]
        self.num_x = len(dataset.x)
        self.num_y = len(dataset.y)
        self.num_predictors = len(dataset.predictor) + len(dataset.static_predictor)
        if self.limit_predictors is not None:
            self.num_predictors = len(self.limit_predictors)
        print(self.num_x, self.num_y)
        dataset.close()

    def read_file(self, index, leadtime_indices=None):
        print("Loading", index)
        s_time = time.time()
        p, t = self.parse_file(self.filenames[index])
        self.logger.add("load", time.time() - s_time)
        return p, t

        # Parallelize leadtimes
        file_index = index.numpy() // self.num_leadtimes
        leadtime_indices = [index.numpy() % self.num_leadtimes]
        return self.parse_file(self.filenames[file_index], leadtime_indices)

    def generate_fake_data(self, index):
        print("Loading", index)
        s_time = time.time()
        shape = [self.num_leadtimes, self.num_y, self.num_x, self.num_predictors]
        p = np.zeros(shape, np.float32)
        t = np.zeros([self.num_leadtimes, self.num_y, self.num_x, 1], np.float32)
        self.logger.add("load", time.time() - s_time)
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
        print(filename, leadtime_indices)
        dataset = xr.open_dataset(filename, decode_timedelta=False)
        Ip = range(len(dataset.predictor))

        # Figure out which dimensions should be limited
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

        dataset = dataset.isel(**limit)

        predictors = dataset["predictors"]
        # Merge static predictors
        if len(dataset.static_predictor) > 0:
            static_predictors0 = dataset["static_predictors"]
            static_predictors = np.zeros(list(predictors.shape[0:-1]) + [static_predictors0.shape[-1]], np.float32)
            for lt in range(predictors.shape[0]):
                static_predictors[lt, ...] = static_predictors0
            predictors = np.concatenate((predictors, static_predictors), axis=3)

        targets = dataset["target_mean"]
        targets = np.expand_dims(targets, 3)

        dataset.close()
        # print("Loaded: ", tf.shape(predictors))
        return tf.convert_to_tensor(predictors), tf.convert_to_tensor(targets)
        # return predictors, targets

    def parse_file_netcdf(self, filename, leadtime_indices=None):
        """
        Args:
            filename (str): Read data from this filename
            leadtimes (list): Only read these leadtimes. If None, read all

        Returns:
            np.array: 4D array of predictors
            np.array: 4D array of observations
        """
        print(filename, leadtime_indices)
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

        return tf.convert_to_tensor(predictors), tf.convert_to_tensor(targets)

    def split(self, predictors, targets):
        pass

    def reorder(self, predictors, targets):
        s_time = time.time()
        # print("REORDER", tf.shape(predictors))
        p = tf.transpose(predictors, [1, 0, 2, 3, 4])
        t = tf.transpose(targets, [1, 0, 2, 3, 4])
        self.logger.add("reorder", time.time() - s_time)
        return p, t

    def patch(self, predictors, targets):
        s_time = time.time()
        # print("PATCHING", tf.shape(predictors))
        # p = predictors
        # t = targets

        ps = self.patch_size
        if ps is None:
            return tf.expand_dims(predictors, 0),  tf.expand_dims(targets, 0)

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
                # print("   #1", time.time() - self.s_time)

                a = tf.image.extract_patches(a, [1, ps, ps, 1], [1, ps, ps, 1], rates=[1, 1, 1, 1], padding="SAME")
                # print("   #2", time.time() - self.s_time)
                a = tf.expand_dims(a, 0)
                # print("   #3", time.time() - self.s_time)
                a = tf.reshape(a, [LT, num_patches_y * num_patches_x, ps, ps, P])
                # print("   #4", time.time() - self.s_time)
                a = tf.transpose(a, [1, 0, 2, 3, 4])
                # print("   #5", time.time() - self.s_time)
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

        p = patch_tensor(predictors, ps)
        # print("   ", time.time() - self.s_time)
        t = patch_tensor(targets, ps)
        # print(p.shape)

        self.logger.add("patch", time.time() - s_time)
        return p, t
