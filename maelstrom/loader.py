import collections
import glob
import json
import math
import netCDF4
import numpy as np
import os
import tensorflow as tf
import time
import xarray as xr
import yaml

import maelstrom

with_horovod = maelstrom.check_horovod()
if with_horovod:
    import horovod.tensorflow as hvd

class Loader:
    """Data loader class

    Use get_dataset() to get a streaming tf.data object
    """

    def __init__(
        self,
        filenames,
        omit_filenames=[],
        limit_leadtimes=None,
        limit_predictors=None,
        x_range=None,
        y_range=None,
        probabilistic_target=False,
        normalization=None,
        patch_size=None,
        predict_diff=False,
        batch_size=1,
        filename_validation_cache=None,
        num_parallel_calls=None,
        cache=False,
        extra_features=[],
        quick_metadata=True,
        debug=False,
        create_fake_data=False,
        to_gpu=True,
        with_leadtime=True,
        with_horovod=False,
        rank_split_strategy="blocks",
        interleave_leadtime=False,
    ):
        """Initialize data loader
        
        Args:
            filenames (list): List of netCDF files to load
            limit_leadtimes (list): Only retrieve these leadtimes
            limit_predictors (list): Only retrieve these predictor names
            x_range (list): Only retrieve these x-axis indices (start, stop) (python range)
            y_range (list): Only retrieve these y-axis indices (start, stop) (python range)
            probabilistic_target (bool): Load both target mean and uncertainty
            normalization (str): Filename with normalization data
            patch_size (int): Patch the data with a stencil of this width (pixels)
            predict_diff (bool): Change the prediction problem to estimate the forecast bias
            batch_size (int): Number of samples to use per batch
            filename_validation_cache (str): Cache validation data in this file
            cache (bool): Cache the dataset?
            num_parallel_calls (int): Number of threads to use for each pipeline stage
            extra_features (dict): Configuration of extra features to generate
            quick_metadata (bool): Deduce date metadata from filename, instead of reading the file
            debug (bool): Turn on debugging information
            create_fake_data (bool): Generate fake data, instead of reading from file
            to_gpu (bool): Move final tensors to GPU in the data processing pipeline
            with_leadtime (bool): Include leadtime dimension in one sample
            with_horovod (bool): Deal with horovod?
            rank_split_strategy (str): How to split the dataset across ranks
            interleave_leadtime (bool): If True, then read leadtimes randomly
        """
        self.debug = debug
        self.extra_features = extra_features
        self.limit_predictors = limit_predictors
        self.limit_leadtimes = limit_leadtimes
        self.with_leadtime = with_leadtime
        self.patch_size = patch_size
        self.predict_diff = predict_diff
        self.batch_size = batch_size
        self.filename_validation_cache = filename_validation_cache
        self.cache = cache
        self.filename_normalization = normalization
        self.create_fake_data = create_fake_data
        self.num_parallel_calls = num_parallel_calls
        self.probabilistic_target = probabilistic_target
        self.with_horovod = with_horovod
        self.rank_split_strategy = rank_split_strategy
        self.interleave_leadtime = interleave_leadtime

        self.x_range = x_range
        self.y_range = y_range
        if self.x_range is not None and self.y_range is not None:
            if len(self.x_range) != 2:
                raise ValueError("x_range must be a 2-vector (start,stop)")
            if len(self.y_range) != 2:
                raise ValueError("y_range must be a 2-vector (start,stop)")
        elif not (self.x_range is None and self.y_range is None):
            raise ValueError("Either both or none of x_range and y_range must be provided")

        if self.filename_validation_cache is not None:
            if os.path.exists(self.filename_validation_cache):
                os.remove(self.filename_validation_cache)

        self.filenames = list()
        for f in filenames:
            self.filenames += glob.glob(f)
        self.filenames = [f for f in self.filenames if f not in omit_filenames]

        if self.with_horovod:
            if len(self.filenames) == 0:
                raise Exception(f"Too few files ({len(self.filenames)}) to divide into {hvd.size()} processes")

            if self.rank_split_strategy == "random":
                """ Randonly assign filenames to ranks. This isn't properly implemented, since we
                are not guaranteed that all files are placed.
                """
                I = np.argssort(np.random.rand(len(self.filenames)))
                self.filenames = [self.filenames[I[f]] for f in range(start, end)]

            elif self.rank_split_strategy == "blocks":
                """ Shard into hvd.size() blocks. This gives each rank a very seasonal subset. The
                advantage with this is that we ensure every batch has one file from each season,
                since gradients are average across ranks. It shouldn't be a problem that each rank
                gets an inhomogenous dataset.
                """
                start = hvd.rank() * math.ceil(len(self.filenames) // hvd.size())
                end = (hvd.rank() + 1) * math.ceil(len(self.filenames) // hvd.size())
                if end > len(self.filenames):
                    end = len(self.filenames)
                self.filenames = [self.filenames[f] for f in range(start, end)]

            elif self.rank_split_strategy == "interleaved":
                """Send every hvd.size()'th file to a given rank. This should give each rank a good
                mix of dates.
                """
                self.filenames = [self.filenames[f] for f in range(hvd.rank(), len(self.filenames), hvd.size())]

            else:
                raise ValueError(f"Invalid rank split strategy {self.rank_split_strategy}")

            if len(self.filenames) == 0:
                raise Exception(f"Too few files ({len(self.filenames)}) to divide into {hvd.size()} processes")

        self.logger = maelstrom.timer.Timer("test.txt")
        self.timing = collections.defaultdict(lambda: 0)
        self.count_reads = 0
        self.count_start_processing = 0
        self.count_done_processing = 0
        # Initialize a timer so that we can track overall processing time
        self.start_time = time.time()

        # Where should data reside during the processing steps? Processing seems faster on CPU,
        # perhaps because the pipeline stages can run in parallel better than on the GPU?
        self.device = "CPU:0"

        # Set up dataset
        # cache=False seems to have no effect
        self.data = xr.open_mfdataset(self.filenames, decode_timedelta=False, decode_times=False, combine="nested", concat_dim="time") # , cache=False)
        limits = self.get_dimension_limits(self.data)
        self.data = self.data.isel(**limits)

        self.load_metadata(self.data)

        # Cache the normalization coefficients
        self.normalize_add = None
        self.normalize_factor = None
        self.coefficients = self.read_normalization()


    @staticmethod
    def from_config(config, with_horovod):
        """Returns a Loader object based on a configuration dictionary"""
        kwargs = {k:v for k,v in config.items() if k != "type"}
        kwargs["with_horovod"] = with_horovod
        range_variables = ["limit_leadtimes"]

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

        if self.interleave_leadtime:
            dataset = tf.data.Dataset.range(self.num_files * self.num_leadtimes)
            if randomize_order:
                dataset = dataset.shuffle(len(self.filenames * self.num_leadtimes))
        else:
            dataset = tf.data.Dataset.range(self.num_files)
            if randomize_order:
                dataset = dataset.shuffle(len(self.filenames))

        if shard_size is not None:
            # start = shard_index * math.ceil(self.num_files // shard_size)
            # end = (shard_index + 1) * math.ceil(self.num_files // shard_size)
            # print("SHARD", shard_index, start, end)
            # dataset = tf.data.Dataset.range(start, end)
            dataset = dataset.shard(shard_size, shard_index)

        if repeat is not None:
            dataset = dataset.repeat(repeat)

        # Read data from NETCDF files
        # Outputs three tensors:
        #     predictors: 59, 2321, 1796, 8
        #     static_predictor: 2321, 1796, 6
        #     targets: 59, 2321, 1796, 1
        # Set number of parallel calls to 1, so that the pipeline doesn't get too far ahead on the
        # reading, causing the memory requirement to be large. The reading is not the bottleneck so
        # we don't need to read multiple files in parallel.
        dataset = dataset.map(self.read, num_parallel_calls=1)
        dataset = dataset.prefetch(1)

        # Broadcast static_predictors to leadtime dimension
        # Set parallel_calls to 1 here as well, to prevent the pipeline from getting too far ahead
        dataset = dataset.map(self.expand_static_predictors, num_parallel_calls=1)

        # Unbatch the leadtime dimension, so that each leadtime can be processed in parallel
        dataset = dataset.unbatch()
        dataset = dataset.batch(1)

        # Processing steps
        if 1:
            # Merge static_predictors into predictors and add a few more predictors
            dataset = dataset.map(self.feature_extraction, num_parallel_calls)
            # Predictor shape: 1, 2321, 1796, 16

            # Sutract the raw forecast from the targets
            dataset = dataset.map(self.diff, num_parallel_calls)
            # Predictor shape: 1, 2321, 1796, 16

            # Normalize the predictors
            dataset = dataset.map(self.normalize, num_parallel_calls)
            # Predictor shape: 1, 2321, 1796, 16

            # Split the y,x dimensions into patches of size 256x256
            dataset = dataset.map(self.patch, num_parallel_calls)
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
            if randomize_order:
                dataset = dataset.shuffle(self.num_patches_per_file)

            # Predictor shape: 59, 256, 256, 14
        else:
            # Unbatch the leadtime dimension
            dataset = dataset.unbatch()
            # Predictor shape: 63, 256, 256, 14

            # Unbatch the patch dimension
            dataset = dataset.unbatch()
            # Predictor shape: 256, 256, 14

            if randomize_order:
                dataset = dataset.shuffle(self.num_patches_per_file)

            # Batch so that the dataset has 4 dimensions
            dataset = dataset.batch(1)
            # Predictor shape: 1, 256, 256, 14

        dataset = dataset.batch(self.batch_size)

        if self.filename_validation_cache is not None:
            dataset = dataset.cache(self.filename_validation_cache)

        if self.cache:
            dataset = dataset.cache()

        # Copy data to the GPU
        dataset = dataset.map(self.to_gpu, num_parallel_calls)
        dataset = dataset.prefetch(1)

        self.start_time = time.time()
        return dataset


    """
    Functions for getting dataset metadata
    """
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
    def batch_predictor_shape(self):
        shape = [self.batch_size] + [i for i in self.predictor_shape]
        if not self.with_leadtime:
            shape[1] = 1
        if self.patch_size is not None:
            shape[2] = self.patch_size
            shape[3] = self.patch_size
        return shape

    @property
    def sample_predictor_shape(self):
        """Size of one sample predictor"""
        return self.batch_predictor_shape[1:]

    @property
    def batch_target_shape(self):
        shape = [self.batch_size] + [i for i in self.target_shape]
        if not self.with_leadtime:
            shape[1] = 1
        if self.patch_size is not None:
            shape[2] = self.patch_size
            shape[3] = self.patch_size
        return shape

    @property
    def sample_target_shape(self):
        """Size of one sample target"""
        return self.batch_target_shape[1:]

    @property
    def num_files(self):
        """Returns the total number of files in the dataset"""
        return len(self.filenames)

    @property
    def num_patches_per_file(self):
        """Returns the number of patches in each NetCDF file"""
        return self.num_x_patches * self.num_y_patches

    @property
    def num_batches(self):
        return int(np.ceil(self.num_samples_per_file * self.num_files / self.batch_size))

    @property
    def num_batches_per_file(self):
        return int(np.ceil(self.num_samples_per_file / self.batch_size))

    @property
    def num_samples(self):
        return self.num_samples_per_file * self.num_files

    @property
    def num_samples_per_file(self):
        """Returns the number of samples in each NetCDF file"""
        if self.with_leadtime:
            return self.num_patches_per_file
        else:
            return self.num_patches_per_file * self.num_leadtimes

    @property
    def num_patches(self):
        """Returns the total number of patches in the dataset"""
        return self.num_patches_per_file * self.num_files

    @property
    def num_leadtimes(self):
        """Returns the number of leadtimes in the dataset"""
        return len(self.leadtimes)

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

    @property
    def raw_predictor_index(self):
        """Returns the predictor index corresponding to the raw forecast"""
        raw_predictor_index = self.predictor_names.index("air_temperature_2m")
        return raw_predictor_index

    """
    Functions used by the data processing pipeline
    """
    @maelstrom.map_decorator1_to_4
    def read(self, index):
        """Read data from NetCDF

        Args:
            index (int): Read data from this index

        Returns:
            predictors (tf.tensor): Predictors tensor (leadtime, y, x, predictor)
            static_predictors (tf.tensor): Satic predictor tensor (y, x, static_predictor)
            targets (tf.tensor): Targets tensor (leadtime, y, x, 1)
            leadtimes (tf.tensor): Leadtime tensor
        """
        s_time = time.time()
        index = index.numpy()
        self.print(f"Start reading index={index}")

        if self.interleave_leadtime:
            tindex = index // self.num_leadtimes
            lindex = [index % self.num_leadtimes]
        else:
            tindex = index
            lindex = Ellipsis

        with tf.device(self.device):
            if not self.create_fake_data:
                predictors = self.data["predictors"][tindex, lindex, ...]
                static_predictors = self.data["static_predictors"][tindex, ...]

                if self.probabilistic_target:
                    mean = np.expand_dims(self.data["target_mean"][tindex, lindex, ...], -1)
                    std = np.expand_dims(self.data["target_std"][tindex, lindex, ...], -1)
                    targets = np.concatenate((mean, std), -1)
                else:
                    targets = self.data["target_mean"][tindex, lindex, ...]
                    targets = np.expand_dims(targets, -1)

                # Force explicit conversion here, so that we can account the time it takes
                # Otherwise the conversion happens when the function returns
                predictors = tf.convert_to_tensor(predictors)
                static_predictors = tf.convert_to_tensor(static_predictors)
                targets = tf.convert_to_tensor(targets)
            else:
                predictors = tf.random.uniform(self.predictor_shape)
                targets = tf.expand_dims(tf.random.uniform(self.target_shape), 3)
                static_predictors = tf.random.uniform(self.static_predictor_shape)
        e_time = time.time()
        self.timing["read"] += e_time - s_time

        leadtimes = self.leadtimes[lindex]
        # print(predictors.shape, static_predictors.shape, targets.shape, leadtimes.shape)
        return predictors, static_predictors, targets, leadtimes

    @maelstrom.map_decorator4_to_4
    def expand_static_predictors(self, predictors, static_predictors, targets, leadtimes):
        """Copies static predictors to leadtime dimension. Also subsets spatially."""
        s_time = time.time()
        self.print("Start processing")
        with tf.device(self.device):
            if self.x_range is not None and self.y_range is not None:
                predictors = predictors[:, self.y_range[0]:self.y_range[-1], self.x_range[0]:self.x_range[-1], ...]
                static_predictors = static_predictors[self.y_range[0]:self.y_range[-1], self.x_range[0]:self.x_range[-1], ...]
                targets = targets[:, self.y_range[0]:self.y_range[-1], self.x_range[0]:self.x_range[-1], ...]
            shape = [predictors.shape[0], 1, 1, 1]
            static_predictors = tf.expand_dims(static_predictors, 0)
            static_predictors = tf.tile(static_predictors, shape)

        self.timing["expand"] += time.time() - s_time
        return predictors, static_predictors, targets, leadtimes

    @maelstrom.map_decorator4_to_2
    def feature_extraction(self, predictors, static_predictors, targets, leadtimes):
        """Extract features and append to predictors and merge in static predictors

        Input: leadtime, y, x, predictor
        Output: leadtime, y, x, predictor
        """
        s_time = time.time()
        features = [predictors, static_predictors]
        with tf.device(self.device):
            shape = list(predictors.shape[:-1]) + [1]
            for f, feature in enumerate(self.extra_features):
                feature_type = feature["type"]
                if feature_type == "x":
                    x = tf.range(shape[2], dtype=tf.float32)
                    curr = self.broadcast(x, shape, 2)
                elif feature_type == "y":
                    x = tf.range(shape[1], dtype=tf.float32)
                    curr = self.broadcast(x, shape, 1)
                elif feature_type == "leadtime":
                    x = leadtimes
                    curr = self.broadcast(x, shape, 0)
                curr = tf.expand_dims(curr, -1)
                features += [curr]

            predictors = tf.concat(features, axis=3)
        self.logger.add("extract", time.time() - s_time)
        return predictors, targets

    @maelstrom.map_decorator2_to_2
    def normalize(self, predictors, targets):
        """Normalize predictors

        Input: leadtime, patch, y, x, predictor
        Output: leadtime, patch, y, x, predictor
        """
        s_time = time.time()
        if self.coefficients is None:
            self.logger.add("normalize", time.time() - s_time)
            return predictors, targets

        self.print("Normalize", predictors.shape, self.coefficients.shape)

        with tf.device(self.device):
            # Check for the existance of both vectors, since when this runs in parallel, the
            # first vector may be available before the other
            if self.normalize_add is None or self.normalize_factor is None:
                a = self.coefficients[:, 0]
                s = self.coefficients[:, 1]
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

        self.logger.add("normalize", time.time() - s_time)
        return predictors, targets

    def read_normalization0(self):
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

    @maelstrom.map_decorator2_to_2
    def patch(self, predictors, targets):
        """Decompose grid into patches

        Input: leadtime, y, x, predictor
        Output: leadtime, patch, y_patch, x_patch, predictor
        """
        s_time = time.time()
        self.print("Start patch", time.time() - self.start_time, predictors.shape)

        if self.patch_size is None:
            # A patch dimension is still needed when patching is not done
            with tf.device("CPU:0"):
                p, t = tf.expand_dims(predictors, 1),  tf.expand_dims(targets, 1)
            self.print(p.device)
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

        self.logger.add("patch", time.time() - s_time)
        self.print("Done patching", time.time() - self.start_time, p.shape)
        return p, t

    @maelstrom.map_decorator2_to_2
    def diff(self, predictors, targets):
        """Subtract the raw forecast from predictors and targets

        Input: leadtime, patch, y, x, predictor
        Output: leadtime, patch, y, x, predictor
        """
        s_time = time.time()
        if not self.predict_diff:
            return predictors, targets

        """Subtract the raw_forecast from the target"""
        with tf.device(self.device):
            raw_predictor = tf.expand_dims(predictors[..., self.raw_predictor_index], -1)
            targets = tf.math.subtract(targets, raw_predictor)

        self.logger.add("diff", time.time() - s_time)
        return predictors, targets

    @maelstrom.map_decorator3_to_2
    def process(self, predictors, static_predictors, targets):
        """Perform all processing steps in one go"""
        with tf.device(self.device):
            p, t = self.feature_extraction(predictors, static_predictors, targets)
            p, t = self.normalize(p, t)
            p, t = self.patch(p, t)
            p, t = self.diff(p, t)
        return p, t

    @maelstrom.map_decorator2_to_2
    def reorder(self, predictors, targets):
        """Move patch dimension to be the first dimension

        Input: leadtime, patch, y, x, predictor
        Output: patch, leadtime, y, x, predictor
        """
        s_time = time.time()
        with tf.device(self.device):
            p = tf.transpose(predictors, [1, 0, 2, 3, 4])
            t = tf.transpose(targets, [1, 0, 2, 3, 4])
        self.timing["reorder"] += time.time() - s_time
        return p, t

    @maelstrom.map_decorator2_to_2
    def to_gpu(self, predictors, targets):
        s_time = time.time()
        p = tf.convert_to_tensor(predictors)
        t = tf.convert_to_tensor(targets)
        self.timing["to_gpu"] += time.time() - s_time
        return p, t

    def print(self, *args):
        if self.debug:
            curr_time = time.time() - self.start_time
            message = ' '.join(["%s" % s for s in args])
            print(f"{curr_time:.2f}: {message}")

    @maelstrom.map_decorator2_to_2
    def print_shape(self, p, t):
        """Helper function to print out the shape of tensors"""
        print(p.shape, t.shape)
        return p, t

    @maelstrom.map_decorator2_to_2
    def print_start_processing(self, p, t):
        print("%.4f" % (time.time() - self.start_time), "Start processing", self.count_start_processing)
        self.count_start_processing += 1
        return p, t

    @maelstrom.map_decorator2_to_2
    def print_done_processing(self, p, t):
        print("%.4f" % (time.time() - self.start_time), "Done processing", self.count_done_processing)
        self.count_done_processing += 1
        return p, t

    @maelstrom.map_decorator1_to_1
    def print_start_time(self, p):
        print("Start", p.numpy(), "%.4f" % (time.time() - self.start_time))
        return p

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

        # Subsetting x and y makes data loading too slow. It is better to subset after the data has
        # been loaded.
        # if self.x_range is not None:
        #     limit["x"] = self.x_range
        # if self.y_range is not None:
        #     limit["y"] = self.y_range
        return limit

    def load_metadata(self, dataset):
        """Reads matadata from one file and stores relevant information in self"""
        self.leadtimes = dataset.leadtime.to_numpy()
        self.times = dataset.time.to_numpy()
        if not maelstrom.util.is_list(self.times):
            self.times = np.array([self.times])
        # if self.limit_leadtimes is not None:
        #     self.leadtimes = [dataset.leadtime[i] for i in self.limit_leadtimes]
        if self.x_range is not None and self.y_range is not None:
            self.num_x_input = self.x_range[1] - self.x_range[0]
            self.num_y_input = self.y_range[1] - self.y_range[0]
        else:
            self.num_x_input = len(dataset.x)
            self.num_y_input = len(dataset.y)
        if self.patch_size is not None:
            if self.patch_size > self.num_x_input:
                raise ValueError("Patch size too small")
            if self.patch_size > self.num_y_input:
                raise ValueError("Patch size too small")
        self.predictor_names_input = [p for p in dataset.predictor.to_numpy()] + [p for p in dataset.static_predictor.to_numpy()]
        self.predictor_names = self.predictor_names_input + [self.get_feature_name(p) for p in self.extra_features]

        if self.limit_predictors is not None:
            self.num_predictors = len(self.limit_predictors) + len(self.extra_features)
        else:
            self.num_predictors = len(dataset.predictor) + len(dataset.static_predictor) + len(self.extra_features)

        self.num_targets = 1 + self.probabilistic_target

        # dataset.close()

    def description(self):
        d = dict()
        def vec_to_str(vec):
            return ", ".join(["%d" % i for i in vec])
        d["Predictor size (batch)"] = vec_to_str(self.batch_predictor_shape)
        d["Predictor size (sample)"] = vec_to_str(self.sample_predictor_shape)
        d["Target shape (sample)"] = vec_to_str(self.sample_target_shape)
        d["Total num targets"] = f"{int(np.product(self.sample_target_shape) * self.num_samples):,}"
        d["Num files"] = self.num_files
        d["Num leadtimes"] = self.num_leadtimes
        d["Num predictors"] = self.num_predictors
        d["Num targets fields"] = self.num_targets
        d["Batch size"] = self.batch_size
        d["With leadtime dim"] = self.with_leadtime

        if self.patch_size is not None:
            d["Patch size"] = self.patch_size
            d["Patches per file"] = self.num_patches_per_file

        d["Total num batches"] = self.num_samples_per_file * self.num_files // self.batch_size
        d["Total num samples"] = self.num_samples_per_file * self.num_files
        d["Sample size (MB)"] = self.size_gb * 1024 / self.num_samples
        d["Total size (GB)"] = self.size_gb

        d["Predictors"] = list()
        if self.predictor_names is not None:
            for q in self.predictor_names:
                d["Predictors"] += [str(q)]
        return d

    def __str__(self):
        """Returns a string representation of the dataset"""
        return json.dumps(self.description(), indent=4)

    def read_normalization(self):
        ret = None
        if self.filename_normalization is not None:
            ret = np.zeros([self.num_predictors, 2], np.float32)
            ret[:, 1] = 1
            with open(self.filename_normalization) as file:
                coefficients = yaml.load(file, Loader=yaml.SafeLoader)
                # Add normalization information for the extra features
                for k,v in self.get_extra_features_normalization(self.extra_features).items():
                    coefficients[k] = v

                for i, name in enumerate(self.predictor_names):
                    if name in coefficients:
                        ret[i, :] = coefficients[name]
                    elif name in self.extra_features:
                        ret[i, :] = [0, 1]
                    else:
                        ret[i, :] = [0, 1]
        return ret

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
                val = self.leadtimes
                curr = [np.mean(val), np.std(val)]

            normalization[feature_name] = curr
        return normalization

    def get_feature_name(self, feature):
        if "name" in feature:
            return feature["name"]
        else:
            return feature["type"]

    @property
    def size_gb(self):
        """Returns the number of bytes needed to store the full dataset"""
        size_bytes = np.product(self.predictor_shape) * 4
        size_bytes += np.product(self.target_shape) * 4
        return size_bytes * len(self.filenames) * self.num_patches_per_file / 1024 ** 3

    def check_compatibility(self, other_loader):
        status = self.predictor_names == other_loader.predictor_names
        status &= self.predict_diff == other_loader.predict_diff
        status &= self.probabilistic_target == other_loader.probabilistic_target
        return status

    def get_time_from_batch(self, batch, sample):
        file_index = batch // self.num_batches_per_file
        return self.times[file_index]

    def get_leadtime_from_batch(self, batch, sample, ileadtime):
        if self.with_leadtime:
            leadtime_index = ileadtime
        else:
            assert ileadtime == 0
            sample_index = batch * self.batch_size + sample
            leadtime_index = sample_index % self.num_leadtimes
        # print(batch, sample, sample_index, leadtime_index)
        return self.leadtimes[leadtime_index]

    def get_frequency(self, freq, with_horovod=False):
        """ Computes the number of batches corresponding to a frequency specification

        Args:
            freq (str|int): Either a number of batches, or a string of the form "number units"
                where units is one of "epoch", "file", "batch"
            with_horovod (bool): True if we are running with horovod

        Returns:
            freq (int): Frequency in number of batches
        """
        if freq is None:
            raise ValueError("Frequency cannot be None")

        words = freq.split(" ")
        if len(words) == 1:
            freq = int(words[0])
            freq_units = "batch"
        else:
            if len(words) != 2:
                raise ValueError(
                    "must be in the form <value> <unit>"
                )
            freq, freq_units = words
            freq = int(freq)
        if freq_units == "epoch":
            output_frequency = self.num_batches * freq
        elif freq_units == "file":
            output_frequency = self.num_batches_per_file * freq
            # The convept of "file" needs to be handled differently when horovod processes several
            # files in parallel. The most intuitive would be that if we want validation every 36
            # files, and 4 are run in parallel, we should validated every 9 files relative to one
            # process's data loader.
            if with_horovod:
                output_frequency //= hvd.size()
        elif freq_units == "batch":
            output_frequency = freq
        else:
            raise ValueError(
                f"Unknown frequency units '{freq_units}'"
            )
        return output_frequency
