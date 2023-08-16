import collections
import numpy as np
import os
import pkgutil
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import yaml

tf.random.set_seed(1000)
np.random.seed(1000)

__version__ = "0.1.0"

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


def map_decorator1_to_1(func):
    """Decorator to wrap a 1-argument function as a tf.py_function"""
    def wrapper(self, i):
        return tf.py_function(
                lambda i: func(self, i),
                inp=(i,),
                Tout=(tf.float32, )
                )
    return wrapper

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

def load_yaml(filename):
    """Load yaml from file and return dictionary

    Args:
        filename (str): Filename to read YAML from

    Returns:
        defauldict: Dictionary representing the YAML structure. A key that does exist will return
            None

    Raises:
        IOError: when file does not exist
    """

    if not os.path.exists(filename):
        raise IOError(f"{filename} does not exist")

    with open(filename) as file:
        S = collections.defaultdict(lambda: None)
        temp = yaml.load(file, Loader=yaml.SafeLoader)
        for k, v in temp.items():
            S[k] = v
    return S


"""
def calc_sensitivity(model, loader):
    num_predictors = loader.num_predictors
    for i in range(num_predictors):
"""


def merge_configs(configs):
    config = configs[0]
    for i in range(1, len(configs)):
        config.update(configs[i])
    return config


__all__ = []
# Can't use the name "loader", because then there is a namespace conflict with maelstrom.loader
for the_loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    if module_name == "__main__":
        continue

    __all__.append(module_name)
    _module = the_loader.find_module(module_name).load_module(module_name)
    globals()[module_name] = _module


def main():
    import maelstrom.__main__

    maelstrom.__main__.main()

def predict():
    import maelstrom.predict
    maelstrom.predict.main()
