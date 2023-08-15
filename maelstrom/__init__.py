import collections
import os
import pkgutil

import numpy as np
import yaml

__version__ = "0.1.0"


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

def main():
    import maelstrom.__main__

    maelstrom.__main__.main()
