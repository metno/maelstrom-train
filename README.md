# MAELSTROM train

This repository contains a command-line program to train models one the MAELSTROM A1 dataset


## Installation
```
pip install .
```

This installs the `maelstrom-train` command-line tool. You also need the MAELSTROM dataset to be available in the folder `./data`. On the PPI, you can simply symlink it as
follows:
```
ln -s /lustre/storeB/project/nwp/maelstrom data
```

## Quick example
Check out the jupyter notebook, available in [`./example.ipynb`](example.ipynb)

Alternatively, run the commandline program like this:
```
maelstrom-train --config etc/example.yml -m regression
```
The results are then output in `results/regression_*/`. `regression_validation.txt` Stores the validation score for each time
the validation is run. `regression_agg.txt` stores the testing results for each output date and output leadtime.
