# MAELSTROM train

This repository contains a command-line program to train models one the MAELSTROM A1 dataset


## Installation
```
pip install .
```

This installs the `maelstrom-train` command-line tool.

## Example

This example assumes that the MAELSTROM dataset is available in the folder `./data`. On the PPI, you can simply symlink it as
follows:
```
ln -s /lustre/storeB/project/nwp/maelstrom data
```

To run the example, do this:
```
```maelstrom-train --config etc/example.yml
```

