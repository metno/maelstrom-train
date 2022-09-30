#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import io
import setuptools


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return io.open(file_path, encoding="utf-8").read()


package_name = "maelstrom"

version = None
init_py = os.path.join(package_name.replace("-", "_"), "__init__.py")
for line in read(init_py).split("\n"):
    if line.startswith("__version__"):
        version = line.split("=")[-1].strip()[1:-1]
assert version

setuptools.setup(
    name=package_name,
    version=version,
    description="Maelstrom package for A1",
    url="",
    author="Thomas Nipen",
    author_email="thomasn@met.no",
    packages=setuptools.find_packages(exclude=["test"]),
    install_requires=[
        "argparse",
        "datetime",
        "future",
        "gridpp==0.6.0.dev16",
        "matplotlib",
        "jsonschema",
        "netCDF4<1.6.0",
        "nose",
        "numpy",
        "psutil",
        "pyyaml",
        "requests",
        "scipy",
        "six",
        "tensorflow",
        "tqdm",
        "wandb",
        "xarray",
    ],
    extras_require={
        "test": ["coverage", "pep8"],
    },
    package_data={"etc": ["*.yml"]},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "maelstrom-train=maelstrom.__main__:main",
        ],
    },
)
