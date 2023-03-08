#!/bin/bash -x
thisfile=`readlink -f ${BASH_SOURCE:-$0}`
BASE_DIR=`dirname $thisfile`
VIRT_ENV=${BASE_DIR}/../.venv_jwc
echo $VIRT_ENV
if [ -e $VIRT_ENV ]; then
    echo "Environment already set up"
    exit
fi

#load modules
module --force purge
module use $OTHERSTAGES
# module load Stages/2020
# module load GCCcore/.10.3.0
# module load TensorFlow/2.5.0-Python-3.8.5
module load Stages/2022
module load GCCcore/.11.2.0
module load TensorFlow/2.6.0-CUDA-11.5

module load GCC/11.2.0
module load OpenMPI/4.1.2
# module load TensorFlow/2.5.0-Python-3.8.5
module load mpi4py/3.1.3  # Added for horovod
module load Horovod/0.24.3

if [ -d $VIRT_ENV ]; then
    rm -rf $VIRT_ENV
fi
python3 -m venv $VIRT_ENV
source $VIRT_ENV/bin/activate
# export PYTHONPATH=.venv/lib/python3.8/site-packages:$PYTHONPATH
# Complex installation proceedure, because we want tensorflow from the module system, but
# climetlab/xarray/pandas from virtual environment, and they have conflicting requirements
# pip install xarray pandas --ignore-installed
# pip install -r requirements_julich.txt
# python setup.py develop
cd $BASE_DIR/../
pip3 install -e .
# pip3 install -r requirements.txt
