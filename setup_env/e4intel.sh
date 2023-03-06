#!/bin/bash
thisfile=`readlink -f ${BASH_SOURCE:-$0}`
BASE_DIR=`dirname $thisfile`
echo $thisfile $BASE_DIR
VIRT_ENV=${BASE_DIR}/../.venv_e4intel

module load python/3.9.6
python -m venv $VIRT_ENV
source $VIRT_ENV/bin/activate
pip install -r $BASE_DIR/requirements_e4_intel.txt
pip install numpy==1.21.1

