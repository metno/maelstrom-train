VIRT_ENV=.venv_jwb

module --force purge
module use $OTHERSTAGES
module load Stages/2022
module load GCCcore/.11.2.0
module load TensorFlow/2.6.0-CUDA-11.5
module load GCC/11.2.0
module load OpenMPI/4.1.2  # Added for horovod
module load mpi4py/3.1.3  # Added for horovod
module load Horovod/0.24.3

#HOROVOD
# module load NVHPC/22.9
# module load GCC/11.2.0
# module load OpenMPI/4.1.2
# module load Horovod/0.24.3
# module load mpi4py/3.1.3
# module load Stages/2020
# module load GCCcore/.10.3.0
# module load TensorFlow/2.5.0-Python-3.8.5

# export PYTHONPATH=.venv/lib/python3.8/site-packages:$PYTHONPATH
export PYTHONUNBUFFERED=TRUE
source $VIRT_ENV/bin/activate
