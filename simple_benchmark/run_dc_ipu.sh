#!/bin/bash -x
#SBATCH --job-name=maelstrom-train
#SBATCH --account=exalab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/jureca-out.%j
#SBATCH --error=logs/jureca-err.%j
#SBATCH --time=00:10:00
# #SBATCH --gres=gpu:4
#SBATCH --partition=dc-ipu
# #SBATCH --partition=dc-h100
#SBATCH --mail-type=END

srun apptainer run benchmark.sif -- python -u benchmark/simple_ipu_example.py -e 3 -s 100 -b 10 --hardware cpu
srun apptainer run benchmark.sif -- python -u benchmark/simple_ipu_example.py -e 3 -s 100 -b 10 --hardware ipu
