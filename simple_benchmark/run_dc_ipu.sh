#!/bin/bash -x
#SBATCH --job-name=maelstrom-train
#SBATCH --account=exalab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/jureca-out.%j
#SBATCH --error=logs/jureca-err.%j
#SBATCH --time=00:10:00
#SBATCH --partition=dc-ipu
#SBATCH --mail-type=END

# srun apptainer run benchmark.sif -- python -u simple_benchmark.py -e 3 -s 100 -b 10 --hardware cpu
srun apptainer run benchmark.sif -- python -u simple_benchmark.py -e 3 -s 100 -b 10 --hardware ipu
