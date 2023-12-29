#!/bin/bash
#SBATCH --job-name=maelstrom-vef
#SBATCH --account=deepacf # Do not change the account name
#SBATCH --nodes=2
#SBATCH --ntasks=8
##SBATCH --nodes=1
##SBATCH --ntasks=4
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/jewels-booster-out.%j
#SBATCH --error=logs/jewels-booster-err.%j
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:4
#SBATCH --partition=booster

cd ..

module load Stages/2022 && module load GCCcore/.11.2.0 && module load TensorFlow/2.6.0-CUDA-11.5 && module load GCC/11.2.0 && module load OpenMPI/4.1.2 && module load mpi4py/3.1.3 && module load Horovod/0.24.3; source /p/project/deepacf/maelstrom/nipen1/maelstrom-train/jube/../.venv_jwb/bin/activate;

export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"
export LD_PRELOAD=/p/home/jusers/nipen1/juwels/local/lib/libvefprospector_full.so
export files=/p/scratch/deepacf/maelstrom/maelstrom_data/ap1/air_temperature//5TB/202???01T*.nc 
#export files=/p/scratch/deepacf/maelstrom/maelstrom_data/ap1/air_temperature//5TB/202?????T*.nc 

# srun  python -u /p/project/deepacf/maelstrom/nipen1/maelstrom-train/jube/../benchmark/benchmark.py\
#    $files -m train -b 72 -p 512 -j 12 -e 1 -val /p/scratch/deepacf/maelstrom/maelstrom_data/ap1/air_temperature//5TB/20200615T*.nc --norm /p/scratch/deepacf/maelstrom/maelstrom_data/ap1/air_temperature/normalization.yml

srun maelstrom-train --config etc/d1-4/opt1_e1.yml etc/d1-4/loader_debug.yml etc/d1-4/common.yml -m unet_f16_l6_c1_p2 -o results/vef --test=0
