#!/bin/bash -x
#SBATCH --job-name=maelstrom-train
#SBATCH --account=exalab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/jureca-h100-out.%j
#SBATCH --error=logs/jureca-h100-err.%j
#SBATCH --time=00:10:00
# #SBATCH --gres=gpu:4
#SBATCH --partition=dc-h100
#SBATCH --mail-type=END

cd setup_env
bash setup_jh100.sh
cd ..

module load Architecture/jureca_spr
module load CUDA/12.0
module load Python/3.10.4

source env_jh100.sh
python -u simple_ipu_example.py -e 3 -s 10 -b 30 --hardware gpu
