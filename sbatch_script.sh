#!/bin/bash
#SBATCH --time=INFINITE
#SBATCH -p long
#SBATCH --job-name="SC-SFM-L"
#SBATCH -A raghava.modhugu
#SBATCH -n 30
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G 
#SBATCH --nodelist=gnode32
#module load use.own

module load cuda/10.0
module load cudnn/7-cuda-10.0

wait

cd /home/raghava.modhugu/SC-SfMLearner-Release

wait

sh scripts/train_resnet18_depth_256.sh

wait

