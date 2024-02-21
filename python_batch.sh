#!/bin/bash
#SBATCH --time=00:59:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=inv_kinematics
#SBATCH --mem=8000

module load Python/3.9.6-GCCcore-11.2.0

source ~/env/bin/activate

python train.py --name model_2D
python train.py --dimensions 3 --name model_3D


deactivate