#!/bin/bash
#SBATCH -J EEGJob
#SBATCH -o Job.out
#SBATCH -N 1
#SBATCH -p dlq
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH -t    150:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dumuzhichun@163.com

python HOCV.py
