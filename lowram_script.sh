#!/bin/bash

#SBATCH --job-name=muvi_test
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1                # Request 1 GPU, modify if more GPUs are needed
#SBATCH -c 4 
#SBATCH -p low
#SBATCH --mem=20g 
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=jingyiz4@andrew.cmu.edu 
#SBATCH -e /work/jingyiz4/logs/low_out.err 
#SBATCH -o /work/jingyiz4/logs/low_out.log 

# Run your program
python main.py
