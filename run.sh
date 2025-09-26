#!/bin/bash
#SBATCH --job-name=quant_dpo
#SBATCH --output=./out/quant_dpo_%j.out
#SBATCH --error=./err/quant_dpo_%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josephmcinerney7575@gmail.com

pip install -r requirements.txt
# run it
accelerate launch dpo_train.py

