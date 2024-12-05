#!/bin/sh
#SBATCH --time=12:00:00
#SBATCH -A raiselab
#SBATCH --mem=16000M
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1


source ../dola_env/bin/activate

python tfqa_mc_eval.py --model-name huggyllama/llama-7b --early-exit-layers 16,18,20,22,24,26,28,30,32 --data-path ./ --output-path output-path_truthfulqa.json --num-gpus 1

deactivate