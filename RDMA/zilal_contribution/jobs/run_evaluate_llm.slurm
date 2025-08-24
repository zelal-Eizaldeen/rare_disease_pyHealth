#!/bin/bash
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --job-name=eval_llm
#SBATCH --output=logs/eval_llm_%j.out
#SBATCH --error=logs/eval_llm_%j.err
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

echo "Starting Evaluate LLM at: $(date)" 

module load cuda/12.6
source /projects/illinois/eng/cs/jimeng/zelalae2/scratch/PyHealth/venv_pyhealth/bin/activate
cd /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/notebook/john_code


python rd_diff_diagnosis_benchmark_eval.py

echo "Finished Evaluate LLM at: $(date)"
