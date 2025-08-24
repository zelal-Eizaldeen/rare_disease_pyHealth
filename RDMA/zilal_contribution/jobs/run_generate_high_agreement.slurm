#!/bin/bash
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --job-name=high_agreement
#SBATCH --output=logs/high_agreement_%j.out
#SBATCH --error=logs/high_agreement_%j.err
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

echo "Starting Generating High Agreement Results at: $(date)" 

module load cuda/12.6
source /projects/illinois/eng/cs/jimeng/zelalae2/scratch/PyHealth/venv_pyhealth/bin/activate
cd /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution


python notebook/john_code/rd_diff_diagnosis_benchmark_construction.py 

echo "Finished  Generating High Agreement Results at: $(date)"
