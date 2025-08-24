#!/bin/bash
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --job-name=step5_test_llm
#SBATCH --output=logs/step5_test_llm_%j.out
#SBATCH --error=logs/step5_test_llm_%j.err
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00


echo "Starting Step 5 Testing at: $(date)" 

module load cuda/12.6
source /projects/illinois/eng/cs/jimeng/zelalae2/scratch/PyHealth/venv_pyhealth/bin/activate
cd /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA


python zilal_contribution/notebook/test_llm_hpo.py \
    --cache_dir /projects/illinois/eng/cs/jimeng/zelalae2/scratch/rdma_cache \

echo "Finished Step1 Extracting at: $(date)"
