#!/bin/bash
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --job-name=step3_eval2_pho
#SBATCH --output=logs/step3_eval2_hpo_%j.out
#SBATCH --error=logs/step3_eval2_hpo_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G


# Load environment (adjust based on your environment manager)
echo "Starting  Step 3 Evaluating2 for Phenotyping at: $(date)" 

module load cuda/12.6
source /projects/illinois/eng/cs/jimeng/zelalae2/scratch/PyHealth/venv_pyhealth/bin/activate
cd /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA

# Run the evaluation
python hpo_steps/evaluate_step2.py \
  --predictions /projects/illinois/eng/cs/jimeng/zelalae2/scratch/output_checkpoints/pass2_hpo_results/pass2_step2_ver_hpo_context_output_v4.json \
  --ground-truth /projects/illinois/eng/cs/jimeng/zelalae2/scratch/output_checkpoints/step2_ver_hpo_context_output_v4.json \
  --output /projects/illinois/eng/cs/jimeng/zelalae2/scratch/output_checkpoints/step3_evaluation2_hpo_results.json
