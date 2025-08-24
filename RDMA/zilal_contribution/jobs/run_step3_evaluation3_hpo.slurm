#!/bin/bash
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --job-name=step3_eval3_pho
#SBATCH --output=logs/step3_eval3_hpo_%j.out
#SBATCH --error=logs/step3_eval3_hpo_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G


# Load environment (adjust based on your environment manager)
echo "Starting  Step 3 Evaluating3 for Phenotyping at: $(date)" 

module load cuda/12.6
source /projects/illinois/eng/cs/jimeng/zelalae2/scratch/PyHealth/venv_pyhealth/bin/activate
cd /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA

# Run the evaluation
python hpo_steps/evaluate_step3.py \
  --predictions /projects/illinois/eng/cs/jimeng/zelalae2/scratch/output_checkpoints/pass2_hpo_results/pass2_step3_match_hpo_context_output.json \
  --ground-truth /projects/illinois/eng/cs/jimeng/zelalae2/scratch/output_checkpoints/step3_match_hpo_context_output.json \
  --output /projects/illinois/eng/cs/jimeng/zelalae2/scratch/output_checkpoints/step3_evaluation3_hpo_results.json
