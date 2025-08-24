#!/bin/bash
#SBATCH --partition=eng-research-gpu
#SBATCH --account=jimeng-ic
#SBATCH --job-name=step3_eval1_pho
#SBATCH --output=logs/step3_eval1_hpo_%j.out
#SBATCH --error=logs/step3_eval1_hpo_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G


# Load environment (adjust based on your environment manager)
echo "Starting  Step 3 Evaluating1 for Phenotyping at: $(date)" 

module load cuda/12.6
source /projects/illinois/eng/cs/jimeng/zelalae2/scratch/PyHealth/venv_pyhealth/bin/activate
cd /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA

# Run the evaluation
python hpo_steps/evaluate_step2.py \
  --entities /projects/illinois/eng/cs/jimeng/zelalae2/scratch/output_checkpoints/pass2_hpo_results/pass2_step1_extract_entity_context_hpo.json \
  --ground-truth /projects/illinois/eng/cs/jimeng/zelalae2/scratch/output_checkpoints/step1_extract_entity_context_hpo.json \
  --output /projects/illinois/eng/cs/jimeng/zelalae2/scratch/output_checkpoints/step3_evaluation1_hpo_results.json
