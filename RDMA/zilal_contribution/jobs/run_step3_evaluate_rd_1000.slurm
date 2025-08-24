#!/bin/bash
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --job-name=step3_eval_rd_1000
#SBATCH --output=logs/step3_eval_rd_1000_%j.out
#SBATCH --error=logs/step3_eval_rd_1000_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G


# Load environment (adjust based on your environment manager)
echo "Starting  Step 3 Evaluating at: $(date)" 

module load cuda/12.6
source /projects/illinois/eng/cs/jimeng/zelalae2/scratch/PyHealth/venv_pyhealth/bin/activate
cd /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA

# Run the evaluation
python rd_steps/step3_evaluate_rd.py \
  --ground-truth /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/output_checkpoints/rare_disease/pass1/step3_match_rd_1000_context_output.json \
  --predictions /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/output_checkpoints/rare_disease/pass2/step3_match_rd_1000_context_output.json \
  --confidence-threshold 0.0 \
  --output /projects/illinois/eng/cs/jimeng/zelalae2/scratch/output_checkpoints/step3_rd_evaluation_second_1000_results.json
