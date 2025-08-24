#!/bin/bash
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --job-name=step3_match_rd_1000
#SBATCH --output=logs/step3_match_rd_1000_%j.out
#SBATCH --error=logs/step3_match_rd_1000_%j.err
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

echo "Starting Step 3 Matching RD at: $(date)" 

module load cuda/12.6
source /projects/illinois/eng/cs/jimeng/zelalae2/scratch/PyHealth/venv_pyhealth/bin/activate
cd /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/scripts


python rd_steps/step3_match_rd.py \
  --input_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/output_checkpoints/rare_disease/pass2/step2_ver_rd_1000_context_output.json \
  --output_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/output_checkpoints/rare_disease/pass2/step3_match_rd_1000_context_output.json \
  --embeddings_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/data/vector_stores/rd_orpha_medembed.npy \
  --retriever sentence_transformer \
  --llm_type local \
  --model_type mistral_24b \
  --temperature 0.7 \
  --cache_dir /projects/illinois/eng/cs/jimeng/zelalae2/scratch/rdma_cache \
  --checkpoint_interval 100 \
  --resume \
  --debug \
  --condor

  echo "Finished Step3 Matching RD at: $(date)"
