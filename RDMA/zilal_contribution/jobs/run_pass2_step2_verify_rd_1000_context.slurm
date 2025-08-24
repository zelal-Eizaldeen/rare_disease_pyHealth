#!/bin/bash
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --job-name=pass2_step2_ver_rd_1000
#SBATCH --output=logs/pass2_step2_ver_rd_1000_%j.out
#SBATCH --error=logs/pass2_step2_ver_rd_1000_%j.err
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

echo "Starting Pass2 Step 2 verification at: $(date)" 

module load cuda/12.6
source /projects/illinois/eng/cs/jimeng/zelalae2/scratch/PyHealth/venv_pyhealth/bin/activate
cd /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA

python rd_steps/step2_verify_rd_context.py \
  --input_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/output_checkpoints/pass2_rd_results/testing_step1_extract_rd_1000_context_output.json \
  --output_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/output_checkpoints/pass2_rd_results/testing_step2_ver_rd_1000_context_output.json \
  --embeddings_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/data/vector_stores/rd_orpha_medembed.npy \
  --model_type mistral_24b \
  --verifier_type multi_stage \
  --abbreviations_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/data/vector_stores/abbreviations_medembed_sm.npy \
  --min_context_length 5 \
  --retriever sentence_transformer \
  --llm_type local \
  --temperature 0.001 \
  --cache_dir /projects/illinois/eng/cs/jimeng/zelalae2/scratch/rdma_cache \
  --checkpoint_interval 10 \
  --debug \
  --condor

  echo "Finished Pass2 Step2 verification at: $(date)"
