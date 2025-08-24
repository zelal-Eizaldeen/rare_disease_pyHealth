#!/bin/bash
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --job-name=pass2_step3_direct_mat_rd
#SBATCH --output=logs/pass2_step3_direct_mat_rd_%j.out
#SBATCH --error=logs/pass2_step3_direct_mat_rd_%j.err
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

echo "Starting Pass2 Step 3 Direct Matching at: $(date)" 

echo "Starting Step 3 Direct Matching at: $(date)" 

module load cuda/12.6
source /projects/illinois/eng/cs/jimeng/zelalae2/scratch/PyHealth/venv_pyhealth/bin/activate
cd /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA

python rd_steps/step3_direct_match_rd.py \
  --input_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/output_checkpoints/pass2_rd_results/testing_step2_ver_rd_context_output.json \
  --output_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/output_checkpoints/pass2_rd_results/testing_step3_direct_mat_rd_context_output.json \
  --embeddings_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/data/vector_stores/rd_orpha_medembed.npy \
  --retriever sentence_transformer \
  --llm_type local \
  --model_type mistral_24b \
  --temperature 0.001 \
  --cache_dir /projects/illinois/eng/cs/jimeng/zelalae2/scratch/rdma_cache \
  --checkpoint_int 10 \
  --debug \
  --condor

  echo "Finished Pass2 Step3 Direct Match at: $(date)"
