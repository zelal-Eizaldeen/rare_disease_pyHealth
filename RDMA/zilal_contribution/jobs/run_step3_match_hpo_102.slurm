#!/bin/bash
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --job-name=step3_match_hpo
#SBATCH --output=logs/step3_match_hpo_201_%j.out
#SBATCH --error=logs/step3_match_hpo_201_%j.err
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

echo "Starting Step 3 Matching HPO 201 at: $(date)" 

module load cuda/12.6
source /projects/illinois/eng/cs/jimeng/zelalae2/scratch/PyHealth/venv_pyhealth/bin/activate
cd /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/scripts

python 3.hpo_steps/5.step3_match_hpo.py \
  --input_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/output_checkpoints/hpo/pass1/step2_verify_hpo_102.json \
  --output_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/output_checkpoints/hpo/pass1/step3_match_hpo_102_context_output.json \
  --embeddings_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/data/vector_stores/G2GHPO_metadata_medembed.npy \
  --retriever sentence_transformer \
  --llm_type local \
  --model_type sentence_transformer \
  --retriever_model abhinand/MedEmbed-small-v0.1 \
  --temperature 0.001 \
  --cache_dir /projects/illinois/eng/cs/jimeng/zelalae2/scratch/rdma_cache \
  --checkpoint_interval 100 \
  --resume \
  --system_prompt_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/scripts/3.hpo_steps/data/prompts/system_prompts.json \
  --debug \
  --condor

  echo "Finished Step3 verification HPO 201 at: $(date)"
