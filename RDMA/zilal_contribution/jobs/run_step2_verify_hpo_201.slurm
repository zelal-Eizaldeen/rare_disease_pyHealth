#!/bin/bash
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --job-name=step2_ver_hpo_102
#SBATCH --output=logs/step2_ver_hpo_102_%j.out
#SBATCH --error=logs/step2_ver_hpo_102_%j.err
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00

echo "Starting Step 2 verification HPO 1000 at: $(date)" 

module load cuda/12.6
source /projects/illinois/eng/cs/jimeng/zelalae2/scratch/PyHealth/venv_pyhealth/bin/activate
cd /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/scripts

python 3.hpo_steps/4.step2_verify_entity_context.py \
  --input_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/output_checkpoints/hpo/pass1/step1_extract_entity_context_hpo_102.json \
  --output_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/output_checkpoints/hpo/pass1/step2_verify_hpo_102.json \
  --embeddings_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/data/vector_stores/G2GHPO_metadata_medembed.npy \
  --lab_embeddings_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/data/vector_stores/lab_tables_medembed_sm.npy \
  --min_context_length 5 \
  --verifier_version v4 \
  --retriever sentence_transformer \
  --retriever_model abhinand/MedEmbed-small-v0.1 \
  --llm_type local \
  --model_type sentence_transformer \
  --temperature 0.001 \
  --cache_dir /projects/illinois/eng/cs/jimeng/zelalae2/scratch/rdma_cache \
  --checkpoint_interval 100 \
  --resume \
  --debug \
  --condor

  echo "Finished Step2 verification at HPO: $(date)"
