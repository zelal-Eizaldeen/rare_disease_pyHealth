#!/bin/bash
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --job-name=step1_ext_hpo_201
#SBATCH --output=logs/step1_ext_hpo_201_%j.out
#SBATCH --error=logs/step1_ext_hpo_201_%j.err
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00 


echo "Starting Step 1 Extracting HPO 1000 at: $(date)" 

module load cuda/12.6
source /projects/illinois/eng/cs/jimeng/zelalae2/scratch/PyHealth/venv_pyhealth/bin/activate
cd /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/scripts


python 3.hpo_steps/3.step1_extract_entity_context.py \
  --input_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/processed_ids_folder/hpo_102_ids_data.json \
  --output_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/output_checkpoints/hpo/pass1/step1_extract_entity_context_hpo_102.json \
  --llm_type local \
  --entity_extractor retrieval \
  --model_type mistral_24b \
  --retriever_model  BAAI/bge-small-en-v1.5 \
  --retriever sentence_transformer \
  --embeddings_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/data/vector_stores/G2GHPO_metadata_medembed.npy \
  --window_size 50 \
  --temperature 0.001 \
  --cache_dir /projects/illinois/eng/cs/jimeng/zelalae2/scratch/rdma_cache \
  --checkpoint_interval 100 \
  --debug \
  --resume \
  --condor

  echo "Finished Step1 Extracting HPO 1000 at: $(date)"
