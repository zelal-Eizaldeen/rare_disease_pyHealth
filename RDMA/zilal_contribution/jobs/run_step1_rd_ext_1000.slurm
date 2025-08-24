#!/bin/bash
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --job-name=step1_rd_ext_1000
#SBATCH --output=logs/step1_rd_ext_1000_%j.out
#SBATCH --error=logs/step1_rd_ext_1000_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00


echo "Starting Step 1 Extracting Rare disease at: $(date)" 

module load cuda/12.6
source /projects/illinois/eng/cs/jimeng/zelalae2/scratch/PyHealth/venv_pyhealth/bin/activate
cd /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/scripts


python rd_steps/step1_extract_rd_context.py \
  --input_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/processed_ids_folder/input_patient_notes_1000.json \
  --output_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/output_checkpoints/rare_disease/pass1/step1_extract_rd_1000_context_output.json \
  --embeddings_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/data/vector_stores/rd_orpha_medembed.npy \
  --entity_extractor llm \
  --llm_type local \
  --model_type mistral_24b \
  --retriever sentence_transformer \
  --retriever_model abhinand/MedEmbed-small-v0.1 \
  --window_size 50 \
  --temperature 0.001 \
  --cache_dir /projects/illinois/eng/cs/jimeng/zelalae2/scratch/rdma_cache \
  --checkpoint_interval 100 \
  --resume \
  --debug \
  --condor

  echo "Finished Step1 Extracting Rare disease at: $(date)"
