#!/bin/bash
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --job-name=load_data
#SBATCH --output=logs/load_data_%j.out
#SBATCH --error=logs/load_data_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00



module load cuda/12.6
source /projects/illinois/eng/cs/jimeng/zelalae2/scratch/PyHealth/venv_pyhealth/bin/activate
cd /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/scripts


echo "Loading Data from MIMIC_IV ..."
python load_mimic_iv_notes.py \
   --note_root /projects/illinois/eng/cs/jimeng/zelalae2/scratch/mimic4_note/physionet.org/files/mimic-iv-note/2.2/ \
   --processed_file /projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/processed_ids_folder/processed_ids.json \

echo "Loading MIMIC_IV notes completed."
echo "Finished Loading MIMIC_IV notes at: $(date)"