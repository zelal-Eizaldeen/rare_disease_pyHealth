import json

# Paths
original_data_file = "/projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/processed_ids_folder/input_patient_notes_1000.json"
patients_ids_file = "/projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/processed_ids_folder/patients_rd_ids.json"
filtered_output_file = "/projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/processed_ids_folder/hpo_ids_data.json"

# Load original full data (dict with patient IDs as keys)
with open(original_data_file, "r") as f:
    original_data = json.load(f)

# Load the list of patient IDs to keep
with open(patients_ids_file, "r") as f:
    patient_ids_to_keep = set(json.load(f))  # convert to set for faster lookup

# Filter original data by keys in patient_ids_to_keep
filtered_data = {pid: original_data[pid] for pid in patient_ids_to_keep if pid in original_data}

# Save filtered data to new JSON file
with open(filtered_output_file, "w") as f:
    json.dump(filtered_data, f, indent=2)

print(f"Original data contains {len(original_data)} patients")
print(f"Filtered data contains {len(filtered_data)} patients saved to {filtered_output_file}") 
