import json

# Load your processed JSON file

input_file= "/projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/output_checkpoints/rare_disease/pass1/step3_match_rd_1000_context_output.json"
output_file = "/projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/processed_ids_folder/patients_rd_ids.json"  # where to save results



with open(input_file, "r") as f:
    data = json.load(f)

rare_disease_patients = []

for patient_id, patient_info in data.get("results", {}).items():
    matched_diseases = patient_info.get("matched_diseases", [])
    
    # Check if any disease has a non-empty 'orpha_id'
    if any(disease.get("orpha_id") for disease in matched_diseases):
        rare_disease_patients.append(patient_id)

# Save the IDs to a JSON file
with open(output_file, "w") as f:
    json.dump(rare_disease_patients, f, indent=2)

print(f"Saved {len(rare_disease_patients)} rare disease patient IDs to {output_file}")
print(f"Number of rare disease patient IDs: {len(rare_disease_patients)}")
