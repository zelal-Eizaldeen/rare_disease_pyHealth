import json
import pandas as pd
import sys
import os





# # Replace this with the path to your JSON file
# json_file = "/projects/illinois/eng/cs/jimeng/zelalae2/scratch/med_students/finished_Jeffrey_rare_disease_reviews_2.json"

# with open(json_file, "r") as f:
#      gt_data = json.load(f)

# print(len(gt_data))  # should show 'results', 'metadata', etc.



# Load the JSON file
# with open("/u/zelalae2/scratch/output_checkpoints/step3_direct_mat_rd_context_output.json", "r") as f:
#     data = json.load(f)


# # Count how many patient IDs are in 'results'
# num_patients = len(data["results"])
# print(f"Number of patient IDs: {num_patients}")


# notes = pd.read_pickl e("/u/zelalae2/scratch/mimic4_note/sampling/patient_notes.pkl")
# print(f"Total patient notes: {len(notes)}")
# print(notes.head())




# with open("/projects/illinois/eng/cs/jimeng/zelalae2/scratch/output_checkpoints/phenotype_results/pass2_phenotype_results.json", "r") as f:
#       data = json.load(f)
# num_patients = len(data["results"])
# print(f"Number of patient IDs: {num_patients}")




# with open("/projects/illinois/eng/cs/jimeng/zelalae2/scratch/med_students/finished_Jeffrey_rare_disease_reviews_2.json", "r") as f2:
#     pho_ids = set(json.load(f2))

# # Compare
# print("Are the ID sets identical?", rd_ids == pho_ids)
# print(rd_ids)
# # Show difference if not identical
# if rd_ids != pho_ids:
#     print("In rd but not in pho:", rd_ids - pho_ids)
#     print("In pho but not in rd:", pho_ids - rd_ids)

# Check if they are identical
# Load both files
# Load the two JSON files
with open("/projects/illinois/eng/cs/jimeng/zelalae2/scratch/med_students/finished_Jeffrey_rare_disease_reviews_2.json", "r") as file1, \
     open("/projects/illinois/eng/cs/jimeng/zelalae2/scratch/med_students/Jennifer_finished_rare_disease_reviews.json", "r") as file2, \
     open("/projects/illinois/eng/cs/jimeng/zelalae2/scratch/med_students/lauren_rare_disease_reviews.json", "r") as file3, \
     open("/projects/illinois/eng/cs/jimeng/zelalae2/scratch/med_students/Mya_rare_disease_reviews.json", "r") as file4, \
     open("/projects/illinois/eng/cs/jimeng/zelalae2/scratch/output_checkpoints/testing_step4_supervisor_rd_1000_results.json", "r") as file5:
          

    jennifer_data = json.load(file1)
    jennifer_data2 = json.load(file2)
    lauren_data = json.load(file3)
    Mya_data = json.load(file4)
    zilal_data=json.load(file5)
# Check for exact match
# if jennifer_data == jennifer_data2:
#     print("✅ The two JSON files are identical.")
# else:
#     print("❌ The two JSON files are different.")
    



# Count unique entities
# Extract unique document IDs from corrected_annotations
annotations_lauren = zilal_data.get("corrected_annotations", [])
unique_doc_ids_lauren = set(entry["document_id"] for entry in annotations_lauren)
num_unique_doc_ids_lauren = len(unique_doc_ids_lauren)

# print(num_unique_doc_ids_jennifer) #119

print(num_unique_doc_ids_lauren)