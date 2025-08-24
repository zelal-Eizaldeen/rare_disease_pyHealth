import json
import pathlib
import argparse
from pyhealth.datasets import MIMIC4NoteDataset


def load_processed_ids(filepath: str) -> set:
    """Load the set of already processed patient IDs from JSON file."""
    try:
        with open(filepath, "r") as f:
            processed = json.load(f)
            if not isinstance(processed, list):
                raise ValueError("Processed file must contain a JSON list.")
            return set(processed)
    except (FileNotFoundError, json.JSONDecodeError):
        # If file missing or empty/invalid, return empty set
        return set()


def save_processed_ids(filepath: str, processed_ids: set):
    """Save the updated set of processed patient IDs to JSON file."""
    with open(filepath, "w") as f:
        json.dump(sorted(processed_ids), f, indent=2)


def sample_new_patients(note_root: str, processed_file: str, sample_size: int, text_column: str) -> dict:
    """
    Sample new patient notes from MIMIC-IV excluding already processed patients.

    Args:
        note_root: Root directory of the MIMIC-IV note dataset.
        processed_file: JSON file path containing processed patient IDs.
        sample_size: Number of new patients to sample.
        text_column: Column name for clinical text.

    Returns:
        Dictionary of sampled patient notes keyed by patient_id.
    """
    processed_ids = load_processed_ids(processed_file)
    print(f"Loaded {len(processed_ids)} processed patient IDs.")

    # Load dataset
    dataset = MIMIC4NoteDataset(root=note_root, tables=["discharge"])
    note_df = dataset.global_event_df.collect().to_pandas()

    # Group notes by patient_id and join text entries
    patient_notes = note_df.groupby("patient_id")[text_column].apply(lambda texts: "\n\n".join(texts))

    # Filter long notes
    filtered_notes = patient_notes[patient_notes.str.len() > 500]

    # Exclude already processed patients
    remaining_notes = filtered_notes[~filtered_notes.index.isin(processed_ids)]

    # Sample new patients
    new_samples = remaining_notes.sample(n=sample_size, random_state=42)

    # Update processed ids
    processed_ids.update(new_samples.index.astype(str))
    save_processed_ids(processed_file, processed_ids)

    print(f"Sampled {len(new_samples)} new patients.")
    print(f"Total processed patient count is now {len(processed_ids)}.")

    # Prepare output dictionary
    sampled_data = {
        str(pid): {
            "clinical_text": text,
            "patient_id": str(pid),
            "hadm_id": "",
            "category": "",
            "chartdate": ""
        }
        for pid, text in new_samples.items()
    }

    return sampled_data


def main():
    parser = argparse.ArgumentParser(description="Sample new MIMIC-IV patient notes excluding processed IDs.")
    parser.add_argument("--note_root", type=str, required=True, help="Root directory for MIMIC-IV notes dataset")
    parser.add_argument("--processed_file", type=str, required=True, help="Path to JSON file with processed patient IDs")
    parser.add_argument("--sample_size", type=int, default=1000, help="Number of new patients to sample")
    parser.add_argument("--text_column", type=str, default="discharge/text", help="Text column to use from dataset")

    args = parser.parse_args()

    new_patient_notes = sample_new_patients(
        note_root=args.note_root,
        processed_file=args.processed_file,
        sample_size=args.sample_size,
        text_column=args.text_column,
    )

    output_path = pathlib.Path(args.processed_file).parent / f"input_patient_notes_{args.sample_size}.json"
    with open(output_path, "w") as f_out:
        json.dump(new_patient_notes, f_out, indent=2)
    print(f"Saved new patient notes to {output_path}")


if __name__ == "__main__":
    main()
