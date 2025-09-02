import argparse
import sys
import os
import json
import re
from pathlib import Path

from datetime import datetime
from typing import List, Dict, Tuple, Any,  Optional, Union
import pronto
sys.path.append("/projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA")


####################################################
# Extract rare diseases from MONDO OBO
####################################################
def build_rare_disease_set(obo_path: str, out_path: str = "/projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/ontology_data/rare_diseases.json"):
    ontology = pronto.Ontology(obo_path)
    rare_diseases = set()

    for term in ontology.terms():
        if not term.name:
            continue
        # Check ORDO (Orphanet) cross refs
        xrefs = [xref.id.lower() for xref in term.xrefs]
        if any("ordo" in x for x in xrefs):
            rare_diseases.add(term.name.lower())
            continue
        # Check subsets
        subsets = [s.lower() for s in term.subsets]
        if any("rare" in s for s in subsets):
            rare_diseases.add(term.name.lower())

    # Save once to JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sorted(list(rare_diseases)), f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(rare_diseases)} rare diseases from MONDO")
    return rare_diseases

####################################################
# Load rare diseases (use pre-saved JSON if available)
####################################################
def load_rare_diseases(obo_file="/projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/ontology_data/mondo.obo", json_file="/projects/illinois/eng/cs/jimeng/zelalae2/scratch/RDMA/zilal_contribution/ontology_data/rare_diseases.json"):
    if Path(json_file).exists():
        print("Zilal json files YES")
        with open(json_file, "r") as f:
            return set(json.load(f))
    else:
        return build_rare_disease_set(obo_file, json_file)

#Added by Zilal
def clean_llm_predictions(predictions: List[str], rare_disease_set: Optional[set] = None) -> List[str]:
    """
    Filter and clean LLM predictions to remove non-diseases or common conditions.
    If rare_disease_set is provided, only keep items that match it.
    """
    cleaned = []
    seen = set()

    for item in predictions:
        # Strip reasoning text like "analysis..." or "we need..."
        item = re.sub(r'analysis.*|We need.*', '', item, flags=re.IGNORECASE).strip()

        if not item or len(item) < 3:
            continue

        item_norm = item.lower()

        # Example: skip obvious non-rare/common terms
        blacklist = {"diverticulosis", "hyperlipidemia", "hypertension", "splenomegaly", 
                     "anemia", "mediastinal lymphadenopathy"}  
        if item_norm in blacklist:
            continue

        # If ontology set provided, only keep matches
        if rare_disease_set and item_norm not in rare_disease_set:
            continue

        if item not in seen:
            seen.add(item)
            cleaned.append(item)

    return cleaned
#Ended by Zilal

def parse_diseases_list(response: str) -> List[str]:
    """
    Robust function to parse LLM response into Python list
    """
    try:
        # First, try direct JSON parsing
        cleaned = response.strip()
        diseases_list = json.loads(cleaned)

        # Validate it's a list
        if isinstance(diseases_list, list):
            return diseases_list
        else:
            return []

    except json.JSONDecodeError:
        # Try to extract JSON array using regex
        json_pattern = r'\[(.*?)\]'
        match = re.search(json_pattern, response, re.DOTALL)

        if match:
            try:
                json_content = '[' + match.group(1) + ']'
                diseases_list = json.loads(json_content)
                return diseases_list
            except json.JSONDecodeError:
                pass

        # Last resort: manual parsing for common formats
        try:
            # Remove brackets and split by comma
            content = response.strip()
            content = re.sub(r'^\[|\]$', '', content)  # Remove outer brackets

            # Split by comma and clean each item
            items = [item.strip().strip('"\'') for item in content.split(',')]
            items = [item for item in items if item]  # Remove empty items

            return items

        except Exception:
            return []

def normalize_disease_name(disease: str) -> str:
    """
    Normalize disease name for comparison: lowercase and strip whitespace
    """
    return disease.lower().strip()
####################################################
# Integrate rare-disease filtering here
####################################################
def benchmark_rare_disease_diagnosis(
    data: Dict[str, Any],
    llm_client: Any,
    rare_diseases: set, 
    num_samples: int = None,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Benchmark rare disease diagnosis performance

    Args:
        data: Dictionary containing patient data with phenotypes and disease entities
        llm_client: LLM client with query method
        num_samples: Number of samples to evaluate (None for all)
        verbose: Whether to print detailed results for each case

    Returns:
        Dictionary with benchmark metrics
    """

    # System prompt for LLM
    diff_diag_sys_prompt = """Given the following phenotypes, identify the top 10 most likely rare diseases.

CRITICAL: Your response must be EXACTLY in this JSON format with no additional text:

["Disease Name 1", "Disease Name 2", "Disease Name 3", "Disease Name 4", "Disease Name 5", "Disease Name 6", "Disease Name 7", "Disease Name 8", "Disease Name 9", "Disease Name 10"]

Rules:
1. Return ONLY the JSON array - no explanations, no additional text
2. Use double quotes around disease names
3. Separate diseases with commas
4. First disease = most likely, last disease = least likely
5. If you find fewer than 10 diseases, that's okay
6. If you cannot find any rare diseases, return: []

Example of correct format:
["Marfan Syndrome", "Ehlers-Danlos Syndrome", "Osteogenesis Imperfecta"]

Your response:"""
   
    K_LIST = (1, 5, 10)

    # Micro (disease-level) counters
    total_diseases = 0
    hits_by_k = {k: 0 for k in K_LIST}
    # Patient-level counters
    total_patients = 0
    patient_hits_by_k = {k: 0 for k in K_LIST}

    parsing_failures = 0

    # Get samples to process
    items_to_process = list(data.items())
    if num_samples is not None:
        items_to_process = items_to_process[:num_samples]

    if verbose:
        print(f"Evaluating {len(items_to_process)} patients...")
        print("=" * 60)

    for patient_id, patient_data in items_to_process:
        if 'matched_phenotypes' not in patient_data:
            continue

        total_patients += 1
    
        # Build phenotypes string
        phenotypes = ", ".join(p['phenotype'] for p in patient_data['matched_phenotypes']) #By Zilal. more efficient

        # Query LLM
        phenotypes_prompt = f"Phenotypes: {phenotypes}"
        llm_response = llm_client.query(
            system_message=diff_diag_sys_prompt,
            user_input=phenotypes_prompt
        )

        # Parse response
        # predicted_diseases = parse_diseases_list(llm_response) #By Zilal
        predicted_diseases = parse_diseases_list(llm_response)
        # keep only rare diseases
        predicted_diseases = [d for d in predicted_diseases if normalize_disease_name(d) in rare_diseases]
        # predicted_diseases = clean_llm_predictions(predicted_diseases, rare_disease_set=None)

        if not predicted_diseases:
            parsing_failures += 1
            if verbose:
                print(f"Patient {patient_id}: Failed to parse LLM response")
                print(f"Raw response: '{llm_response}'")
            continue

        # Normalize predicted diseases for comparison
        predicted_normalized = [normalize_disease_name(d) for d in predicted_diseases]

        # Get ground truth diseases
        observed_diseases = patient_data.get('disease_entities', [])
        
        #Zilal
        observed_normalized = [normalize_disease_name(d) for d in observed_diseases]


        if verbose:
            print(f"Patient ID: {patient_id}")
            print(f"Phenotypes: {phenotypes.strip(', ')}")
            print(f"Predicted diseases: {predicted_diseases}")
            print(f"Observed diseases: {observed_diseases}")
            
        # Precompute top-K sets for quick membership checks #By Zilal
        topk_sets = {k: set(predicted_normalized[:k]) for k in K_LIST}
        
        #Zilal
        # ----- Micro (disease-level) -----
        for obs in observed_normalized:
            total_diseases += 1
            for k in K_LIST:
                if obs in topk_sets[k]:
                    hits_by_k[k] += 1

        #Zilal ----- Patient-level -----
        for k in K_LIST:
            if set(observed_normalized).intersection(topk_sets[k]):
                patient_hits_by_k[k] += 1
          
        #Zilal
        if verbose:
            # Show per-patient hit flags
            flags = {k: (set(observed_normalized).intersection(topk_sets[k]) != set()) for k in K_LIST}
            print(f"  Patient hits@K: {flags}")
            print("-" * 60)  
            
    #Zilal - Final metrics
    micro_hit_at_k_rate = {k: (hits_by_k[k] / total_diseases if total_diseases > 0 else 0.0) for k in K_LIST}
    patient_hit_at_k_rate = {k: (patient_hits_by_k[k] / total_patients if total_patients > 0 else 0.0) for k in K_LIST}
    parsing_success_rate = 1 - (parsing_failures / total_patients) if total_patients > 0 else 0.0

    # Zilal-Backward-compatible fields (your earlier names)
    results = {
        # Micro @K (disease-level)
        'micro_hit_at_1_rate':  micro_hit_at_k_rate[1],
        'micro_hit_at_5_rate':  micro_hit_at_k_rate[5],
        'micro_hit_at_10_rate': micro_hit_at_k_rate[10],
        'micro_hits_at_1':      hits_by_k[1],
        'micro_hits_at_5':      hits_by_k[5],
        'micro_hits_at_10':     hits_by_k[10],

        # Patient-level @K
        'patient_hit_at_1_rate':  patient_hit_at_k_rate[1],
        'patient_hit_at_5_rate':  patient_hit_at_k_rate[5],
        'patient_hit_at_10_rate': patient_hit_at_k_rate[10],
        'patients_with_hits_at_1':  patient_hits_by_k[1],
        'patients_with_hits_at_5':  patient_hits_by_k[5],
        'patients_with_hits_at_10': patient_hits_by_k[10],

        # Totals
        'total_diseases': total_diseases,
        'total_patients': total_patients,
        'parsing_success_rate': parsing_success_rate,
        'parsing_failures': parsing_failures,
    }

    # Legacy aliases so your existing code/plots don’t break:
    results.update({
        'hit_rate':        micro_hit_at_k_rate[10],   # == micro Hit@10
        'hit_at_1_rate':   micro_hit_at_k_rate[1],    # == micro Hit@1
        'patient_hit_rate': patient_hit_at_k_rate[10],# == patient Hit@10
        'hits':            hits_by_k[10],
        'hits_at_1':       hits_by_k[1],
        'patients_with_hits': patient_hits_by_k[10],
    })    
    
    return results

def print_benchmark_results(results: Dict[str, float]) -> None:
    """
    Print benchmark results in a formatted way
    """
    print("\n" + "=" * 50)
    print("RARE DISEASE DIAGNOSIS BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Total patients evaluated: {results['total_patients']}")
    print(f"Total diseases to predict: {results['total_diseases']}")
    print(f"LLM parsing success rate: {results['parsing_success_rate']:.2%}")
    print("-" * 50)
    print(f"Hit Rate (Top-10): {results['hit_rate']:.2%} ({results['hits']}/{results['total_diseases']})")
    print(f"Hit@1 Rate: {results['hit_at_1_rate']:.2%} ({results['hits_at_1']}/{results['total_diseases']})")
    print(f"Patient Hit Rate: {results['patient_hit_rate']:.2%} ({results['patients_with_hits']}/{results['total_patients']})")
    print("=" * 50)
#Save computed results of llm inference into json
def build_pretty_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Add human-readable percentage strings alongside the raw numbers."""
    d = dict(results)  # raw (floats/ints)
    pretty = {
        "hit_rate": f"{results.get('hit_rate', 0):.2%}",
        "hit_at_1_rate": f"{results.get('hit_at_1_rate', 0):.2%}",
        "patient_hit_rate": f"{results.get('patient_hit_rate', 0):.2%}",
        "parsing_success_rate": f"{results.get('parsing_success_rate', 0):.2%}",
        "counts": {
            "total_patients": results.get("total_patients", 0),
            "total_diseases": results.get("total_diseases", 0),
            "hits": results.get("hits", 0),
            "hits_at_1": results.get("hits_at_1", 0),
            "patients_with_hits": results.get("patients_with_hits", 0),
            "parsing_failures": results.get("parsing_failures", 0),
        }
    }
    return {"raw": d, "pretty": pretty}

def save_benchmark_results_json(
    results: Dict[str, Any],
    out_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Save benchmark results to JSON (raw numbers + pretty strings)."""
    payload = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            **(metadata or {})
        },
        "results": build_pretty_results(results)
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return str(out_path)


if __name__ == "__main__":
    from rdma.utils.data import read_json_file

    parser = argparse.ArgumentParser(description="Rare Disease Benchmark")
    parser.add_argument("--model", type=str, default="qwen3_32b",
                        help="Model type (e.g., mistral_24b, llama3_70b, qwen3_32b, gpt_oss_20b)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run the model on (e.g., cuda:0, cpu)")
    parser.add_argument("--temperature", type=float, default=0.001,
                        help="Sampling temperature")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset JSON file")
    parser.add_argument("--out", type=str, default=None,
                        help="Path to save benchmark results JSON")
    parser.add_argument("--mondo", type=str, default="mondo.obo", help="Path to MONDO OBO file")


    args = parser.parse_args()

    # Load dataset
    data = read_json_file(args.dataset)
   
    # Load rare diseases from MONDO
    rare_diseases = load_rare_diseases(args.mondo)

    # Load client dynamically
    from rdma.utils.llm_client import LocalLLMClient
    llm_client = LocalLLMClient(
        model_type=args.model,
        device=args.device,
        temperature=args.temperature
    )
    
    # Run benchmark
    results = benchmark_rare_disease_diagnosis(
        data=data,
        llm_client=llm_client,
        rare_diseases=rare_diseases,
        verbose=True
    )

    # Prepare metadata
    meta = {
        "model": args.model,
        "device": args.device,
        "temperature": args.temperature,
        "dataset": args.dataset,
        "mondo": args.mondo
    }

    # Output file path
    out_path = args.out or f"hits_{args.model}.json"
    out = save_benchmark_results_json(results, out_path, metadata=meta)
    print("Saved:", out)
