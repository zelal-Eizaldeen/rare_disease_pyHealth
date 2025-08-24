#!/usr/bin/env python
# coding: utf-8

# # Rare Disease Extraction

# In[1]:


# Simple RDMA Pipeline Test for Jupyter Notebook

import numpy as np
import torch
from rdma.rd.extractor import RDMAExtractor
from rdma.rd.verifier import RDMAVerifier
from rdma.rd.matcher import RDMAMatcher
from rdma.utils.llm_client import LocalLLMClient
from rdma.utils.embedding import EmbeddingsManager

# Sample clinical texts
sample_texts = [
    "Patient presents with signs of Fabry disease including angiokeratomas and acroparesthesias. "
    "Family history is significant for renal failure in maternal uncle. "
    "No evidence of Gaucher disease. Labs show elevated globotriaosylceramide levels.",
    
    "43-year-old female with SLE and progressive dyspnea. Workup revealed PAH with RVSP of 68 mmHg. "
    "Diagnosis of CTD-PAH was made. Also noted was mild achalasia and Raynaud's phenomenon, "
    "raising suspicion for limited scleroderma (CREST syndrome)."
]

# Initialize LLM client
device = "cuda:0" if torch.cuda.is_available() else "cpu"
llm_client = LocalLLMClient(
    model_type="mistral_24b",  # Using a smaller model for faster testing
    device=device,
    temperature=0.1
)

# Initialize embedding manager
embeddings_file = "/u/zelalae2/scratch/data/vector_stores/rd_orpha_medembed.npy"
embedding_manager = EmbeddingsManager(
    model_type="sentence_transformer",
    model_name="abhinand/MedEmbed-small-v0.1",
    device=device
)

# Load embedded documents
embedded_documents = np.load(embeddings_file, allow_pickle=True)

# Initialize extractor
extractor = RDMAExtractor(
    llm_client=llm_client,
    extraction_method="retrieval",
    embedding_manager=embedding_manager,
    embedded_documents=embedded_documents,
    window_size=1,
    top_k=5,
    min_sentence_size=200,
    debug=True
)

# Extract entities from sample texts
extraction_results = {}
for i, text in enumerate(sample_texts):
    print(f"\nExtracting entities from text {i+1}:")
    entities_with_contexts = extractor.extract_from_text(text)
    print(f"Found {len(entities_with_contexts)} potential entities")
    
    # Display some extracted entities
    for j, entity in enumerate(entities_with_contexts[:2]):
        print(f"  Entity {j+1}: {entity.get('entity', '')}")
        
    # Store for next step
    extraction_results[f"text_{i+1}"] = {
        "clinical_text": text,
        "entities_with_contexts": entities_with_contexts,
    }

# Initialize verifier
verifier = RDMAVerifier(
    llm_client=llm_client,
    embedding_manager=embedding_manager,
    embedded_documents=embedded_documents,
    verifier_type="multi_stage",
    min_context_length=5,
    debug=True
)

# Verify entities
verification_results = verifier.verify_from_json(extraction_results)

# Display verification results
for text_id, result in verification_results.items():
    verified_entities = result.get("verified_rare_diseases", [])
    print(f"\n{text_id}: Verified {len(verified_entities)} entities as rare diseases")
    
    # Display some verified entities
    for j, entity in enumerate(verified_entities[:2]):
        print(f"  Verified entity {j+1}: {entity.get('entity', '')}")

# Initialize matcher
matcher = RDMAMatcher(
    llm_client=llm_client,
    embedding_manager=embedding_manager,
    embedded_documents=embedded_documents,
    top_k=5,
    debug=True
)

# Match entities to ORPHA codes
matching_results = matcher.match_from_json(verification_results)

# Display matching results
for text_id, result in matching_results.items():
    matched_entities = result.get("matched_diseases", [])
    print(f"\n{text_id}: Matched {len(matched_entities)} entities to ORPHA codes")
    
    # Display some matched entities
    for j, entity in enumerate(matched_entities[:2]):
        print(f"  Entity: {entity.get('entity', '')}")
        print(f"  Matched to: {entity.get('rd_term', '')} (ORPHA:{entity.get('orpha_id', '')})")
        print(f"  Method: {entity.get('match_method', '')}, Confidence: {entity.get('confidence_score', 0):.2f}")


# # Phenotype Extraction

# In[2]:


#!/usr/bin/env python
# coding: utf-8

import torch
from typing import List, Dict, Any
from pprint import pprint

# Import the HPO pipeline modules
from rdma.hpo.extractor import PhenotypeExtractor
from rdma.hpo.verifier import HPOVerifier
from rdma.hpo.matcher import HPOMatcher
from rdma.utils.llm_client import LocalLLMClient

def run_hpo_pipeline(clinical_texts: List[str]):
    """
    Run the complete HPO pipeline on a list of clinical texts.
    
    Args:
        clinical_texts: List of clinical text strings to process
        
    Returns:
        Tuple of (extracted_entities, verified_phenotypes, matched_phenotypes)
    """
    # Configuration parameters
    model_type = "mistral_24b"
    temperature = 0.001
    cache_dir = "/u/zelalae2/scratch/rdma_cache"
    entity_extractor_type = "retrieval"
    embeddings_file = "/u/zelalae2/scratch/data/vector_stores/G2GHPO_metadata_medembed.npy"
    lab_embeddings_file = "/u/zelalae2/scratch/data/vector_stores/lab_tables_medembed_sm.npy"
    retriever = "sentence_transformer"
    retriever_model = "abhinand/MedEmbed-small-v0.1"
    verifier_version = "v4"
    top_k = 5
    debug = True

    # Initialize LLM client
    print("Initializing LLM client...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    llm_client = LocalLLMClient(
        model_type=model_type,
        device=device,
        cache_dir=cache_dir,
        temperature=temperature
    )
    
    # Initialize pipeline components
    print("\nInitializing pipeline components...")
    extractor = PhenotypeExtractor(
        llm_client=llm_client,
        extractor_type=entity_extractor_type,
        retriever=retriever,
        retriever_model=retriever_model,
        embeddings_file=embeddings_file,
        top_k=top_k,
        debug=debug
    )
    
    verifier = HPOVerifier(
        verifier_version=verifier_version,
        embeddings_file=embeddings_file,
        lab_embeddings_file=lab_embeddings_file,
        retriever=retriever,
        retriever_model=retriever_model,
        llm_client=llm_client,
        debug=debug,
        use_demographics=True
    )
    
    matcher = HPOMatcher(
        llm_client=llm_client,
        embeddings_file=embeddings_file,
        retriever=retriever,
        retriever_model=retriever_model,
        top_k=top_k,
        debug=debug
    )
    
    # Process all clinical texts
    all_extracted_entities = []
    all_verified_phenotypes = []
    all_matched_phenotypes = []
    
    for i, clinical_text in enumerate(clinical_texts):
        print(f"\n\n{'='*80}")
        print(f"Processing Clinical Text #{i+1}")
        print(f"{'='*80}")
        
        # Step 1: Extract phenotype entities
        print("\nStep 1: Extracting entities...")
        entities_with_contexts = extractor.extract([clinical_text])
        print(f"Extracted {len(entities_with_contexts)} entities with contexts.")
        
        for entity in entities_with_contexts:
            print(f"Entity: {entity.get('entity')}")
            print(f"Context: {entity.get('context')}")
            print("---")
        
        # Step 2: Verify entities as phenotypes
        print("\nStep 2: Verifying phenotypes...")
        verified_phenotypes = verifier.verify(entities_with_contexts, clinical_text)
        print(f"Verified {len(verified_phenotypes)} phenotypes.")
        
        for phenotype in verified_phenotypes:
            print(f"Phenotype: {phenotype.get('phenotype', phenotype.get('entity', ''))}")
            print(f"Status: {phenotype.get('status', 'unknown')}")
            if 'lab_info' in phenotype:
                lab_info = phenotype.get('lab_info', {})
                print(f"Lab: {lab_info.get('lab_name')} = {lab_info.get('value')} ({lab_info.get('direction')})")
            print("---")
        
        # Step 3: Match verified phenotypes to HPO codes
        print("\nStep 3: Matching phenotypes to HPO codes...")
        matched_phenotypes = matcher.match(verified_phenotypes)
        print(f"Matched {len(matched_phenotypes)} phenotypes to HPO codes.")
        
        for match in matched_phenotypes:
            print(f"Entity: {match.get('phenotype', '')}")
            print(f"Context: {match.get('context', '')}")
            print(f"Type: {match.get('status', '')}")
            print(f"Code: {match.get('hp_id', '')}")
            print("---")
        
        # Collect results for each text
        all_extracted_entities.extend(entities_with_contexts)
        all_verified_phenotypes.extend(verified_phenotypes)
        all_matched_phenotypes.extend(matched_phenotypes)
    
    # Print summary
    print("\n\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    print(f"Total extracted entities: {len(all_extracted_entities)}")
    print(f"Total verified phenotypes: {len(all_verified_phenotypes)}")
    print(f"Total matched HPO codes: {len(all_matched_phenotypes)}")
    
    # Count direct vs. implied phenotypes
    direct_count = sum(1 for p in all_verified_phenotypes if p.get('status') == 'direct_phenotype')
    implied_count = sum(1 for p in all_verified_phenotypes if p.get('status') == 'implied_phenotype')
    print(f"Direct phenotypes: {direct_count}")
    print(f"Implied phenotypes: {implied_count}")
    
    return all_extracted_entities, all_verified_phenotypes, all_matched_phenotypes


if __name__ == "__main__":
    # Example clinical texts to process
    clinical_texts = [
        """
        Patient is a 5-year-old male with history of developmental delay and seizures. 
        Physical examination reveals macrocephaly, with head circumference in the 98th percentile.
        He has hypotonia and hyperreflexia. EEG showed abnormal spike-wave discharges.
        Genetic testing reveals a pathogenic variant in MECP2. 
        Lab results show elevated ammonia at 150 μmol/L (normal range 10-35).
        """,
        
        """
        42-year-old female with progressive vision loss. Fundoscopic examination shows 
        bilateral retinitis pigmentosa. Patient reports night blindness since adolescence.
        Family history is significant for similar symptoms in mother and maternal uncle.
        Physical exam also reveals polydactyly of both hands with six digits on each hand.
        Renal function is abnormal with elevated creatinine of 2.1 mg/dL.
        """
    ]
    
    # Run the HPO pipeline
    extracted, verified, matched = run_hpo_pipeline(clinical_texts)

