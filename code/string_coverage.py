import pandas as pd
import urllib.request
import os
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

def evaluate_string_coverage():
    print("1. Loading Myllia Competition Genes...")
    # Load Myllia target genes
    sub_df = pd.read_csv('sample_submission.csv', nrows=0)
    myllia_targets = set([c for c in sub_df.columns if c != 'pert_id'])
    
    # Load Myllia Perturbed genes (Train + Val)
    train_df = pd.read_csv('training_data_means.csv', usecols=['pert_symbol'])
    train_perts = set(train_df['pert_symbol'].dropna().str.upper()) - {'NON-TARGETING'}
    
    val_df = pd.read_csv('pert_ids_val.csv')
    val_perts = set(val_df['pert'].dropna().str.upper())
    
    all_myllia_genes = myllia_targets.union(train_perts).union(val_perts)
    print(f"   Myllia Target Genes: {len(myllia_targets)}")
    print(f"   Myllia Perturbed Genes: {len(train_perts) + len(val_perts)}")
    print(f"   Total Unique Myllia Genes to map: {len(all_myllia_genes)}")

    print("\n2. Downloading/Loading STRING DB Metadata (v12.0)...")
    string_info_file = "9606.protein.info.v12.0.txt.gz"
    string_url = "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz"
    
    if not os.path.exists(string_info_file):
        print(f"   Downloading {string_info_file} (~3MB)...")
        urllib.request.urlretrieve(string_url, string_info_file)
        print("   Download complete.")
    else:
        print(f"   Found existing {string_info_file}.")

    # Load STRING mapping (columns: protein_external_id, preferred_name, protein_size, annotation)
    string_df = pd.read_csv(string_info_file, sep='\t')
    
    # The 'preferred_name' in STRING human is usually the HGNC gene symbol!
    string_genes = set(string_df['preferred_name'].dropna().str.upper())
    print(f"   STRING Total Network Nodes (Genes/Proteins): {len(string_genes)}")

    print("\n3. Calculating Overlaps...")
    
    # Target Overlap
    target_overlap = myllia_targets.intersection(string_genes)
    target_pct = (len(target_overlap) / len(myllia_targets)) * 100
    print(f"   --> Target Gene Coverage in STRING: {len(target_overlap)} / {len(myllia_targets)} ({target_pct:.1f}%)")

    # Perturbation Overlap (Can we inject the signal into the graph?)
    train_overlap = train_perts.intersection(string_genes)
    val_overlap = val_perts.intersection(string_genes)
    
    print(f"   --> Training Perts in STRING: {len(train_overlap)} / {len(train_perts)} ({(len(train_overlap)/len(train_perts))*100:.1f}%)")
    print(f"   --> Validation Perts in STRING: {len(val_overlap)} / {len(val_perts)} ({(len(val_overlap)/len(val_perts))*100:.1f}%)")

    print("\n4. Generating Plot...")
    plt.figure(figsize=(8, 6))
    venn2([all_myllia_genes, string_genes], ('Myllia All Genes (Targets + Perts)', 'STRING Network Nodes'))
    plt.title("STRING Network Node Coverage", fontsize=14, fontweight='bold')
    plt.savefig("string_coverage_venn.png", dpi=300)
    plt.close()
    
    print("✅ Saved 'string_coverage_venn.png'")
    
    if len(val_overlap) == len(val_perts):
        print("\n🔥 EXCELLENT! All validation perturbations exist in STRING. We can safely build a Graph Neural Network!")

if __name__ == "__main__":
    evaluate_string_coverage()