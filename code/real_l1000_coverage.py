import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import seaborn as sns

def evaluate_real_l1000_coverage():
    print("1. Loading Myllia Competition Targets...")
    # Load Myllia target genes
    sub_df = pd.read_csv('sample_submission.csv', nrows=0)
    myllia_genes = set([c for c in sub_df.columns if c != 'pert_id'])
    print(f"   Myllia Target Genes: {len(myllia_genes)}")

    # Load Myllia Train and Validation Perturbations
    train_df = pd.read_csv('training_data_means.csv', usecols=['pert_symbol'])
    train_perts = set(train_df['pert_symbol'].dropna().str.upper()) - {'NON-TARGETING'}
    
    val_df = pd.read_csv('pert_ids_val.csv')
    val_perts = set(val_df['pert'].dropna().str.upper())
    print(f"   Myllia Training Perts: {len(train_perts)}")
    print(f"   Myllia Validation Perts: {len(val_perts)}")

    print("\n2. Loading LINCS L1000 Metadata...")
    # Load LINCS Genes
    gene_info = pd.read_csv('GSE70138_Broad_LINCS_gene_info_2017-03-06.txt', sep='\t', dtype=str)
    l1000_genes = set(gene_info['pr_gene_symbol'].dropna().str.upper())
    print(f"   L1000 Total Measured Genes: {len(l1000_genes)}")

    # Load LINCS Signatures
    sig_info = pd.read_csv('GSE70138_Broad_LINCS_sig_info_2017-03-06.txt', sep='\t', dtype=str, 
                           usecols=['sig_id', 'pert_iname', 'pert_type', 'cell_id'])
    
    # Filter for genetic knockdowns: trt_sh (shRNA) and trt_xpr (CRISPR)
    # CRISPRi in the competition suppresses gene expression, so both are valid analogs!
    kd_sigs = sig_info[sig_info['pert_type'].isin(['trt_sh', 'trt_xpr'])]
    l1000_perts = set(kd_sigs['pert_iname'].dropna().str.upper())
    print(f"   L1000 Total Knockdown Perturbations: {len(l1000_perts)}")

    print("\n3. Calculating Overlaps...")
    
    # Gene Overlap
    gene_overlap = myllia_genes.intersection(l1000_genes)
    print(f"   --> Gene Overlap: {len(gene_overlap)} out of {len(myllia_genes)} Myllia genes are in L1000!")

    # Perturbation Overlap
    train_overlap = train_perts.intersection(l1000_perts)
    val_overlap = val_perts.intersection(l1000_perts)
    print(f"   --> Training Pert Coverage: {len(train_overlap)} / {len(train_perts)} ({(len(train_overlap)/len(train_perts))*100:.1f}%)")
    print(f"   --> Validation Pert Coverage: {len(val_overlap)} / {len(val_perts)} ({(len(val_overlap)/len(val_perts))*100:.1f}%)")

    print("\n4. Generating Plots...")
    
    # Plot 1: Gene Overlap Venn
    plt.figure(figsize=(8, 6))
    venn2([myllia_genes, l1000_genes], ('Myllia Target Genes', 'L1000 Measured Genes'))
    plt.title("Real Gene Overlap: Myllia vs. LINCS L1000", fontsize=14, fontweight='bold')
    plt.savefig("real_gene_overlap_venn.png", dpi=300)
    plt.close()

    # Plot 2: Perturbation Coverage Bar Chart
    plt.figure(figsize=(7, 5))
    coverage_data = pd.DataFrame({
        'Dataset': ['Training', 'Validation'],
        'Total':[len(train_perts), len(val_perts)],
        'Found in L1000': [len(train_overlap), len(val_overlap)]
    })
    
    # Melting for seaborn barplot
    melted = coverage_data.melt(id_vars='Dataset', var_name='Metric', value_name='Count')
    
    sns.barplot(data=melted, x='Dataset', y='Count', hue='Metric', palette=['lightgray', 'steelblue'])
    plt.title("L1000 Coverage of Myllia Perturbations", fontsize=14, fontweight='bold')
    plt.ylabel("Number of Perturbations (Genes)")
    
    # Add percentage labels
    for i, (total, found) in enumerate(zip(coverage_data['Total'], coverage_data['Found in L1000'])):
        pct = (found / total) * 100
        plt.text(i, found + 1, f"{pct:.1f}%", ha='center', va='bottom', fontweight='bold', color='darkblue')

    plt.tight_layout()
    plt.savefig("real_pert_coverage_bars.png", dpi=300)
    plt.close()

    print("✅ Saved 'real_gene_overlap_venn.png' and 'real_pert_coverage_bars.png'")

    # Save mapping info for the feature extraction step
    overlapping_val_perts = list(val_overlap)
    pd.Series(overlapping_val_perts).to_csv("l1000_matched_val_perts.csv", index=False, header=["pert_symbol"])
    
    return gene_overlap, train_overlap, val_overlap

if __name__ == "__main__":
    evaluate_real_l1000_coverage()