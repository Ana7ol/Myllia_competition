import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import os

def explore_l1000(gctx_path='GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx'):
    print(f"1. Opening L1000 GCTX file: {gctx_path}...")
    
    if not os.path.exists(gctx_path):
        print(f"File {gctx_path} not found. Please check the filename.")
        return
        
    with h5py.File(gctx_path, 'r') as f:
        # GCTX structure: /0/META/ROW/id and /0/META/COL/id
        row_ids =[x.decode('utf-8') for x in f['/0/META/ROW/id'][()]]
        col_ids = [x.decode('utf-8') for x in f['/0/META/COL/id'][()]]
        
        matrix_shape = f['/0/DATA/0/matrix'].shape
        
    print(f"   Matrix Shape: {matrix_shape} (Signatures x Genes)")
    print(f"   Total Measured Genes (Rows): {len(row_ids)}")
    print(f"   Total Signatures (Cols): {len(col_ids)}")
    print(f"   Sample Row IDs (usually Entrez IDs): {row_ids[:5]}")
    print(f"   Sample Col IDs (Signature IDs): {col_ids[:5]}")
    
    print("\n--- NEXT STEPS FOR COVERAGE ---")
    print("Because L1000 uses numeric Entrez IDs and complex Signature IDs,")
    print("we need the 'sig_info.txt' and 'gene_info.txt' files from the GEO database.")
    print("Once we map these to Gene Symbols (e.g., 'TP53'), we can generate the Venn diagram!")

    # SIMULATING the overlap graph just to have the plotting code ready:
    print("\n2. Generating expected coverage graph (Simulation until metadata is mapped)...")
    
    # Let's assume we mapped them and found 9,000 overlapping genes
    myllia_genes = set([f"Gene_{i}" for i in range(5127)])
    l1000_genes = set([f"Gene_{i}" for i in range(2000, 11000)]) # Simulated overlap
    
    plt.figure(figsize=(8, 6))
    venn2([myllia_genes, l1000_genes], ('Myllia Target Genes (5,127)', 'L1000 Landmark Genes (~12k)'))
    plt.title("Expected Gene Overlap: Myllia vs. LINCS L1000", fontsize=14, fontweight='bold')
    plt.savefig("gene_coverage_venn.png", dpi=300)
    print("✅ Saved overlap chart as gene_coverage_venn.png")

if __name__ == "__main__":
    explore_l1000()