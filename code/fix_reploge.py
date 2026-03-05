import os
import pandas as pd
import scanpy as sc

def fix_and_extract_replogle():
    target_file = "K562_gwps_normalized_bulk_01.h5ad"
    
    print("1. Loading Myllia Target Genes and Perturbations...")
    sub_df = pd.read_csv('sample_submission.csv', nrows=0)
    myllia_targets =[c for c in sub_df.columns if c != 'pert_id']
    
    val_df = pd.read_csv('pert_ids_val.csv')
    val_perts = set(val_df['pert'].dropna().str.upper())

    print("\n2. Loading Replogle AnnData...")
    # Load backed='r' to save RAM if necessary, but full load is fine for 357MB
    adata = sc.read_h5ad(target_file)
    
    print("\n3. Fixing Gene Symbol Mismatch (Ensembl -> HGNC)...")
    # Replogle usually stores the human-readable symbol in adata.var['gene_name']
    if 'gene_name' in adata.var.columns:
        adata.var.index = adata.var['gene_name'].astype(str).str.upper()
    else:
        # Fallback: search for any column containing 'name' or 'symbol'
        for col in adata.var.columns:
            if 'name' in col.lower() or 'symbol' in col.lower():
                adata.var.index = adata.var[col].astype(str).str.upper()
                break
                
    valid_genes =[g for g in myllia_targets if g in adata.var_names]
    
    # Handle duplicates in var_names (sometimes 2 Ensembl IDs map to 1 Symbol)
    adata.var_names_make_unique()
    
    # Subset matrix columns to just the Myllia genes
    adata = adata[:, valid_genes]
    print(f"   --> Extracted {len(valid_genes)} / {len(myllia_targets)} target genes.")

    print("\n4. Fixing Perturbation Mismatch...")
    # The perturbation symbol in Replogle K562 is usually the index itself
    # If the index is "TP53_1", splitting by '_' gets "TP53"
    adata.obs['pert_symbol'] = adata.obs.index.astype(str).str.split('_').str[0].str.upper()
    
    rep_perts = set(adata.obs['pert_symbol'])
    val_overlap = val_perts.intersection(rep_perts)
    print(f"   --> Replogle Coverage of Kaggle Validation Set: {len(val_overlap)} / {len(val_perts)} ({(len(val_overlap)/len(val_perts))*100:.1f}%)")

    if len(valid_genes) > 0:
        print("\n5. Saving ML-ready feature matrix...")
        # Convert to Pandas DataFrame
        df_features = pd.DataFrame(adata.X, index=adata.obs.index, columns=adata.var_names)
        df_features['pert_symbol'] = adata.obs['pert_symbol'].values
        
        # Average across multiple sgRNAs for the same gene
        df_consensus = df_features.groupby('pert_symbol').mean().reset_index()
        df_consensus.to_csv("replogle_myllia_features.csv", index=False)
        print("✅ Saved 'replogle_myllia_features.csv'")

if __name__ == "__main__":
    fix_and_extract_replogle()