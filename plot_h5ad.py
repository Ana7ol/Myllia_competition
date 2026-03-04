import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from tqdm import tqdm
import gc

def generate_pseudobulks(
    h5ad_path='training_cells.h5ad', 
    target_genes_csv='sample_submission.csv',
    n_bulks_per_pert=100, 
    cells_per_bulk=40,
    output_file='augmented_train_deltas.csv'
):
    print("1. Loading target genes...")
    sub_df = pd.read_csv(target_genes_csv, nrows=0)
    target_genes =[c for c in sub_df.columns if c != 'pert_id']
    print(f"   Found {len(target_genes)} target genes.")

    print(f"2. Loading raw single-cell data from {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    
    print("   Available obs columns:", list(adata.obs.columns))
    
    # ---------------- FIXED COLUMN NAME ----------------
    obs_col = 'sgrna_symbol' 
    # ---------------------------------------------------

    print("3. Applying strict Kaggle host normalization (log1p base 2)...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata, base=2)
    
    print("4. Subsetting to the 5,127 challenge genes...")
    valid_genes =[g for g in target_genes if g in adata.var_names]
    adata = adata[:, valid_genes].copy()
    
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    elif not sp.isspmatrix_csr(adata.X):
        adata.X = adata.X.tocsr()

    print("5. Calculating the 'non-targeting' baseline...")
    # Let's verify exactly what the control is called in the data
    unique_perts = adata.obs[obs_col].unique()
    print(f"   Total unique perturbations found: {len(unique_perts)}")
    
    control_name = 'non-targeting'
    if control_name not in unique_perts:
        raise ValueError(f"Could not find '{control_name}' in adata.obs['{obs_col}']. Available: {unique_perts[:5]}...")

    ctrl_cells = adata[adata.obs[obs_col] == control_name]
    print(f"   Found {len(ctrl_cells)} control cells.")
    
    # Compute the arithmetic mean of the normalized log-counts (baseline)
    baseline_expr = np.ravel(ctrl_cells.X.mean(axis=0))
    
    print(f"6. Generating {n_bulks_per_pert} pseudo-bulks per perturbation...")
    perturbations = [p for p in unique_perts if p != control_name]
    
    augmented_rows =[]
    
    for pert in tqdm(perturbations, desc="Processing Perturbations"):
        pert_adata = adata[adata.obs[obs_col] == pert]
        n_available_cells = pert_adata.n_obs
        
        if n_available_cells == 0:
            continue
            
        for _ in range(n_bulks_per_pert):
            idx = np.random.choice(n_available_cells, size=cells_per_bulk, replace=True)
            sampled_X = pert_adata.X[idx, :]
            
            # Arithmetic mean of the log fold-change vectors
            pseudo_bulk_mean = np.ravel(sampled_X.mean(axis=0))
            
            # Calculate Delta
            delta = pseudo_bulk_mean - baseline_expr
            
            row_dict = {gene: val for gene, val in zip(valid_genes, delta)}
            row_dict['pert_symbol'] = pert
            augmented_rows.append(row_dict)
            
    print("7. Compiling and saving augmented dataset...")
    df_augmented = pd.DataFrame(augmented_rows)
    
    cols = ['pert_symbol'] + valid_genes
    df_augmented = df_augmented[cols]
    
    df_augmented.to_csv(output_file, index=False)
    print(f"✅ Done! Saved {len(df_augmented)} augmented training rows to '{output_file}'.")
    
    del adata, ctrl_cells, pert_adata
    gc.collect()

    return df_augmented

if __name__ == "__main__":
    df = generate_pseudobulks()
    print("Preview of augmented data:")
    print(df.iloc[:5, :5])  # Just print a small 5x5 corner