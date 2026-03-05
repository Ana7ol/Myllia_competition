import requests
import os
import pandas as pd
import scanpy as sc
from tqdm import tqdm

def fetch_and_extract_replogle():
    target_file = "K562_gwps_normalized_bulk_01.h5ad"
    article_id = "20029387"  # Figshare ID for Replogle 2022
    api_url = f"https://api.figshare.com/v2/articles/{article_id}/files"
    
    print(f"1. Querying Figshare API for the 357MB summary file...")
    
    if not os.path.exists(target_file):
        response = requests.get(api_url)
        if response.status_code != 200:
            print("API Error. Please download manually from: https://figshare.com/articles/dataset/_/20029387")
            return
            
        files = response.json()
        download_url = None
        for f in files:
            if f['name'] == target_file:
                download_url = f['download_url']
                break
                
        if not download_url:
            print(f"Could not find {target_file} in the API response.")
            return
            
        print(f"2. Downloading {target_file} (~357 MB) instead of the 155GB raw file...")
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(target_file, 'wb') as f, tqdm(
                desc=target_file,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
        print("   Download complete!")
    else:
        print(f"   {target_file} already exists locally. Skipping download.")

    print("\n3. Loading Myllia Target Genes and Perturbations...")
    sub_df = pd.read_csv('sample_submission.csv', nrows=0)
    myllia_targets =[c for c in sub_df.columns if c != 'pert_id']
    
    val_df = pd.read_csv('pert_ids_val.csv')
    val_perts = set(val_df['pert'].dropna().str.upper())

    print("\n4. Extracting features from Replogle AnnData...")
    # Load the 357MB h5ad file
    adata = sc.read_h5ad(target_file)
    
    # 1. Match Genes (Columns)
    # The var_names in this dataset are usually Gene Symbols or Ensembl IDs.
    # We use upper() to ensure a safe match.
    adata.var.index = adata.var.index.str.upper()
    valid_genes =[g for g in myllia_targets if g in adata.var_names]
    
    # Subset matrix columns to just the Myllia genes
    adata = adata[:, valid_genes]
    print(f"   --> Extracted {len(valid_genes)} / {len(myllia_targets)} target genes.")

    # 2. Match Perturbations (Rows)
    # In Replogle pseudobulk, the perturbation name is typically the index or in adata.obs['condition']
    # If the index looks like "TP53_1", we extract "TP53".
    print(adata.obs.columns)
    adata.obs['pert_symbol'] = adata.obs.index.str.split('_').str[0].str.upper()
    
    # Convert the sparse matrix to a dense Pandas DataFrame
    df_features = pd.DataFrame(adata.X, index=adata.obs.index, columns=adata.var_names)
    df_features['pert_symbol'] = adata.obs['pert_symbol'].values
    
    # Average across multiple sgRNAs for the same gene to get one clean consensus signature
    df_consensus = df_features.groupby('pert_symbol').mean().reset_index()
    
    rep_perts = set(df_consensus['pert_symbol'])
    val_overlap = val_perts.intersection(rep_perts)
    print(f"   --> Replogle Coverage of Kaggle Validation Set: {len(val_overlap)} / {len(val_perts)} ({(len(val_overlap)/len(val_perts))*100:.1f}%)")

    print("\n5. Saving tiny ML-ready feature matrix...")
    df_consensus.to_csv("replogle_myllia_features.csv", index=False)
    print("✅ Saved 'replogle_myllia_features.csv'")
    print(f"   File size: {os.path.getsize('replogle_myllia_features.csv') / (1024*1024):.2f} MB")

if __name__ == "__main__":
    fetch_and_extract_replogle()