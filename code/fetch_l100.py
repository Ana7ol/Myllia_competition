"""
L1000 Knockdown Signature Fetcher — Multiple API fallbacks
Run this standalone BEFORE main.py to populate l1000_cache_v6.csv

Priority order:
1. SigCom LINCS (maayanlab.cloud) — no key needed, returns z-scores
2. CLUE.io API  — free key needed (register at clue.io/api)
3. Manual CSV   — if you downloaded GSE70138 or GSE92742 from GEO

Usage:
    python fetch_l1000.py
    # produces l1000_cache_v6.csv
    # then run main.py normally
"""

import requests, time, os, json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_FILE   = 'l1000_cache_v6.csv'
CLUE_API_KEY = ''   # optional — get free key at https://clue.io/api
CELL_LINE_PREF = ['MCF7', 'A549', 'PC3', 'VCAP', 'HCC515', 'HA1E']

# ── Load target genes ─────────────────────────────────────────────────────────
def load_targets():
    targets = set()
    if os.path.exists('training_data_means.csv'):
        df = pd.read_csv('training_data_means.csv')
        syms = df[df['pert_symbol'] != 'non-targeting']['pert_symbol'].tolist()
        targets.update(syms)
    if os.path.exists('pert_ids_val.csv'):
        val = pd.read_csv('pert_ids_val.csv')
        targets.update(val['pert'].tolist())
    return sorted(targets)

def load_gene_space():
    df = pd.read_csv('training_data_means.csv')
    return df.columns[1:].tolist()


# ── API 1: SigCom LINCS (maayanlab.cloud/sigcom-lincs) ───────────────────────
def fetch_sigcom(gene, all_genes_set, verbose=True):
    """
    SigCom LINCS REST API — no key needed.
    Returns gene -> log2FC dict or None.
    """
    base = "https://maayanlab.cloud/sigcom-lincs/api/v1"
    try:
        # Step 1: find entity ID for this gene (as KD perturbagen)
        r = requests.get(
            f"{base}/metadata/entities",
            params={"filter": json.dumps({"where": {"meta.gene_symbol": gene,
                                                      "meta.pert_type": "KD"}}),
                    "limit": 20},
            timeout=15)
        if r.status_code != 200:
            return None
        entities = r.json()
        if not entities:
            return None

        # pick best entity
        entity_id = entities[0].get('id')
        if not entity_id:
            return None

        # Step 2: get signatures for this entity
        r2 = requests.get(
            f"{base}/metadata/signatures",
            params={"filter": json.dumps({"where": {"entity_id": entity_id}}),
                    "limit": 50},
            timeout=15)
        if r2.status_code != 200:
            return None
        sigs = r2.json()
        if not sigs:
            return None

        # prefer specific cell lines
        sig_id = None
        for pref in CELL_LINE_PREF:
            for s in sigs:
                if pref.lower() in str(s.get('meta', {}).get('cell_id', '')).lower():
                    sig_id = s['id']
                    break
            if sig_id:
                break
        if not sig_id:
            sig_id = sigs[0]['id']

        # Step 3: get actual values
        r3 = requests.get(
            f"{base}/data/signatures",
            params={"id": sig_id},
            timeout=20)
        if r3.status_code != 200:
            return None

        data = r3.json()
        # data is typically {"genes": [...], "values": [...]}
        genes  = data.get('genes', [])
        values = data.get('values', [])
        if not genes or not values:
            return None

        result = {g: float(v) for g, v in zip(genes, values) if g in all_genes_set}
        if verbose and result:
            print(f"    ✓ SigCom {gene}: {len(result)} genes")
        return result

    except Exception as e:
        if verbose:
            print(f"    ✗ SigCom {gene}: {e}")
        return None


# ── API 2: CLUE.io ────────────────────────────────────────────────────────────
def fetch_clue(gene, all_genes_set, api_key='', verbose=True):
    """
    CLUE.io API — get free key at https://clue.io/api
    Returns gene -> value dict or None.
    """
    if not api_key:
        return None
    base    = "https://api.clue.io/api"
    headers = {"user_key": api_key}
    try:
        # Find pert_id for this gene KD
        r = requests.get(f"{base}/perts",
                          params={"filter": json.dumps({"where": {"pert_iname": gene,
                                                                    "pert_type": "trt_sh"}}),
                                  "limit": 5},
                          headers=headers, timeout=15)
        perts = r.json()
        if not perts:
            return None
        pert_id = perts[0].get('pert_id')
        if not pert_id:
            return None

        # Get level 5 signatures
        r2 = requests.get(f"{base}/sigs",
                           params={"filter": json.dumps({"where": {"pert_id": pert_id,
                                                                     "pert_type": "trt_sh"},
                                                          "fields": ["sig_id", "cell_id",
                                                                      "zscore_norm_median_n_wt"]}),
                                   "limit": 20},
                           headers=headers, timeout=20)
        sigs = r2.json()
        if not sigs:
            return None

        # prefer cell lines
        sig = None
        for pref in CELL_LINE_PREF:
            for s in sigs:
                if pref.lower() in s.get('cell_id', '').lower():
                    sig = s; break
            if sig: break
        if not sig:
            sig = sigs[0]

        zscores = sig.get('zscore_norm_median_n_wt', {})
        if not zscores:
            return None

        result = {g: float(v) for g, v in zscores.items() if g in all_genes_set}
        if verbose and result:
            print(f"    ✓ CLUE {gene}: {len(result)} genes")
        return result

    except Exception as e:
        if verbose:
            print(f"    ✗ CLUE {gene}: {e}")
        return None


# ── API 3: Enrichr L1000 gene sets (lightweight fallback) ────────────────────
def fetch_enrichr_l1000(gene, all_genes_set, verbose=True):
    """
    Enrichr has L1000 KD up/down gene sets.
    Not a full signature but gives directionality for top genes.
    Library: LINCS_L1000_Chem_Pert_Consensus_Sigs (or KD version)
    """
    base = "https://maayanlab.cloud/Enrichr"
    libs = [
        "LINCS_L1000_CRISPR_KO_Consensus_Sigs",
        "LINCS_L1000_Ligand_Perturbations_down",
    ]
    for lib in libs:
        try:
            r = requests.get(f"{base}/geneSetLibrary",
                              params={"mode": "text", "libraryName": lib},
                              timeout=30)
            if r.status_code != 200:
                continue
            # Parse text format: "term\t\tgene1\tgene2\t..."
            up_genes, dn_genes = set(), set()
            for line in r.text.strip().split('\n'):
                parts = line.split('\t')
                if len(parts) < 3: continue
                term = parts[0]
                # Look for lines mentioning our gene
                if gene.upper() in term.upper():
                    genes_in_set = [p.split(',')[0] for p in parts[2:] if p]
                    if 'up' in term.lower():
                        up_genes.update(genes_in_set)
                    elif 'dn' in term.lower() or 'down' in term.lower():
                        dn_genes.update(genes_in_set)

            if up_genes or dn_genes:
                result = {}
                for g in up_genes:
                    if g in all_genes_set:
                        result[g] = 1.0
                for g in dn_genes:
                    if g in all_genes_set:
                        result[g] = -1.0
                if result:
                    if verbose:
                        print(f"    ✓ Enrichr {gene}: {len(result)} genes (binary)")
                    return result
        except Exception as e:
            if verbose:
                print(f"    ✗ Enrichr {gene}/{lib}: {e}")
    return None


# ── Manual GEO fallback ───────────────────────────────────────────────────────
def load_geo_manual(geo_csv_path, target_genes, all_genes):
    """
    If you download L1000 data from GEO manually:
    GSE70138 (Phase II) or GSE92742 (Phase I)
    
    Expected format: rows=genes, cols=perturbation IDs
    or a pre-processed CSV with columns: gene_symbol, <pert_gene1>, <pert_gene2>, ...
    
    Usage:
        sigs = load_geo_manual('GSE70138_processed.csv', target_genes, all_genes)
    """
    if not os.path.exists(geo_csv_path):
        print(f"  [GEO] File not found: {geo_csv_path}")
        return {}
    
    print(f"  [GEO] Loading {geo_csv_path}...")
    df = pd.read_csv(geo_csv_path, index_col=0)
    
    all_genes_set = set(all_genes)
    gene_idx = {g: i for i, g in enumerate(all_genes)}
    result = {}
    
    for gene in target_genes:
        # Try exact match or case-insensitive
        if gene in df.columns:
            col = df[gene]
        elif gene.upper() in [c.upper() for c in df.columns]:
            matches = [c for c in df.columns if c.upper() == gene.upper()]
            col = df[matches[0]]
        else:
            continue
        
        vec = np.zeros(len(all_genes))
        for row_gene, val in col.items():
            if row_gene in gene_idx:
                vec[gene_idx[row_gene]] = float(val)
        
        nonzero = (vec != 0).sum()
        if nonzero >= 10:
            result[gene] = vec
            print(f"    ✓ GEO {gene}: {nonzero} genes")
    
    print(f"  [GEO] Loaded {len(result)}/{len(target_genes)} signatures")
    return result


# ── Main fetch routine ────────────────────────────────────────────────────────
def fetch_all(target_genes, all_genes, clue_key=''):
    if os.path.exists(CACHE_FILE):
        print(f"Cache exists: {CACHE_FILE}")
        ans = input("Rebuild? [y/N]: ").strip().lower()
        if ans != 'y':
            df = pd.read_csv(CACHE_FILE, index_col=0)
            print(f"Loaded {len(df)} cached signatures")
            return

    all_genes_set = set(all_genes)
    gene_idx      = {g: i for i, g in enumerate(all_genes)}
    results       = {}  # gene -> np.array

    print(f"\nFetching signatures for {len(target_genes)} genes...")
    print("APIs: SigCom LINCS → CLUE.io → Enrichr L1000")
    print("-" * 50)

    for i, gene in enumerate(target_genes):
        sig_dict = None

        # Try SigCom first
        sig_dict = fetch_sigcom(gene, all_genes_set, verbose=True)

        # Fallback to CLUE
        if sig_dict is None and clue_key:
            sig_dict = fetch_clue(gene, all_genes_set, api_key=clue_key, verbose=True)

        # Fallback to Enrichr (binary direction only)
        if sig_dict is None:
            sig_dict = fetch_enrichr_l1000(gene, all_genes_set, verbose=True)

        if sig_dict:
            vec = np.zeros(len(all_genes))
            for g, v in sig_dict.items():
                if g in gene_idx:
                    vec[gene_idx[g]] = v
            results[gene] = vec
        else:
            print(f"    – {gene}: no signature found")

        if (i + 1) % 10 == 0:
            print(f"\n  Progress: {i+1}/{len(target_genes)}, "
                  f"found={len(results)}\n")
        time.sleep(0.4)

    # Save cache
    print(f"\nSaving {len(results)} signatures to {CACHE_FILE}...")
    if results:
        rows = {}
        for gene, vec in results.items():
            rows[gene] = {all_genes[j]: vec[j]
                          for j in range(len(all_genes)) if vec[j] != 0}
        pd.DataFrame(rows).T.fillna(0).to_csv(CACHE_FILE)
        print(f"✓ Saved → {CACHE_FILE}")
    else:
        print("✗ No signatures found.")
        print("\n── Manual download instructions ──────────────────────────────")
        print("Option A: Download from GEO (large, ~2GB):")
        print("  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138")
        print("  Download: GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328.gctx.gz")
        print("  Then run: python parse_gctx.py  (see instructions below)")
        print()
        print("Option B: Use CLUE.io API (easier):")
        print("  1. Register free at https://clue.io/api")
        print("  2. Get your API key")
        print("  3. Re-run: python fetch_l1000.py --clue_key YOUR_KEY")
        print()
        print("Option C: Use GEO2R processed signature:")
        print("  Search GEO for your target genes + 'L1000 knockdown'")
        print("  Download as CSV, then: load_geo_manual('your_file.csv', ...)")


if __name__ == '__main__':
    import sys
    clue_key = ''
    for arg in sys.argv[1:]:
        if arg.startswith('--clue_key='):
            clue_key = arg.split('=', 1)[1]

    if not os.path.exists('training_data_means.csv'):
        print("ERROR: Run from your competition directory (training_data_means.csv not found)")
        sys.exit(1)

    print("=" * 60)
    print("  L1000 Signature Fetcher")
    print("=" * 60)

    target_genes = load_targets()
    all_genes    = load_gene_space()
    print(f"Targets: {len(target_genes)} genes")
    print(f"Gene space: {len(all_genes)} genes")

    fetch_all(target_genes, all_genes, clue_key=clue_key)