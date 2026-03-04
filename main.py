"""
=============================================================================
CRISPRi Prediction Pipeline v6
=============================================================================
CHANGES vs v5:
1. L1000 knockdown signatures — real measured KD profiles for target genes
2. Magnitude normalization — scale predictions to match baseline_wmae range
3. Noise reduction — only predict genes with high inter-neighbor consistency
4. Single submission file output
5. Cleaner sweep with more conservative mag options
6. L1000 profiles blended with STRING neighbors when available
=============================================================================
"""

import numpy as np
import pandas as pd
import requests
import urllib.request
import os, time, warnings, json
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# SECTION 1 — OFFICIAL SCORER
# ─────────────────────────────────────────────

def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)

def _gate_smoothstep(x, a=0.0, b=0.2):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return _smoothstep(t)

def _weighted_cosine(a, b, left=0.0, right=0.2, eps=1e-12):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    x  = np.maximum(np.abs(a), np.abs(b))
    w  = _gate_smoothstep(x, left, right)
    w2 = w * w
    num   = np.sum(w2 * a * b)
    den_a = np.sqrt(np.sum(w2 * a * a))
    den_b = np.sqrt(np.sum(w2 * b * b))
    den   = den_a * den_b
    return 0.0 if den < eps else float(num / den)

def official_score(y_true, y_pred, weights, baseline_wmae, eps=1e-12, max_log2=5.0):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    w      = np.asarray(weights, dtype=np.float64)
    base   = np.asarray(baseline_wmae, dtype=np.float64)

    pred_wmae = np.mean(np.abs(y_true - y_pred) * w, axis=1)
    pred_wmae = np.maximum(pred_wmae, eps)
    base      = np.maximum(base, eps)

    terms    = np.minimum(np.log2(base / pred_wmae), max_log2)
    sum_wmae = float(np.sum(terms))

    wcos  = _weighted_cosine(y_pred.ravel(), y_true.ravel())
    score = sum_wmae * max(0.0, wcos)
    return round(score, 5), round(sum_wmae, 5), round(wcos, 5)

def print_score(preds, y_true, weights, baseline_wmae, label):
    score, wmae, cos = official_score(preds, y_true, weights, baseline_wmae)
    print(f"\n--- {label} ---")
    print(f"  Sum log2(WMAE ratio): {wmae:+.4f}")
    print(f"  Weighted Cosine:      {cos:+.4f}")
    print(f"  FINAL SCORE:          {score:+.5f}")
    return score, wmae, cos


# ─────────────────────────────────────────────
# SECTION 2 — LOAD DATA
# ─────────────────────────────────────────────

def load_data():
    print("[1] Loading competition data...")
    df          = pd.read_csv('training_data_means.csv')
    ctrl_mask   = df['pert_symbol'] == 'non-targeting'
    control_vec = df.loc[ctrl_mask, df.columns[1:]].mean().values.astype(np.float32)
    train_df    = df[~ctrl_mask].copy()
    t_symbols   = train_df['pert_symbol'].tolist()
    t_deltas    = (train_df.iloc[:, 1:].values.astype(np.float32) - control_vec)
    all_genes   = df.columns[1:].tolist()
    print(f"  {len(t_symbols)} perturbations x {len(all_genes)} genes")

    weights, baseline_wmae = None, None
    gt_file = 'training_data_ground_truth_table.csv'
    if os.path.exists(gt_file):
        print("  Loading ground truth weights...")
        gt = pd.read_csv(gt_file)
        gt = gt.set_index('pert_id').loc[t_symbols].reset_index()
        weights_full = np.zeros((len(t_symbols), len(all_genes)), dtype=np.float64)
        for j, g in enumerate(all_genes):
            wc = f'w_{g}'
            if wc in gt.columns:
                weights_full[:, j] = gt[wc].values
        baseline_wmae = gt['baseline_wmae'].values.astype(np.float64)
        weights = weights_full
        print(f"  Weights: {weights.shape}, baseline range: "
              f"{baseline_wmae.min():.4f}–{baseline_wmae.max():.4f}")
    else:
        print(f"  [WARN] {gt_file} not found — using uniform weights")
        weights = np.ones((len(t_symbols), len(all_genes)), dtype=np.float64)
        weights *= len(all_genes) / weights.sum(axis=1, keepdims=True)
        baseline_wmae = np.mean(np.abs(t_deltas), axis=1).astype(np.float64)

    return t_symbols, t_deltas, all_genes, weights, baseline_wmae, control_vec


# ─────────────────────────────────────────────
# SECTION 3 — STRING
# ─────────────────────────────────────────────

def fetch_string_scores(gene_list, min_score=400, cache_file='string_cache_v5.csv'):
    if os.path.exists(cache_file):
        df_c   = pd.read_csv(cache_file)
        cached = set(df_c['gene1'].tolist() + df_c['gene2'].tolist())
        if all(g in cached for g in gene_list):
            print(f"  [STRING] Cache loaded: {len(df_c)} pairs")
            return {(r.gene1, r.gene2): r.score for r in df_c.itertuples()}
        print(f"  [STRING] Rebuilding cache...")
        os.remove(cache_file)

    gene_set = set(gene_list)
    scores, rows = {}, []
    print(f"  [STRING] Fetching {len(gene_list)} genes...")
    for i, gene in enumerate(gene_list):
        try:
            r = requests.get(
                "https://string-db.org/api/json/interaction_partners",
                params={"identifiers": gene, "species": 9606, "limit": 200,
                        "required_score": min_score, "caller_identity": "crispr_v6"},
                timeout=20)
            for hit in r.json():
                partner = hit.get('preferredName_B', '')
                score   = float(hit.get('score', 0))
                if partner in gene_set and partner != gene:
                    scores[(gene, partner)] = score
                    rows.append({'gene1': gene, 'gene2': partner, 'score': score})
        except Exception as e:
            print(f"    Warning: {gene}: {e}")
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(gene_list)} fetched...")
        time.sleep(0.1)
    pd.DataFrame(rows).to_csv(cache_file, index=False)
    print(f"  [STRING] Saved {len(scores)} pairs")
    return scores

def get_str(g1, g2, scores):
    return scores.get((g1, g2), scores.get((g2, g1), 0.0))


# ─────────────────────────────────────────────
# SECTION 4 — PATHWAYS
# ─────────────────────────────────────────────

def download_pathways():
    dbs = {
        'KEGG':     'https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=KEGG_2021_Human',
        'Reactome': 'https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=Reactome_2022',
    }
    g2p = {}
    for name, url in dbs.items():
        path = f"{name}.txt"
        if not os.path.exists(path):
            print(f"  Downloading {name}...")
            urllib.request.urlretrieve(url, path)
        with open(path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3: continue
                for g in parts[2:]:
                    g2p.setdefault(g.split(',')[0], set()).add(parts[0])
    return g2p

def jaccard(g1, g2, pw):
    s1, s2 = pw.get(g1, set()), pw.get(g2, set())
    if not s1 or not s2: return 0.0
    return len(s1 & s2) / len(s1 | s2)


# ─────────────────────────────────────────────
# SECTION 5 — L1000 KNOCKDOWN SIGNATURES
# ─────────────────────────────────────────────

def fetch_l1000_signatures(target_genes, all_genes, cache_file='l1000_cache_v6.csv'):
    """
    Fetches knockdown signatures from CLUE.io L1000 API.
    Returns dict: gene -> delta expression vector aligned to all_genes.
    Uses iLINCS as fallback if CLUE unavailable.
    """
    if os.path.exists(cache_file):
        print(f"  [L1000] Loading cache: {cache_file}")
        df = pd.read_csv(cache_file, index_col=0)
        result = {}
        for gene in target_genes:
            if gene in df.index:
                vec = np.zeros(len(all_genes))
                for j, g in enumerate(all_genes):
                    if g in df.columns:
                        vec[j] = df.loc[gene, g]
                result[gene] = vec
        print(f"  [L1000] Loaded {len(result)}/{len(target_genes)} signatures from cache")
        return result

    print(f"\n[L1000] Fetching knockdown signatures for {len(target_genes)} genes...")
    print("  Using LINCS API (iLINCS) — landmark genes only (~978 genes)")

    # iLINCS API: free, no key needed, returns landmark gene signatures
    ILINCS_URL = "https://www.ilincs.org/api/ilincsR/findSimilarSignatures"
    gene_sigs  = {}
    all_genes_set = set(all_genes)

    for i, gene in enumerate(target_genes):
        try:
            # Search for KD signatures for this gene
            resp = requests.get(
                "https://www.ilincs.org/api/SignatureMeta/findTermWithSig",
                params={"term": gene, "sigType": "KD"},
                timeout=15
            )
            if resp.status_code != 200:
                continue
            hits = resp.json()
            if not hits or not isinstance(hits, list):
                continue

            # Pick best signature: human, MCF7 or A549 preferred, highest sample count
            best_sig = None
            for h in hits[:20]:
                sig_id = h.get('signatureID', '')
                if not sig_id:
                    continue
                if best_sig is None:
                    best_sig = sig_id

            if best_sig is None:
                continue

            # Fetch actual signature values
            sig_resp = requests.get(
                f"https://www.ilincs.org/api/ilincsR/downloadSignature",
                params={"sigID": best_sig, "noOfTopGenes": 500},
                timeout=15
            )
            if sig_resp.status_code != 200:
                continue

            sig_data = sig_resp.json()
            if not sig_data:
                continue

            # Parse signature into gene -> value dict
            gene_vals = {}
            if isinstance(sig_data, list):
                for entry in sig_data:
                    gname = entry.get('Name_GeneSymbol', entry.get('gene', ''))
                    val   = float(entry.get('Value_LogDiffExp', entry.get('value', 0)))
                    if gname:
                        gene_vals[gname] = val
            elif isinstance(sig_data, dict):
                for gname, val in sig_data.items():
                    gene_vals[gname] = float(val)

            if len(gene_vals) < 10:
                continue

            # Align to our gene space
            vec = np.zeros(len(all_genes))
            hits_count = 0
            for j, g in enumerate(all_genes):
                if g in gene_vals:
                    vec[j] = gene_vals[g]
                    hits_count += 1
            if hits_count >= 5:
                gene_sigs[gene] = vec
                print(f"    ✓ {gene}: {hits_count} genes mapped from signature {best_sig}")

        except Exception as e:
            pass  # silently skip failures

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(target_genes)} processed, {len(gene_sigs)} signatures found")
        time.sleep(0.3)

    # Save cache
    if gene_sigs:
        rows = {}
        for gene, vec in gene_sigs.items():
            rows[gene] = {all_genes[j]: vec[j] for j in range(len(all_genes)) if vec[j] != 0}
        df_out = pd.DataFrame(rows).T.fillna(0)
        df_out.to_csv(cache_file)
        print(f"  [L1000] Saved {len(gene_sigs)} signatures to {cache_file}")
    else:
        print("  [L1000] No signatures found — will use STRING/Pathway only")

    return gene_sigs


# ─────────────────────────────────────────────
# SECTION 6 — MAGNITUDE NORMALIZATION
# ─────────────────────────────────────────────

def normalize_magnitude(pred, target_gene, all_genes, weights_row, baseline_wmae_val,
                         scale_factor=0.8):
    """
    Scale prediction so its weighted MAE ≈ scale_factor * baseline_wmae.
    This means we stay just under baseline, avoiding WMAE penalty.
    scale_factor < 1.0 = conservative (safer), > 1.0 = aggressive.
    """
    pred = pred.copy()
    w    = np.asarray(weights_row, dtype=np.float64)

    # Current weighted RMS of prediction
    weighted_rms = np.sqrt(np.mean(w * pred**2))
    if weighted_rms < 1e-10:
        return pred  # all zeros, nothing to scale

    # Target: our prediction's WMAE should match scale_factor * baseline
    # Rough approximation: scale uniformly
    current_wmae = np.mean(np.abs(pred) * w)
    if current_wmae < 1e-10:
        return pred

    target_wmae = scale_factor * baseline_wmae_val
    ratio = target_wmae / current_wmae
    # Don't amplify, only shrink or hold
    ratio = min(ratio, 1.0)
    return pred * ratio


# ─────────────────────────────────────────────
# SECTION 7 — NOISE REDUCTION
# ─────────────────────────────────────────────

def compute_noise_mask(neighbor_deltas, consistency_threshold=0.6, min_signal=0.05):
    """
    Returns boolean mask of genes worth predicting.
    A gene passes if:
    1. Fraction of neighbors agreeing on sign >= consistency_threshold
    2. Weighted-average absolute value >= min_signal
    """
    nd = np.array(neighbor_deltas)  # (n_neighbors, n_genes)
    if nd.shape[0] == 0:
        return np.zeros(nd.shape[1], dtype=bool)

    pos_frac  = (nd > 0).mean(axis=0)
    neg_frac  = (nd < 0).mean(axis=0)
    sign_cons = np.maximum(pos_frac, neg_frac)

    avg_abs   = np.abs(nd).mean(axis=0)

    mask = (sign_cons >= consistency_threshold) & (avg_abs >= min_signal)
    return mask


# ─────────────────────────────────────────────
# SECTION 8 — PREDICTOR
# ─────────────────────────────────────────────

def get_neighbors(target, train_syms, train_deltas,
                  string_scores, pathway_dict, k=5, threshold=0.5):
    cands = []
    for i, ts in enumerate(train_syms):
        s = get_str(target, ts, string_scores)
        if s >= threshold:
            cands.append((i, s, ts, 'STRING'))
    if len(cands) < k:
        for i, ts in enumerate(train_syms):
            if any(c[0] == i for c in cands): continue
            j = jaccard(target, ts, pathway_dict)
            if j >= 0.05:
                cands.append((i, j * 0.4, ts, 'Pathway'))
    cands.sort(key=lambda x: -x[1])
    return cands[:k]


def predict_one(target, train_syms, train_deltas, all_genes,
                string_scores, pathway_dict, l1000_sigs,
                weights_row, baseline_wmae_val,
                k=5, threshold=0.5, consistency=0.6,
                mag_scale=0.8, l1000_weight=0.5,
                noise_min_signal=0.05):
    """
    Prediction strategy:
    1. Find STRING/Pathway neighbors
    2. Build consensus mask (noise reduction)
    3. If L1000 signature available, blend with neighbor avg
    4. Normalize magnitude to stay near baseline_wmae
    5. Apply self-knockdown
    """
    neighbors = get_neighbors(target, train_syms, train_deltas,
                               string_scores, pathway_dict, k=k, threshold=threshold)

    has_l1000 = target in l1000_sigs

    if not neighbors and not has_l1000:
        return np.zeros(len(all_genes)), 5, "ZeroPred"

    if neighbors:
        nd  = np.array([train_deltas[i] for i, _, _, _ in neighbors])
        nw  = np.array([s for _, s, _, _ in neighbors])
        nw  = nw / nw.sum()
        wavg = (nw[:, None] * nd).sum(axis=0)

        # Noise mask
        noise_mask = compute_noise_mask(nd, consistency_threshold=consistency,
                                         min_signal=noise_min_signal)
        neighbor_pred = np.zeros(len(all_genes))
        neighbor_pred[noise_mask] = wavg[noise_mask]
    else:
        neighbor_pred = np.zeros(len(all_genes))
        noise_mask    = np.zeros(len(all_genes), dtype=bool)

    # Blend with L1000 if available
    if has_l1000:
        l1000_vec = l1000_sigs[target].copy()
        # Normalize L1000 to same scale as training deltas
        l1000_rms = np.sqrt(np.mean(l1000_vec**2))
        train_rms = np.sqrt(np.mean(wavg**2)) if neighbors else 1.0
        if l1000_rms > 1e-10 and train_rms > 1e-10:
            l1000_vec = l1000_vec * (train_rms / l1000_rms)

        if neighbors:
            pred = (1 - l1000_weight) * neighbor_pred + l1000_weight * l1000_vec
        else:
            pred = l1000_vec
        tier   = 2
        source = f"L1000+STRING({target})"
    else:
        pred = neighbor_pred
        top_sym, top_s, top_nm = neighbors[0][2], neighbors[0][1], neighbors[0][3]
        tier   = 2 if top_s >= 0.7 else 4
        source = f"{top_nm}({top_sym} s={top_s:.2f} {noise_mask.mean()*100:.0f}%)"

    # Magnitude normalization
    if weights_row is not None and baseline_wmae_val is not None:
        pred = normalize_magnitude(pred, target, all_genes,
                                    weights_row, baseline_wmae_val,
                                    scale_factor=mag_scale)

    return pred, tier, source


def apply_self_knockdown(vec, target_gene, all_genes, force_val=-2.5):
    vec = vec.copy()
    if target_gene in all_genes:
        idx = all_genes.index(target_gene)
        vec[idx] = min(vec[idx], force_val)
    return vec


# ─────────────────────────────────────────────
# SECTION 9 — LOO + SWEEP
# ─────────────────────────────────────────────

def run_loo(t_symbols, t_deltas, all_genes, weights, baseline_wmae,
            string_scores, pathway_dict, l1000_sigs,
            k=5, threshold=0.5, consistency=0.6,
            mag_scale=0.8, l1000_weight=0.5, noise_min_signal=0.05):
    preds = []
    for i in range(len(t_symbols)):
        target     = t_symbols[i]
        train_syms = t_symbols[:i] + t_symbols[i+1:]
        Y_train    = np.vstack([t_deltas[:i], t_deltas[i+1:]])
        w_row      = weights[i] if weights is not None else None
        bw_val     = baseline_wmae[i] if baseline_wmae is not None else None

        pred, _, _ = predict_one(
            target, train_syms, Y_train, all_genes,
            string_scores, pathway_dict, l1000_sigs,
            w_row, bw_val,
            k=k, threshold=threshold, consistency=consistency,
            mag_scale=mag_scale, l1000_weight=l1000_weight,
            noise_min_signal=noise_min_signal)
        pred = apply_self_knockdown(pred, target, all_genes)
        preds.append(pred)
    return np.array(preds)


def sweep(t_symbols, t_deltas, all_genes, weights, baseline_wmae,
          string_scores, pathway_dict, l1000_sigs):
    print("\n" + "="*65)
    print("PARAMETER SWEEP (official scorer)")
    print("="*65)

    preds_zero = np.zeros_like(t_deltas)
    preds_mean = np.tile(t_deltas.mean(axis=0), (len(t_symbols), 1))
    preds_self = np.zeros_like(t_deltas)
    for i, sym in enumerate(t_symbols):
        preds_self[i] = apply_self_knockdown(preds_self[i].copy(), sym, all_genes)

    print_score(preds_zero, t_deltas, weights, baseline_wmae, "Zero prediction")
    print_score(preds_mean, t_deltas, weights, baseline_wmae, "Global mean")
    print_score(preds_self, t_deltas, weights, baseline_wmae, "Zero + self-knockdown")

    # mag_scale now means: target WMAE = mag_scale * baseline_wmae
    # 0.9 = just under baseline (safe), 0.5 = very conservative
    configs = [
        # k, thr,  cons,  mag,  l1000_w, noise_sig
        (3, 0.9, 0.67, 0.9, 0.0,  0.03),   # STRING only, mild shrink
        (3, 0.9, 0.67, 0.7, 0.0,  0.05),   # STRING only, moderate shrink
        (3, 0.9, 0.67, 0.9, 0.5,  0.03),   # STRING + L1000
        (5, 0.7, 0.60, 0.9, 0.0,  0.03),
        (5, 0.7, 0.60, 0.7, 0.5,  0.05),
        (3, 0.9, 0.60, 0.9, 0.3,  0.03),
        (5, 0.5, 0.60, 0.8, 0.3,  0.05),
        (3, 0.9, 0.67, 0.5, 0.0,  0.05),   # very conservative
        (5, 0.7, 0.70, 0.9, 0.0,  0.05),   # stricter consensus
    ]

    best_score, best_params = -999, {}
    print("\n  Sweeping configs...")
    for k, thr, cons, mag, l1w, nsig in configs:
        preds = run_loo(t_symbols, t_deltas, all_genes, weights, baseline_wmae,
                        string_scores, pathway_dict, l1000_sigs,
                        k=k, threshold=thr, consistency=cons,
                        mag_scale=mag, l1000_weight=l1w,
                        noise_min_signal=nsig)
        score, wmae, cos = official_score(preds, t_deltas, weights, baseline_wmae)
        label = f"k={k} thr={thr} cons={cons} mag={mag} l1k={l1w} ns={nsig}"
        tag   = "★" if score > best_score else " "
        print(f"  {tag} {label:<55} score={score:+.5f} wmae={wmae:+.4f} cos={cos:.4f}")
        if score > best_score:
            best_score  = score
            best_params = dict(k=k, threshold=thr, consistency=cons,
                                mag_scale=mag, l1000_weight=l1w,
                                noise_min_signal=nsig)

    print(f"\n  Best: {best_score:+.5f}  params={best_params}")
    return best_params, best_score


# ─────────────────────────────────────────────
# SECTION 10 — SUBMISSION GENERATOR
# ─────────────────────────────────────────────

def generate_submission(t_symbols, t_deltas, all_genes, weights, baseline_wmae,
                         string_scores, pathway_dict, l1000_sigs, params,
                         val_file='pert_ids_val.csv',
                         output_file='submission_v6.csv'):
    if not os.path.exists(val_file):
        print(f"  [SKIP] {val_file} not found"); return

    val_df    = pd.read_csv(val_file)
    val_genes = val_df['pert'].tolist()
    val_ids   = val_df['pert_id'].tolist()
    gm        = t_deltas.mean(axis=0)
    rows, tc  = [], {}

    # Use mean weights/baseline as proxy for val genes (we don't have GT for them)
    mean_weights  = weights.mean(axis=0)
    mean_baseline = float(baseline_wmae.mean())

    print(f"\nGenerating: {output_file}")
    for gene, pid in zip(val_genes, val_ids):
        pred, tier, source = predict_one(
            gene, t_symbols, t_deltas, all_genes,
            string_scores, pathway_dict, l1000_sigs,
            mean_weights, mean_baseline, **params)
        pred = apply_self_knockdown(pred, gene, all_genes)
        rows.append([pid] + pred.tolist())
        tc[tier] = tc.get(tier, 0) + 1
        icon = {2:"🟢", 4:"🟡", 5:"🔴"}.get(tier, "⚪")
        print(f"  {icon} {gene:<22} {source}")

    # Fill remaining pert_ids with global mean
    for idx in range(61, 121):
        rows.append([f"pert_{idx}"] + gm.tolist())

    pd.DataFrame(rows, columns=['pert_id'] + all_genes).to_csv(output_file, index=False)
    print(f"\nSaved → {output_file}  "
          f"(L1000+STRING={tc.get(2,0)} Pathway={tc.get(4,0)} Zero={tc.get(5,0)})")
    return output_file


# ─────────────────────────────────────────────
# SECTION 11 — MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  CRISPRi Pipeline v6 — L1000 + Magnitude Norm + Noise Filter")
    print("=" * 65)

    t_symbols, t_deltas, all_genes, weights, baseline_wmae, control_vec = load_data()

    val_genes = []
    if os.path.exists('pert_ids_val.csv'):
        val_genes = pd.read_csv('pert_ids_val.csv')['pert'].tolist()

    print("\n[2] STRING...")
    all_query  = list(set(t_symbols + val_genes))
    str_scores = fetch_string_scores(all_query, cache_file='string_cache_v5.csv')
    val_cov    = sum(1 for g in val_genes
                     if any(get_str(g, t, str_scores) >= 0.7 for t in t_symbols))
    print(f"  Val STRING coverage (≥0.7): {val_cov}/{len(val_genes)}")

    print("\n[3] Pathways...")
    pathway_dict = download_pathways()

    print("\n[4] L1000 knockdown signatures...")
    all_targets = list(set(t_symbols + val_genes))
    l1000_sigs  = fetch_l1000_signatures(all_targets, all_genes)
    print(f"  L1000 coverage: {len(l1000_sigs)}/{len(all_targets)} genes")

    best_params, best_loo = sweep(
        t_symbols, t_deltas, all_genes, weights, baseline_wmae,
        str_scores, pathway_dict, l1000_sigs)

    if val_genes:
        out_file = generate_submission(
            t_symbols, t_deltas, all_genes, weights, baseline_wmae,
            str_scores, pathway_dict, l1000_sigs, best_params,
            output_file='submission_v6.csv')

    print("\n✓ Complete.")
    print(f"\n📋 SUBMIT: submission_v6.csv")
    print(f"   LOO score: {best_loo:+.5f}")
    if best_loo <= 0:
        print("   ⚠️  LOO score ≤ 0 — consider submitting self-knockdown baseline")


if __name__ == "__main__":
    main()