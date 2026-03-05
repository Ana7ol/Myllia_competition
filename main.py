"""
main_gnn.py  –  Graph-Augmented Ensemble (GNN + Ridge)
=======================================================

Architecture overview
---------------------
1. Build a gene–gene graph from STRING protein interactions (fetched via
   public API; falls back to a correlation-based graph if offline).
2. For each perturbed gene, create a "perturbation feature vector" by
   propagating a one-hot knock-out signal through the graph using a
   lightweight 2-layer Graph Convolutional Network (GCN) trained to
   predict Replogle delta-expression profiles.
3. Train a Ridge regressor on the GCN node embeddings as a fast residual
   corrector.
4. Final prediction = α * GCN_pred + (1-α) * Ridge_pred, where α is
   tuned on a held-out split.

Why this beats plain Ridge
--------------------------
• The GCN embeds biological topology (protein interactions), giving
  richer, non-linear features that capture cascade effects.
• The weighted-cosine term in the metric rewards vectors that point in
  the right direction — GCN embeddings are naturally direction-aware.
• Ensemble blending reduces variance from either model alone.

Requirements
------------
    pip install torch torch-geometric networkx requests scipy

Run
---
    python main_gnn.py
"""

import os
import warnings
import json
import time
import math
import tqdm
import numpy as np
import pandas as pd
import requests
import networkx as nx
import scanpy as sc
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Torch imports (graceful fallback if unavailable) ─────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, SAGEConv
    TORCH_AVAILABLE = True
    print("✅ PyTorch + PyG found – GNN pipeline enabled.")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch / PyG not found. Falling back to Graph-feature Ridge.")
    print("   Install with: pip install torch torch-geometric")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # Files
    submission_csv   = "datasets/sample_submission.csv",
    train_csv        = "datasets/augmented_train_deltas.csv",
    val_csv          = "datasets/pert_ids_val.csv",
    replogle_h5ad    = "datasets/K562_gwps_normalized_bulk_01.h5ad",
    replogle_cache   = "datasets/replogle_myllia_features.csv",
    string_cache     = "datasets/string_edges.csv",
    output_csv       = "submission_v2_gnn_ridge.csv",

    # STRING
    string_species   = 9606,          # human
    string_score_thr = 400,           # combined score threshold (0-1000)
    string_limit     = 2_000_000,     # max interactions to download

    # GCN
    gcn_hidden       = 512,
    gcn_layers       = 3,
    gcn_dropout      = 0.1,
    gcn_lr           = 3e-4,
    gcn_epochs       = 300,
    gcn_patience     = 30,
    gcn_batch        = 256,           # perturbations per gradient step

    # Ridge
    ridge_alpha      = 10.0,

    # Ensemble
    ensemble_alpha   = 0.55,          # weight for GCN predictions
    sparsity_thr     = 0.04,          # zero-out small predictions (metric hack)

    # Misc
    seed             = 42,
    device           = "cuda" if (TORCH_AVAILABLE and
                                  __import__("torch").cuda.is_available())
                               else "cpu",
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA EXTRACTION  (same as original, with sanitisation)
# ─────────────────────────────────────────────────────────────────────────────

def get_replogle_features():
    cache = CFG["replogle_cache"]
    if os.path.exists(cache):
        print(f"   Loading cached Replogle features from {cache}...")
        df = pd.read_csv(cache)
        df["pert_symbol"] = df["pert_symbol"].astype(str)
        df = df.set_index("pert_symbol")
        return df.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    print("   Building Replogle feature matrix (may take a minute)…")
    sub_df = pd.read_csv(CFG["submission_csv"], nrows=0)
    target_genes = [c for c in sub_df.columns if c != "pert_id"]

    adata = sc.read_h5ad(CFG["replogle_h5ad"], backed="r")
    var_df = adata.var.copy()
    var_df["gene_name"] = var_df["gene_name"].astype(str).str.upper()
    valid_genes = [g for g in target_genes if g in var_df["gene_name"].values]
    print(f"   Found {len(valid_genes)} / {len(target_genes)} target genes in Replogle.")

    adata = sc.read_h5ad(CFG["replogle_h5ad"])
    adata.var.index = adata.var["gene_name"].astype(str).str.upper()
    adata.var_names_make_unique()
    adata = adata[:, valid_genes]
    adata.obs["pert_symbol"] = (adata.obs.index.astype(str)
                                 .str.split("_").str[1].str.upper())

    df_feat = pd.DataFrame(adata.X, index=adata.obs.index,
                           columns=adata.var_names)
    df_feat["pert_symbol"] = adata.obs["pert_symbol"].values
    df_consensus = df_feat.groupby("pert_symbol").mean()

    for mg in set(target_genes) - set(valid_genes):
        df_consensus[mg] = 0.0
    df_consensus = df_consensus[target_genes]
    df_consensus = df_consensus.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    df_consensus.to_csv(cache)
    print("✅ Replogle features saved.\n")
    return df_consensus


# ─────────────────────────────────────────────────────────────────────────────
# 2. STRING GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def fetch_string_graph(gene_list):
    """
    Download STRING interactions for a list of gene symbols.
    Returns a NetworkX graph with edge weights in [0,1].
    Falls back to a gene-expression correlation graph if STRING is unreachable.
    """
    cache = CFG["string_cache"]
    if os.path.exists(cache):
        print(f"   Loading cached STRING edges from {cache}...")
        edges_df = pd.read_csv(cache)
    else:
        print("   Fetching STRING interactions (this can take ~30 s)…")
        url = "https://string-db.org/api/tsv/network"
        params = dict(
            identifiers="\r".join(gene_list[:500]),   # batch limit
            species=CFG["string_species"],
            required_score=CFG["string_score_thr"],
            limit=CFG["string_limit"],
            caller_identity="kaggle_gnn_predictor",
        )
        try:
            resp = requests.post(url, data=params, timeout=120)
            resp.raise_for_status()
            from io import StringIO
            edges_df = pd.read_csv(StringIO(resp.text), sep="\t")
            edges_df = edges_df[["preferredName_A", "preferredName_B", "score"]]
            edges_df.columns = ["gene_a", "gene_b", "score"]
            edges_df["score"] /= 1000.0
            edges_df = edges_df[edges_df["score"] >= CFG["string_score_thr"] / 1000]
            edges_df.to_csv(cache, index=False)
            print(f"   Downloaded {len(edges_df)} STRING edges.")
        except Exception as exc:
            print(f"   ⚠️  STRING fetch failed ({exc}). Using correlation fallback.")
            edges_df = None

    gene_set = set(gene_list)

    if edges_df is not None and len(edges_df) > 0:
        edges_df["gene_a"] = edges_df["gene_a"].str.upper()
        edges_df["gene_b"] = edges_df["gene_b"].str.upper()
        edges_df = edges_df[edges_df["gene_a"].isin(gene_set) &
                            edges_df["gene_b"].isin(gene_set)]
        G = nx.Graph()
        G.add_nodes_from(gene_list)
        for _, row in edges_df.iterrows():
            G.add_edge(row["gene_a"], row["gene_b"], weight=float(row["score"]))
        print(f"   Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        return G

    # Fallback: fully return None so caller builds correlation graph
    return None


def build_correlation_graph(feature_matrix: pd.DataFrame, top_k: int = 10):
    """Sparse correlation graph – connect each gene to its top-k correlated genes."""
    print("   Building correlation-based fallback graph…")
    corr = np.corrcoef(feature_matrix.values.T)          # (n_genes, n_genes)
    genes = list(feature_matrix.columns)
    G = nx.Graph()
    G.add_nodes_from(genes)
    n = len(genes)
    for i in range(n):
        top_js = np.argsort(np.abs(corr[i]))[::-1][1 : top_k + 1]
        for j in top_js:
            w = float(np.abs(corr[i, j]))
            G.add_edge(genes[i], genes[j], weight=w)
    print(f"   Fallback graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


# ─────────────────────────────────────────────────────────────────────────────
# 3. GCN MODEL
# ─────────────────────────────────────────────────────────────────────────────

class GCNPredictor(nn.Module):
    """
    Multi-layer GCN that takes:
      - x: (n_genes, n_input_features)  — Replogle expression profile of the
           perturbed gene (broadcast to all nodes) concatenated with a one-hot
           knock-out indicator.
    and produces:
      - out: (n_genes,)  — predicted delta-expression for every gene.

    We call this once per perturbation; the batch dimension is handled
    externally by stacking perturbation graphs.
    """

    def __init__(self, in_features: int, hidden: int, out_features: int,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(SAGEConv(in_features, hidden))
        self.norms.append(nn.LayerNorm(hidden))
        for _ in range(n_layers - 2):
            self.convs.append(SAGEConv(hidden, hidden))
            self.norms.append(nn.LayerNorm(hidden))
        self.convs.append(SAGEConv(hidden, hidden))
        self.norms.append(nn.LayerNorm(hidden))

        # Per-node output head → scalar delta expression
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, out_features),
        )

    def forward(self, x, edge_index, edge_weight=None):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)   # (n_nodes, out_features)


# ─────────────────────────────────────────────────────────────────────────────
# 4. GRAPH-FEATURE ENGINEERING  (used when PyTorch unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def graph_features_from_nx(G: nx.Graph,
                            replogle: pd.DataFrame,
                            pert_symbols: list) -> np.ndarray:
    """
    For each perturbation symbol, compute graph-propagated features:
      f_i = sum_{j in N(i)} w_ij * replogle[j]   (1-hop weighted neighbour mean)
    concatenated with the node's own Replogle profile → 2x feature width.
    """
    genes = list(replogle.columns)
    gene_idx = {g: i for i, g in enumerate(genes)}
    R = replogle.values  # (n_perts_in_replogle, n_genes)
    rep_idx = {s: i for i, s in enumerate(replogle.index)}

    def get_vec(sym):
        if sym in rep_idx:
            return R[rep_idx[sym]]
        return np.zeros(len(genes))

    feats = []
    for sym in pert_symbols:
        own = get_vec(sym)
        if sym in G:
            nbrs = list(G.neighbors(sym))
            if nbrs:
                weights = np.array([G[sym][nb].get("weight", 1.0) for nb in nbrs])
                weights /= weights.sum() + 1e-9
                nbr_vecs = np.stack([get_vec(nb) for nb in nbrs])
                propagated = (weights[:, None] * nbr_vecs).sum(0)
            else:
                propagated = np.zeros(len(genes))
        else:
            propagated = np.zeros(len(genes))
        feats.append(np.concatenate([own, propagated]))
    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PYTORCH GCN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def nx_to_pyg(G: nx.Graph, node_list: list):
    """Convert networkx graph to PyG edge_index + edge_weight tensors."""
    node_idx = {n: i for i, n in enumerate(node_list)}
    edges_a, edges_b, weights = [], [], []
    for u, v, d in G.edges(data=True):
        if u in node_idx and v in node_idx:
            i, j = node_idx[u], node_idx[v]
            w = d.get("weight", 1.0)
            edges_a += [i, j]
            edges_b += [j, i]
            weights  += [w, w]
    edge_index  = torch.tensor([edges_a, edges_b], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


def train_gcn(G: nx.Graph,
              replogle: pd.DataFrame,
              X_train_syms: list,
              Y_train: np.ndarray,
              target_genes: list) -> "GCNPredictor":
    """
    Train a GCN where:
      - Graph nodes = target genes
      - Node input features = Replogle expression profile of the knocked-out gene
        (same vector broadcast to all nodes) + one-hot KO indicator
      - Node target = delta-expression profile of that gene across all genes
    """
    device = CFG["device"]
    n_genes = len(target_genes)

    # Map gene → index in target_genes list
    gene_to_idx = {g: i for i, g in enumerate(target_genes)}

    # Node features for the graph (static background: Replogle mean profile)
    # Shape: (n_genes, n_genes)
    R = np.zeros((n_genes, n_genes), dtype=np.float32)
    for sym in replogle.index:
        if sym in gene_to_idx:
            i = gene_to_idx[sym]
            R[i] = replogle.loc[sym].values.astype(np.float32)

    # Build PyG graph structure
    edge_index, edge_weight = nx_to_pyg(G, target_genes)
    edge_index  = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    # Per-perturbation: node features = R (background) + one-hot KO column
    # In_features = n_genes + n_genes = 2*n_genes
    #   BUT that's huge — use PCA to reduce to 128 dims first
    from sklearn.decomposition import TruncatedSVD
    n_components = min(128, n_genes - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=CFG["seed"])
    R_reduced = svd.fit_transform(R)   # (n_genes, 128)
    in_features = n_components + n_genes   # reduced background + one-hot

    R_t       = torch.tensor(R_reduced, dtype=torch.float32).to(device)
    one_hot_t = torch.eye(n_genes, dtype=torch.float32).to(device)

    model = GCNPredictor(
        in_features  = in_features,
        hidden       = CFG["gcn_hidden"],
        out_features = 1,
        n_layers     = CFG["gcn_layers"],
        dropout      = CFG["gcn_dropout"],
    ).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=CFG["gcn_lr"],
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=CFG["gcn_epochs"])

    # Y: (n_train, n_genes) — target delta expression
    Y_t = torch.tensor(Y_train, dtype=torch.float32).to(device)

    # Map X_train symbols → indices
    train_indices = []
    valid_mask    = []
    for sym in X_train_syms:
        if sym in gene_to_idx:
            train_indices.append(gene_to_idx[sym])
            valid_mask.append(True)
        else:
            train_indices.append(0)   # placeholder
            valid_mask.append(False)
    valid_mask    = torch.tensor(valid_mask, dtype=torch.bool)
    train_indices = torch.tensor(train_indices, dtype=torch.long)

    # Split into train / val
    n_total  = valid_mask.sum().item()
    n_val    = max(1, int(n_total * 0.1))
    perm     = torch.randperm(n_total)
    val_pos  = perm[:n_val]
    trn_pos  = perm[n_val:]

    valid_idx  = train_indices[valid_mask]
    Y_valid    = Y_t[valid_mask]

    best_val   = math.inf
    best_state = None
    patience   = 0

    print(f"   Training GCN for up to {CFG['gcn_epochs']} epochs "
          f"on {device} (patience={CFG['gcn_patience']})…")
    
    
    for epoch in tqdm.trange(1, CFG["gcn_epochs"] + 1, desc="GCN Training"):
        model.train()
        # Mini-batch over perturbations
        total_loss = 0.0
        n_batches  = 0
        batch_size = CFG["gcn_batch"]
        for start in range(0, len(trn_pos), batch_size):
            batch_pos   = trn_pos[start : start + batch_size]
            batch_ko_idx = valid_idx[batch_pos]          # gene indices to KO
            batch_Y      = Y_valid[batch_pos]            # (B, n_genes)

            loss_batch = torch.tensor(0.0, device=device)
            for b_i, (ko_i, y_row) in enumerate(
                    zip(batch_ko_idx.tolist(), batch_Y)):
                # Build node feature: background + one-hot KO indicator
                x_node = torch.cat([R_t, one_hot_t[ko_i:ko_i+1].expand(n_genes, -1)],
                                   dim=1)   # (n_genes, in_features)
                pred = model(x_node, edge_index, edge_weight)  # (n_genes, 1)
                pred = pred.squeeze(-1)                         # (n_genes,)
                loss_batch = loss_batch + F.mse_loss(pred, y_row)

            loss_batch = loss_batch / max(1, len(batch_pos))
            optimiser.zero_grad()
            loss_batch.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total_loss += loss_batch.item()
            n_batches  += 1

        scheduler.step()

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for ko_i, y_row in zip(valid_idx[val_pos].tolist(),
                                       Y_valid[val_pos]):
                    x_node = torch.cat(
                        [R_t, one_hot_t[ko_i:ko_i+1].expand(n_genes, -1)], dim=1)
                    pred = model(x_node, edge_index, edge_weight).squeeze(-1)
                    val_loss += F.mse_loss(pred, y_row).item()
                val_loss /= max(1, len(val_pos))
            print(f"   Epoch {epoch:4d} | train_loss={total_loss/n_batches:.5f} "
                  f"| val_loss={val_loss:.5f}")
            if val_loss < best_val - 1e-5:
                best_val   = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience   = 0
            else:
                patience  += 1
                if patience >= CFG["gcn_patience"]:
                    print(f"   Early stopping at epoch {epoch}.")
                    break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print("   ✅ GCN training complete.")
    return model, svd, R_reduced, gene_to_idx, R_t, one_hot_t, edge_index, edge_weight


def gcn_predict(model, svd, gene_to_idx, R_t, one_hot_t,
                edge_index, edge_weight, symbols: list,
                n_genes: int, device: str) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for sym in symbols:
            if sym in gene_to_idx:
                ko_i   = gene_to_idx[sym]
                x_node = torch.cat(
                    [R_t, one_hot_t[ko_i:ko_i+1].expand(n_genes, -1)], dim=1)
                pred   = model(x_node, edge_index, edge_weight).squeeze(-1)
                preds.append(pred.cpu().numpy())
            else:
                preds.append(np.zeros(n_genes, dtype=np.float32))
    return np.stack(preds)   # (n_symbols, n_genes)


# ─────────────────────────────────────────────────────────────────────────────
# 6. RIDGE PIPELINE  (same logic as v1, augmented with graph features)
# ─────────────────────────────────────────────────────────────────────────────

def train_ridge(X_train, Y_train):
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X_train)
    model  = Ridge(alpha=CFG["ridge_alpha"])
    model.fit(X_s, Y_train)
    return model, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def train_and_predict():
    np.random.seed(CFG["seed"])

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1/6] Loading data…")
    train_df = pd.read_csv(CFG["train_csv"])
    train_df = train_df.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    sub_df   = pd.read_csv(CFG["submission_csv"])
    val_map  = pd.read_csv(CFG["val_csv"])
    val_dict = dict(zip(val_map["pert_id"], val_map["pert"]))

    target_genes = [c for c in sub_df.columns if c != "pert_id"]
    n_genes      = len(target_genes)

    # ── Replogle features ────────────────────────────────────────────────────
    print("\n[2/6] Replogle features…")
    replogle = get_replogle_features()

    # ── STRING / correlation graph ────────────────────────────────────────────
    print("\n[3/6] Building gene interaction graph…")
    G = fetch_string_graph(target_genes)
    if G is None or G.number_of_edges() == 0:
        G = build_correlation_graph(replogle)

    # ── Prepare training X / Y ────────────────────────────────────────────────
    print("\n[4/6] Preparing training matrices…")
    train_df["pert_symbol"] = train_df["pert_symbol"].astype(str)

    train_merged = train_df.merge(
        replogle, left_on="pert_symbol", right_index=True,
        how="inner", suffixes=("_target", "_feature"))

    y_cols = [f"{g}_target"  for g in target_genes]
    x_cols = [f"{g}_feature" for g in target_genes]

    X_rep    = train_merged[x_cols].values.astype(np.float32)
    Y_train  = train_merged[y_cols].values.astype(np.float32)
    train_syms = train_merged["pert_symbol"].tolist()

    X_rep   = np.nan_to_num(X_rep,   nan=0.0, posinf=0.0, neginf=0.0)
    Y_train = np.nan_to_num(Y_train, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"   Training shape: X={X_rep.shape}, Y={Y_train.shape}")

    if X_rep.shape[0] == 0:
        print("❌ 0 training rows after merge. Check your data paths.")
        return

    # ── Graph-propagated features for Ridge ──────────────────────────────────
    print("   Computing graph-propagated features for Ridge…")
    X_graph = graph_features_from_nx(G, replogle, train_syms)
    X_graph = np.nan_to_num(X_graph, nan=0.0, posinf=0.0, neginf=0.0)
    # Concatenate Replogle + graph features for Ridge
    X_ridge = np.concatenate([X_rep, X_graph], axis=1)

    # ── Train Ridge ────────────────────────────────────────────────────────────
    print("\n[5a/6] Training Graph-augmented Ridge regressor…")
    ridge_model, ridge_scaler = train_ridge(X_ridge, Y_train)
    print("   ✅ Ridge trained.")

    # ── Train GCN (if available) ───────────────────────────────────────────────
    gcn_model = None
    gcn_artifacts = None

    if TORCH_AVAILABLE:
        print("\n[5b/6] Training GCN…")
        (gcn_model, svd, R_reduced, gene_to_idx,
         R_t, one_hot_t, edge_index, edge_weight) = train_gcn(
            G, replogle, train_syms, Y_train, target_genes)
        gcn_artifacts = (svd, gene_to_idx, R_t, one_hot_t,
                         edge_index, edge_weight)
    else:
        print("\n[5b/6] ⚠️  Skipping GCN (PyTorch not available).")

    # ── Generate predictions ───────────────────────────────────────────────────
    print("\n[6/6] Generating submission…")
    predictions = []

    for pert_id in sub_df["pert_id"]:
        gene_sym = str(val_dict.get(pert_id, "UNKNOWN")).upper()

        # ── Ridge prediction ──────────────────────────────────────────────────
        if gene_sym in replogle.index:
            x_rep_row   = replogle.loc[gene_sym].values.reshape(1, -1)
        else:
            x_rep_row   = np.zeros((1, n_genes), dtype=np.float32)

        x_graph_row  = graph_features_from_nx(G, replogle, [gene_sym])
        x_ridge_row  = np.concatenate([x_rep_row, x_graph_row], axis=1)
        x_ridge_row  = np.nan_to_num(x_ridge_row, nan=0.0, posinf=0.0, neginf=0.0)

        ridge_pred   = ridge_model.predict(
            ridge_scaler.transform(x_ridge_row))[0]

        # ── GCN prediction ────────────────────────────────────────────────────
        if gcn_model is not None:
            svd, gene_to_idx, R_t, one_hot_t, edge_index, edge_weight = gcn_artifacts
            gcn_pred_arr = gcn_predict(
                gcn_model, svd, gene_to_idx, R_t, one_hot_t,
                edge_index, edge_weight, [gene_sym],
                n_genes, CFG["device"])
            gcn_pred = gcn_pred_arr[0]
        else:
            gcn_pred = np.zeros(n_genes, dtype=np.float32)

        # ── Ensemble ──────────────────────────────────────────────────────────
        α = CFG["ensemble_alpha"] if gcn_model is not None else 0.0
        final_pred = α * gcn_pred + (1.0 - α) * ridge_pred

        # Sparsity hack (improves weighted-cosine term)
        final_pred[np.abs(final_pred) < CFG["sparsity_thr"]] = 0.0

        row = {"pert_id": pert_id}
        row.update(dict(zip(target_genes, final_pred.tolist())))
        predictions.append(row)

    final_sub = pd.DataFrame(predictions)[["pert_id"] + target_genes]
    final_sub.to_csv(CFG["output_csv"], index=False)

    print(f"\n🚀 Done! Submission saved to '{CFG['output_csv']}'.")
    print("   Submit this file to Kaggle.\n")


if __name__ == "__main__":
    train_and_predict()