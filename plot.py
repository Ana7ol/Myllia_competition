import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Set plot style for professional EDA
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

def generate_eda_graphs():
    print("Generating biological alignment graphs...")
    
    # ---------------------------------------------------------
    # 1. Metric Analysis: Smoothstep Gating Curve
    # ---------------------------------------------------------
    def smoothstep(x, a=0.0, b=0.2):
        t = np.clip((x - a) / (b - a), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    x_vals = np.linspace(0, 0.4, 500)
    y_vals = smoothstep(x_vals)

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, lw=3, color='crimson')
    plt.axvline(0.2, color='grey', linestyle='--', label='Max Gate (0.2)')
    plt.title("Metric Hack: Wcos Smoothstep Gating", fontsize=14, fontweight='bold')
    plt.xlabel("max(|y_true|, |y_pred|)")
    plt.ylabel("Weight in Cosine Similarity")
    plt.text(0.05, 0.6, "Predictions < 0.2\nare suppressed", fontsize=11)
    plt.text(0.25, 0.6, "Predictions > 0.2\nare heavily weighted!", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig("metric_smoothstep_gate.png", dpi=300)
    print("Saved: metric_smoothstep_gate.png")
    plt.close()

    # ---------------------------------------------------------
    # 2. STRING Network vs. Perturbation Effect Spread
    # ---------------------------------------------------------
    # Simulate a STRING PPI network (Scale-free graph)
    G = nx.barabasi_albert_graph(n=500, m=2, seed=42) 
    
    # Simulate perturbation propagation (Signal diffusing from node 0)
    centralities = nx.closeness_centrality(G)
    shortest_paths = nx.single_source_shortest_path_length(G, source=0)
    
    distances = []
    effects =[]
    for node, dist in shortest_paths.items():
        distances.append(dist)
        # Effect decays exponentially with distance in the PPI network
        base_effect = np.exp(-dist) * np.random.normal(1.0, 0.2)
        effects.append(base_effect)

    plt.figure(figsize=(8, 5))
    sns.boxplot(x=distances, y=effects, hue=distances, palette="viridis", legend=False)
    plt.title("STRING DB: Signal Propagation from Perturbed Gene", fontsize=14, fontweight='bold')
    plt.xlabel("Shortest Path Distance in STRING Network")
    plt.ylabel("Absolute Log2 Fold Change (Effect Size)")
    plt.tight_layout()
    plt.savefig("string_network_propagation.png", dpi=300)
    print("Saved: string_network_propagation.png")
    plt.close()

    # ---------------------------------------------------------
    # 3. L1000 Concordance with scRNA-seq (Myllia dataset)
    # ---------------------------------------------------------
    # Simulate L1000 bulk signatures vs scRNA-seq pseudo-bulk signatures
    np.random.seed(42)
    n_genes = 1000
    
    # True biological signal
    true_signal = np.random.laplace(loc=0, scale=0.1, size=n_genes)
    
    # L1000 adds bulk noise, Myllia adds scRNA dropout noise
    l1000_sig = true_signal + np.random.normal(0, 0.05, n_genes)
    myllia_sig = true_signal + np.random.normal(0, 0.08, n_genes)
    
    # Introduce dropout to Myllia (scRNA-seq sparsity)
    dropout_mask = np.random.rand(n_genes) > 0.3
    myllia_sig = myllia_sig * dropout_mask

    df_concord = pd.DataFrame({'L1000_Bulk': l1000_sig, 'Myllia_scRNA': myllia_sig})

    plt.figure(figsize=(7, 6))
    sns.kdeplot(data=df_concord, x="L1000_Bulk", y="Myllia_scRNA", fill=True, cmap="mako", levels=20)
    plt.axhline(0, color='black', lw=1, ls='--')
    plt.axvline(0, color='black', lw=1, ls='--')
    plt.title("L1000 vs. Myllia scRNA-seq Concordance", fontsize=14, fontweight='bold')
    plt.xlabel("L1000 Log2 Fold Change (Bulk)")
    plt.ylabel("Myllia Log2 Fold Change (scRNA-seq)")
    
    # Calculate correlation
    corr = np.corrcoef(l1000_sig, myllia_sig)[0,1]
    plt.text(-0.3, 0.3, f"Pearson R = {corr:.2f}\nNotice dropout on Y=0", fontsize=12, color='darkred')
    
    plt.tight_layout()
    plt.savefig("l1000_vs_myllia_concordance.png", dpi=300)
    print("Saved: l1000_vs_myllia_concordance.png")
    plt.close()

if __name__ == "__main__":
    generate_eda_graphs()
