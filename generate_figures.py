"""Run this script to generate all data and figures for the LaTeX paper."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from src.bio_parser import (
    generate_gcn_normal, generate_gcn_cancer,
    save_matrix, matrix_to_networkx,
)
from src.graph_metrics import (
    spectral_distance, frobenius_distance,
    isomorphism_metric, epsilon_isomorphism_check,
    gromov_wasserstein_approx,
    differential_subgraph, module_isomorphism_scores,
)
from src.visualization import (
    plot_heatmap, plot_difference_heatmap,
    plot_eigenvalue_spectrum, plot_network,
    plot_metric_comparison,
)

DATA_RAW = Path("data/raw")
DATA_PROC = Path("data/processed")
FIGURES  = Path("tex_paper/figures")
for d in [DATA_RAW, DATA_PROC, FIGURES]:
    d.mkdir(parents=True, exist_ok=True)

# ── 1. Generate data ─────────────────────────────────────────────────────────
print("Generating GCN data...")
W_normal, modules, gene_names = generate_gcn_normal(
    n_genes=50, n_modules=5, intra_corr=0.75, inter_corr=0.05, noise=0.05, seed=42
)
W_cancer = generate_gcn_cancer(
    W_normal, modules, disrupted_modules=[4], disruption_strength=0.6, noise=0.08, seed=99
)
save_matrix(W_normal, DATA_RAW / "gcn_normal.csv")
save_matrix(W_cancer, DATA_RAW / "gcn_cancer.csv")
print("  saved data/raw/gcn_normal.csv and gcn_cancer.csv")

# ── 2. Heatmaps ──────────────────────────────────────────────────────────────
print("Plotting heatmaps...")
plot_heatmap(W_normal, title="$G_1$: Normal tissue — Gene Co-expression Network",
             save_path=FIGURES / "heatmap_normal.pdf")
plt.close("all")
plot_heatmap(W_cancer, title="$G_2$: Cancer tissue — Gene Co-expression Network",
             save_path=FIGURES / "heatmap_cancer.pdf")
plt.close("all")

# ── 3. Weight distributions ──────────────────────────────────────────────────
print("Plotting weight distributions...")
mask = ~np.eye(50, dtype=bool)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, W, label, color in [
    (axes[0], W_normal, "Normal tissue", "#2ca02c"),
    (axes[1], W_cancer, "Cancer tissue", "#d62728"),
]:
    flat = W[mask]
    ax.hist(flat, bins=50, color=color, alpha=0.75, edgecolor="white")
    ax.axvline(flat.mean(), color="black", linestyle="--", label=f"Mean={flat.mean():.3f}")
    ax.set_title(f"Weight distribution: {label}", fontweight="bold")
    ax.set_xlabel("Pearson correlation"); ax.set_ylabel("Count")
    ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "weight_distributions.pdf", dpi=150, bbox_inches="tight")
plt.close("all")

# ── 4. Differential heatmap ──────────────────────────────────────────────────
print("Plotting differential analysis...")
plot_difference_heatmap(
    W_normal, W_cancer, label1="Normal", label2="Cancer",
    threshold=0.35, modules=modules,
    save_path=FIGURES / "differential_analysis.pdf",
)
plt.close("all")

# ── 5. Eigenvalue spectra ────────────────────────────────────────────────────
print("Plotting eigenvalue spectra...")
plot_eigenvalue_spectrum(
    W_normal, W_cancer, label1="Normal", label2="Cancer",
    save_path=FIGURES / "eigenvalue_spectrum.pdf",
)
plt.close("all")

# ── 6. Metrics ───────────────────────────────────────────────────────────────
print("Computing isomorphism metrics...")
d_spec = spectral_distance(W_normal, W_cancer)
d_frob = frobenius_distance(W_normal, W_cancer)
d_iso,  best_perm = isomorphism_metric(W_normal, W_cancer)
d_gw   = gromov_wasserstein_approx(W_normal, W_cancer, n_iter=50)

print(f"  d_spec  = {d_spec:.4f}  (LB on d_iso: {d_spec/np.sqrt(2):.4f})")
print(f"  d_frob  = {d_frob:.4f}")
print(f"  d_iso   = {d_iso:.4f}  (Hungarian approx)")
print(f"  d_GW    = {d_gw:.4f}")

# ── 7. Epsilon-isomorphism curve ─────────────────────────────────────────────
print("Plotting epsilon-isomorphism...")
epsilons = np.linspace(0.0, 1.5, 60)
results  = [epsilon_isomorphism_check(W_normal, W_cancer, eps) for eps in epsilons]
eps_star = next((e for e, r in zip(epsilons, results) if r["is_epsilon_isomorphic"]), None)
spec_lb  = results[0]["spectral_lb"]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epsilons, [r["epsilon_achieved"] for r in results], "b-", lw=1.5, label="$d_{\\mathrm{iso}}$ achieved")
ax.axhline(eps_star,  color="red",   linestyle="--", label=f"$\\varepsilon^*={eps_star:.3f}$")
ax.axhline(spec_lb,   color="green", linestyle=":",  label=f"Spectral LB$={spec_lb:.3f}$")
ax.fill_between(epsilons,
    epsilons,
    [r["epsilon_achieved"] for r in results],
    where=[e >= r["epsilon_achieved"] for e, r in zip(epsilons, results)],
    alpha=0.12, color="green", label="$\\varepsilon$-isomorphic region")
ax.set_xlabel("$\\varepsilon$ (tolerance)"); ax.set_ylabel("Value")
ax.set_title("$\\varepsilon$-Isomorphism: Normal vs Cancer GCN", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "epsilon_isomorphism.pdf", dpi=150, bbox_inches="tight")
plt.close("all")

# ── 8. Per-module scores ─────────────────────────────────────────────────────
print("Computing per-module isomorphism scores...")
mod_scores = module_isomorphism_scores(W_normal, W_cancer, modules)
print(f"  {'Module':>8}  {'d_iso':>8}  {'d_spec':>8}  {'Status':>10}")
for k, s in enumerate(mod_scores):
    status = "DISRUPTED" if s["epsilon_achieved"] > 0.4 else "intact"
    print(f"  {k+1:>8}  {s['epsilon_achieved']:>8.4f}  {s['spectral_distance']:>8.4f}  {status:>10}")

x = np.arange(len(mod_scores))
d_iso_vals  = [s["epsilon_achieved"]              for s in mod_scores]
d_spec_vals = [s["spectral_distance"]/np.sqrt(2) for s in mod_scores]
colors = ["#d62728" if v > 0.4 else "#2ca02c" for v in d_iso_vals]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - 0.2, d_iso_vals,  0.4, label="$d_{\\mathrm{iso}}$ (Hungarian)", color=colors, alpha=0.85)
ax.bar(x + 0.2, d_spec_vals, 0.4, label="Spectral LB", color="steelblue", alpha=0.7)
ax.axhline(0.4, color="black", linestyle="--", lw=1, label="Disruption threshold")
ax.set_xticks(x); ax.set_xticklabels([f"$M_{k+1}$" for k in range(len(mod_scores))])
ax.set_ylabel("Distance")
ax.set_title("Per-module isomorphism: Normal vs Cancer", fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "module_isomorphism_scores.pdf", dpi=150, bbox_inches="tight")
plt.close("all")

# ── 9. Metric comparison ─────────────────────────────────────────────────────
print("Plotting metric comparison...")
W_noisy = W_normal + np.random.default_rng(7).normal(0, 0.03, W_normal.shape)
W_noisy = (W_noisy + W_noisy.T) / 2; np.fill_diagonal(W_noisy, 1.0)
d_iso2, _ = isomorphism_metric(W_normal, W_noisy)
metrics_summary = {
    "Normal vs Cancer": {
        "$d_{\\mathrm{Frob}}$": d_frob,
        "$d_{\\mathrm{spec}}$": d_spec,
        "$d_{\\mathrm{iso}}$": d_iso,
        "$d_{\\mathrm{GW}}$": d_gw,
    },
    "Normal vs Noise": {
        "$d_{\\mathrm{Frob}}$": frobenius_distance(W_normal, W_noisy),
        "$d_{\\mathrm{spec}}$": spectral_distance(W_normal, W_noisy),
        "$d_{\\mathrm{iso}}$": d_iso2,
        "$d_{\\mathrm{GW}}$": gromov_wasserstein_approx(W_normal, W_noisy, n_iter=30),
    },
}
plot_metric_comparison(metrics_summary, save_path=FIGURES / "metric_comparison.pdf")
plt.close("all")

# ── 10. Network plots ────────────────────────────────────────────────────────
print("Plotting networks...")
G_normal = matrix_to_networkx(W_normal, gene_names[:20], threshold=0.4)
sub_modules = [[i for i in m if i < 20] for m in modules]
plot_network(G_normal,
    title="$G_1$: Normal tissue GCN (top 20 genes)",
    modules=sub_modules, node_labels=gene_names[:20],
    save_path=FIGURES / "network_normal.pdf")
plt.close("all")

G_cancer = matrix_to_networkx(W_cancer, gene_names[:20], threshold=0.4)
plot_network(G_cancer,
    title="$G_2$: Cancer tissue GCN (top 20 genes)",
    modules=sub_modules, node_labels=gene_names[:20],
    save_path=FIGURES / "network_cancer.pdf")
plt.close("all")

# ── 11. Module disruption grid ───────────────────────────────────────────────
print("Plotting module disruption grid...")
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for k, (mod, ax) in enumerate(zip(modules, axes)):
    diff = np.abs(W_cancer[np.ix_(mod, mod)] - W_normal[np.ix_(mod, mod)])
    im = ax.imshow(diff, cmap="Oranges", vmin=0, vmax=1)
    vals = diff[diff > 0]
    mean_str = f"{vals.mean():.2f}" if vals.size else "0.00"
    ax.set_title(f"$M_{k+1}$\nMean $|\\Delta|$={mean_str}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    if k == 4:
        ax.set_xlabel("DISRUPTED", color="red", fontweight="bold")
fig.suptitle("Per-module disruption $|W_{\\mathrm{cancer}} - W_{\\mathrm{normal}}|$",
             fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES / "module_disruption.pdf", dpi=150, bbox_inches="tight")
plt.close("all")

print("\nDone! All figures saved to tex_paper/figures/")
print(f"  {len(list(FIGURES.glob('*.pdf')))} PDF files generated.")
