"""
Visualization utilities for complete weighted graph analysis.

All plot functions accept an optional `save_path` argument; when provided,
the figure is saved to that path (PDF for LaTeX integration) and also shown.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
from pathlib import Path


_CMAP_CORR = "RdBu_r"   # diverging: red=positive, blue=negative correlation
_CMAP_DIFF = "PiYG"     # for difference matrices
_FIG_DPI = 150


def _save_or_show(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is not None:
        fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

def plot_heatmap(
    W: np.ndarray,
    title: str = "Weight Matrix",
    gene_names: list[str] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple = (8, 7),
) -> plt.Figure:
    """
    Plot a heatmap of a complete weighted graph's weight matrix.
    """
    fig, ax = plt.subplots(figsize=figsize)
    labels = gene_names if gene_names else False
    sns.heatmap(
        W,
        ax=ax,
        cmap=_CMAP_CORR,
        vmin=-1,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Correlation coefficient"},
        linewidths=0 if W.shape[0] > 30 else 0.1,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.tick_params(axis="both", labelsize=6 if W.shape[0] > 20 else 9)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_difference_heatmap(
    W1: np.ndarray,
    W2: np.ndarray,
    label1: str = "Normal",
    label2: str = "Cancer",
    threshold: float | None = None,
    modules: list[list[int]] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple = (9, 7),
) -> plt.Figure:
    """
    Plot |W2 - W1| highlighting structurally disrupted edges.

    Optionally overlays module boundaries and marks edges exceeding `threshold`.
    """
    diff = W2 - W1
    fig, axes = plt.subplots(1, 3, figsize=(figsize[0] * 1.5, figsize[1]))

    for ax, mat, title, vmin, vmax, cmap in [
        (axes[0], W1, f"G₁ ({label1})", -1, 1, _CMAP_CORR),
        (axes[1], W2, f"G₂ ({label2})", -1, 1, _CMAP_CORR),
        (axes[2], diff, f"Δ = G₂ − G₁", -1, 1, _CMAP_DIFF),
    ]:
        sns.heatmap(mat, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                    cbar_kws={"shrink": 0.8}, linewidths=0)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    if threshold is not None:
        mask = np.abs(diff) > threshold
        xs, ys = np.where(mask)
        axes[2].scatter(ys + 0.5, xs + 0.5, s=2, c="black", alpha=0.6,
                        label=f"|Δ|>{threshold:.2f}")
        axes[2].legend(loc="upper right", fontsize=7)

    if modules is not None:
        for mod in modules:
            lo, hi = min(mod), max(mod) + 1
            for ax in axes:
                for lo_, hi_ in [(lo, hi)]:
                    rect = plt.Rectangle(
                        (lo_, lo_), hi_ - lo_, hi_ - lo_,
                        linewidth=1.5, edgecolor="black", facecolor="none"
                    )
                    ax.add_patch(rect)

    fig.suptitle(
        f"Differential Network Analysis: {label1} vs {label2}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Network graphs
# ---------------------------------------------------------------------------

def plot_network(
    G: nx.Graph,
    title: str = "Gene Co-expression Network",
    modules: list[list[int]] | None = None,
    node_labels: list[str] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Draw a complete weighted graph as a network.

    Edge color encodes weight sign (red=positive, blue=negative).
    Edge width encodes |weight|.
    """
    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(G, seed=42, k=1.5)

    # Node colors by module
    node_list = list(G.nodes())
    node_colors = _module_colors(node_list, modules)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=120, alpha=0.9)

    edges = G.edges(data="weight", default=0)
    edge_list = [(u, v) for u, v, _ in edges]
    weights = [d for _, _, d in G.edges(data="weight", default=0)]
    edge_colors = ["#d62728" if w > 0 else "#1f77b4" for w in weights]
    edge_widths = [max(0.3, abs(w) * 3) for w in weights]

    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edge_list,
                           edge_color=edge_colors, width=edge_widths, alpha=0.6)

    if node_labels and len(node_labels) <= 30:
        labels = {node_list[i]: node_labels[i] for i in range(len(node_list))}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=5)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")

    legend_handles = [
        mpatches.Patch(color="#d62728", label="Positive correlation"),
        mpatches.Patch(color="#1f77b4", label="Negative correlation"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------

def plot_eigenvalue_spectrum(
    W1: np.ndarray,
    W2: np.ndarray,
    label1: str = "Normal",
    label2: str = "Cancer",
    save_path: str | Path | None = None,
    figsize: tuple = (9, 4),
) -> plt.Figure:
    """
    Plot sorted eigenvalue spectra of two weight matrices side by side.

    The eigenvalue spectrum is a permutation-invariant graph fingerprint,
    used as a lower bound for the isomorphism metric.
    """
    eigs1 = np.sort(np.linalg.eigvalsh(W1))[::-1]
    eigs2 = np.sort(np.linalg.eigvalsh(W2))[::-1]
    x = np.arange(len(eigs1))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(x, eigs1, "o-", color="#2ca02c", markersize=4, label=label1)
    axes[0].plot(x, eigs2, "s--", color="#ff7f0e", markersize=4, label=label2)
    axes[0].set_title("Eigenvalue spectra", fontweight="bold")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].bar(x, np.abs(eigs1 - eigs2), color="#9467bd", alpha=0.8)
    axes[1].set_title(r"$|\lambda_i(G_1) - \lambda_i(G_2)|$", fontweight="bold")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Absolute difference")
    axes[1].grid(alpha=0.3)

    spec_dist = float(np.linalg.norm(eigs1 - eigs2))
    fig.suptitle(
        f"Spectral distance d_spec(G₁, G₂) = {spec_dist:.4f}  "
        f"(lower bound on d_iso: {spec_dist / np.sqrt(2):.4f})",
        fontsize=11,
    )
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Summary bar chart
# ---------------------------------------------------------------------------

def plot_metric_comparison(
    metrics: dict[str, dict[str, float]],
    save_path: str | Path | None = None,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Bar chart comparing multiple isomorphism metrics across graph pairs.

    Parameters
    ----------
    metrics : {pair_label: {metric_name: value}}
    """
    pairs = list(metrics.keys())
    metric_names = list(next(iter(metrics.values())).keys())
    x = np.arange(len(pairs))
    bar_width = 0.8 / len(metric_names)
    colors = plt.cm.tab10(np.linspace(0, 0.7, len(metric_names)))

    fig, ax = plt.subplots(figsize=figsize)
    for k, (name, color) in enumerate(zip(metric_names, colors)):
        vals = [metrics[p][name] for p in pairs]
        ax.bar(x + k * bar_width, vals, bar_width, label=name, color=color, alpha=0.85)

    ax.set_xticks(x + bar_width * (len(metric_names) - 1) / 2)
    ax.set_xticklabels(pairs, fontsize=10)
    ax.set_ylabel("Distance / metric value")
    ax.set_title("Comparison of CWG isomorphism metrics", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _module_colors(node_list: list, modules: list[list[int]] | None) -> list:
    palette = plt.cm.Set2(np.linspace(0, 1, max(len(modules), 2) if modules else 1))
    if modules is None:
        return ["#1f77b4"] * len(node_list)
    node_to_color = {}
    for k, mod in enumerate(modules):
        for idx in mod:
            label = str(idx) if isinstance(idx, int) else idx
            node_to_color[label] = palette[k]
    return [node_to_color.get(str(n), "#7f7f7f") for n in node_list]
