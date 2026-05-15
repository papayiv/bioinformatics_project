"""
Biological data parser: converts bio data into complete weighted graphs.

Supported formats:
  - Gene Co-expression Networks (GCN) — Pearson correlation matrices
  - Protein distance matrices (Contact Maps)
  - Synthetic data generators for experiments
"""

import numpy as np
import networkx as nx
from pathlib import Path


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def generate_gcn_normal(
    n_genes: int = 50,
    n_modules: int = 5,
    intra_corr: float = 0.75,
    inter_corr: float = 0.05,
    noise: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, list[list[int]], list[str]]:
    """
    Generate a synthetic Gene Co-expression Network for normal (healthy) tissue.

    Simulates a block-correlation structure: genes within the same functional
    module are highly correlated; genes across modules are nearly uncorrelated.

    Parameters
    ----------
    n_genes    : total number of genes
    n_modules  : number of functional gene modules
    intra_corr : correlation within a module
    inter_corr : correlation across modules
    noise      : Gaussian noise added to correlations
    seed       : random seed for reproducibility

    Returns
    -------
    (W, modules, gene_names)
        W          : (n_genes, n_genes) correlation matrix
        modules    : list of gene index lists per module
        gene_names : list of gene name strings
    """
    rng = np.random.default_rng(seed)
    module_sizes = _split_evenly(n_genes, n_modules)
    modules = []
    start = 0
    for size in module_sizes:
        modules.append(list(range(start, start + size)))
        start += size

    W = np.full((n_genes, n_genes), inter_corr)
    for mod in modules:
        for i in mod:
            for j in mod:
                W[i, j] = intra_corr
    np.fill_diagonal(W, 1.0)

    noise_matrix = rng.normal(0, noise, (n_genes, n_genes))
    noise_matrix = (noise_matrix + noise_matrix.T) / 2
    np.fill_diagonal(noise_matrix, 0.0)
    W = W + noise_matrix
    W = np.clip(W, -1.0, 1.0)

    gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]
    return W, modules, gene_names


def generate_gcn_cancer(
    W_normal: np.ndarray,
    modules: list[list[int]],
    disrupted_modules: list[int] | None = None,
    disruption_strength: float = 0.6,
    noise: float = 0.08,
    seed: int = 99,
) -> np.ndarray:
    """
    Generate a synthetic GCN for cancer tissue by disrupting selected modules.

    Models the biological observation that oncogenesis disrupts the correlation
    architecture of specific gene modules while leaving others intact.

    Parameters
    ----------
    W_normal           : normal tissue correlation matrix
    modules            : module structure from generate_gcn_normal
    disrupted_modules  : indices of modules to disrupt (default: last module)
    disruption_strength: how much to randomize weights in disrupted modules
    noise              : global noise level
    seed               : random seed

    Returns
    -------
    W_cancer : (n, n) correlation matrix for cancer tissue
    """
    rng = np.random.default_rng(seed)
    n = W_normal.shape[0]
    W_cancer = W_normal.copy()

    if disrupted_modules is None:
        disrupted_modules = [len(modules) - 1]

    for mod_idx in disrupted_modules:
        mod = modules[mod_idx]
        for i in mod:
            for j in mod:
                if i != j:
                    # Replace high correlation with near-zero random value
                    W_cancer[i, j] = rng.uniform(-disruption_strength, disruption_strength)

    # Cross-module disruption: some previously uncorrelated genes become linked
    cancer_hub = modules[disrupted_modules[0]][0]
    for mod_idx, mod in enumerate(modules):
        if mod_idx not in disrupted_modules:
            gene = mod[0]
            W_cancer[cancer_hub, gene] = rng.uniform(0.4, 0.7)
            W_cancer[gene, cancer_hub] = W_cancer[cancer_hub, gene]

    noise_matrix = rng.normal(0, noise, (n, n))
    noise_matrix = (noise_matrix + noise_matrix.T) / 2
    np.fill_diagonal(noise_matrix, 0.0)
    W_cancer = W_cancer + noise_matrix
    W_cancer = np.clip(W_cancer, -1.0, 1.0)
    np.fill_diagonal(W_cancer, 1.0)

    # Symmetrize
    W_cancer = (W_cancer + W_cancer.T) / 2
    np.fill_diagonal(W_cancer, 1.0)
    return W_cancer


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_correlation_matrix(path: str | Path) -> np.ndarray:
    """
    Load a correlation matrix from a CSV file (no header, numeric values).

    The matrix must be square and symmetric. Values are clipped to [-1, 1].
    """
    W = np.loadtxt(path, delimiter=",")
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {W.shape}")
    W = np.clip(W, -1.0, 1.0)
    return W


def save_matrix(W: np.ndarray, path: str | Path) -> None:
    """Save weight matrix to CSV."""
    np.savetxt(path, W, delimiter=",", fmt="%.6f")


# ---------------------------------------------------------------------------
# Graph conversion
# ---------------------------------------------------------------------------

def matrix_to_networkx(
    W: np.ndarray,
    gene_names: list[str] | None = None,
    threshold: float = 0.3,
) -> nx.Graph:
    """
    Convert a weight matrix to a NetworkX graph, keeping only edges with
    |weight| >= threshold (for visualization clarity).

    Parameters
    ----------
    W          : weight matrix
    gene_names : node labels (defaults to 0, 1, 2, ...)
    threshold  : minimum absolute weight to include an edge

    Returns
    -------
    nx.Graph with 'weight' edge attribute
    """
    n = W.shape[0]
    labels = gene_names if gene_names else [str(i) for i in range(n)]
    n_use = min(n, len(labels))
    W = W[:n_use, :n_use]
    G = nx.Graph()
    G.add_nodes_from(labels[:n_use])
    for i in range(n_use):
        for j in range(i + 1, n_use):
            if abs(W[i, j]) >= threshold:
                G.add_edge(labels[i], labels[j], weight=float(W[i, j]))
    return G


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _split_evenly(total: int, n_parts: int) -> list[int]:
    base, rem = divmod(total, n_parts)
    return [base + (1 if i < rem else 0) for i in range(n_parts)]
