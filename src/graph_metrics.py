"""
Metrics and isomorphism measures for complete weighted graphs (CWGs).

Mathematical framework for the paper:
  "Isomorphism of Complete Weighted Graphs in Bioinformatics"
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.linalg import eigh
from itertools import permutations


# ---------------------------------------------------------------------------
# Basic distances
# ---------------------------------------------------------------------------

def spectral_distance(W1: np.ndarray, W2: np.ndarray) -> float:
    """
    Spectral distance between two CWGs.

    d_spec(G1, G2) = || sort(λ(W1)) - sort(λ(W2)) ||_2

    This is a permutation-invariant lower bound for the isomorphism metric:
        d_iso(G1, G2) >= (1/sqrt(2)) * d_spec(G1, G2)

    Parameters
    ----------
    W1, W2 : ndarray of shape (n, n)
        Symmetric weight matrices of two complete weighted graphs.

    Returns
    -------
    float
        Spectral distance.
    """
    if W1.shape != W2.shape:
        raise ValueError("Graphs must have the same number of vertices.")
    eigs1 = np.sort(np.linalg.eigvalsh(W1))
    eigs2 = np.sort(np.linalg.eigvalsh(W2))
    return float(np.linalg.norm(eigs1 - eigs2))


def frobenius_distance(W1: np.ndarray, W2: np.ndarray) -> float:
    """
    Frobenius distance (no reordering).

    Used as a quick upper bound when node correspondence is known.
    """
    return float(np.linalg.norm(W1 - W2, "fro"))


# ---------------------------------------------------------------------------
# Isomorphism metric
# ---------------------------------------------------------------------------

def isomorphism_metric(W1: np.ndarray, W2: np.ndarray, method: str = "hungarian") -> tuple[float, np.ndarray]:
    """
    Approximate isomorphism metric between two CWGs.

    d_iso(G1, G2) = min_{phi in S_n} max_{i,j} |W1[i,j] - W2[phi(i), phi(j)]|

    Exact computation is NP-hard. Two approaches are provided:

    method='exact'    — brute-force over all n! permutations (only for n <= 8).
    method='hungarian' — Hungarian algorithm on the L2 relaxation (polynomial).

    Returns
    -------
    (distance, permutation)
        Best found distance and the corresponding permutation array.
    """
    n = W1.shape[0]
    if W1.shape != W2.shape:
        raise ValueError("Graphs must have the same number of vertices.")

    if method == "exact":
        if n > 8:
            raise ValueError("Exact method is only feasible for n <= 8.")
        best_dist = np.inf
        best_perm = np.arange(n)
        for perm in permutations(range(n)):
            p = np.array(perm)
            W2p = W2[np.ix_(p, p)]
            dist = np.max(np.abs(W1 - W2p))
            if dist < best_dist:
                best_dist = dist
                best_perm = p
        return float(best_dist), best_perm

    # Hungarian relaxation: minimize sum of squared differences
    # Cost matrix: cost[i, j] = || W1[i, :] - W2[j, :] ||^2
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost[i, j] = np.sum((W1[i, :] - W2[j, :]) ** 2)
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = col_ind
    W2p = W2[np.ix_(perm, perm)]
    dist = float(np.max(np.abs(W1 - W2p)))
    return dist, perm


# ---------------------------------------------------------------------------
# Epsilon-isomorphism
# ---------------------------------------------------------------------------

def epsilon_isomorphism_check(
    W1: np.ndarray,
    W2: np.ndarray,
    epsilon: float,
    method: str = "hungarian",
) -> dict:
    """
    Check whether two CWGs are epsilon-isomorphic.

    G1 and G2 are epsilon-isomorphic iff there exists a bijection phi: V1 -> V2
    such that |W1[i,j] - W2[phi(i), phi(j)]| <= epsilon for all i, j.

    Returns
    -------
    dict with keys:
        'is_epsilon_isomorphic' : bool
        'epsilon_achieved'      : float  (the actual minimum epsilon found)
        'permutation'           : ndarray
        'spectral_lb'           : float  (spectral lower bound on d_iso)
    """
    dist, perm = isomorphism_metric(W1, W2, method=method)
    spec_lb = spectral_distance(W1, W2) / np.sqrt(2)
    return {
        "is_epsilon_isomorphic": dist <= epsilon,
        "epsilon_achieved": dist,
        "permutation": perm,
        "spectral_lb": spec_lb,
    }


# ---------------------------------------------------------------------------
# Gromov-Wasserstein approximation
# ---------------------------------------------------------------------------

def gromov_wasserstein_approx(
    W1: np.ndarray,
    W2: np.ndarray,
    n_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """
    Approximate Gromov-Wasserstein (GW) distance between two CWGs.

    Uses uniform marginals p = q = 1/n and iterative Sinkhorn-like updates
    for the transport plan T.

    GW(G1, G2) = min_{T in U(p,q)} sum_{i,j,k,l} |W1[i,k] - W2[j,l]|^2 * T[i,j] * T[k,l]

    This is a lower-bound approximation; the true GW minimization is non-convex.

    Returns
    -------
    float
        Approximate GW distance (squared).
    """
    n = W1.shape[0]
    # Uniform transport plan
    T = np.full((n, n), 1.0 / n**2)

    def _gw_loss(T_: np.ndarray) -> float:
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                C[i, j] = np.sum(
                    ((W1[i, :].reshape(-1, 1) - W2[j, :].reshape(1, -1)) ** 2) * T_
                )
        return float(np.sum(C * T_))

    prev_loss = np.inf
    for _ in range(n_iter):
        # Gradient step: compute cost matrix C[i,j] = sum_{k,l} |W1[i,k]-W2[j,l]|^2 T[k,l]
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                C[i, j] = np.sum(
                    ((W1[i, :].reshape(-1, 1) - W2[j, :].reshape(1, -1)) ** 2) * T
                )
        # Hungarian step: find optimal assignment under current C
        row_ind, col_ind = linear_sum_assignment(C)
        T_new = np.zeros((n, n))
        T_new[row_ind, col_ind] = 1.0 / n
        loss = _gw_loss(T_new)
        if abs(prev_loss - loss) < tol:
            T = T_new
            break
        T = T_new
        prev_loss = loss

    return float(np.sqrt(max(0.0, _gw_loss(T))))


# ---------------------------------------------------------------------------
# Differential analysis helpers
# ---------------------------------------------------------------------------

def differential_subgraph(
    W1: np.ndarray,
    W2: np.ndarray,
    threshold: float = 0.3,
) -> np.ndarray:
    """
    Return a binary mask of edges where the weight difference exceeds threshold.

    Used to identify "broken isomorphism" subgraphs in differential network analysis.

    Parameters
    ----------
    W1, W2    : weight matrices (same size)
    threshold : absolute difference cutoff

    Returns
    -------
    ndarray of bool, shape (n, n)
    """
    return np.abs(W1 - W2) > threshold


def module_isomorphism_scores(
    W1: np.ndarray,
    W2: np.ndarray,
    modules: list[list[int]],
) -> list[dict]:
    """
    Compute isomorphism metrics for each gene module (subgraph) independently.

    Parameters
    ----------
    W1, W2   : full weight matrices
    modules  : list of node index lists, each defining a subgraph

    Returns
    -------
    list of dicts with 'module', 'size', 'epsilon_achieved', 'spectral_distance'
    """
    results = []
    for mod in modules:
        idx = np.array(mod)
        sub1 = W1[np.ix_(idx, idx)]
        sub2 = W2[np.ix_(idx, idx)]
        dist, _ = isomorphism_metric(sub1, sub2)
        sd = spectral_distance(sub1, sub2)
        results.append({
            "module": mod,
            "size": len(mod),
            "epsilon_achieved": dist,
            "spectral_distance": sd,
        })
    return results
