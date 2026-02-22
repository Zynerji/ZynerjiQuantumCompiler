"""Sparse Dual-Helix spectral engine.

Adapted from HelicalATPG's measure_rho.py sparse construction.
Builds two helical Laplacians (cos/sin basis) with golden-ratio coupling,
extracts k eigenvectors, and applies spectral attenuation for graph matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

PHI = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio = 1.6180339887498949


@dataclass
class HelixParams:
    """Tunable parameters for the Dual-Helix spectral decomposition."""
    omega: float = 0.3           # Phase frequency
    c_log: float = 1.0           # Log spacing constant
    twist_fraction: float = 0.33 # Mobius twist threshold (fraction of n_nodes)
    k: int = 8                   # Number of eigenvectors per helix
    alpha: float = 3.0           # Spectral attenuation exponent
    use_helix: bool = True       # If False, use standard graph Laplacian (no phase modulation)


@dataclass
class SpectralCoords:
    """Weighted spectral coordinates from dual-helix decomposition."""
    coords: np.ndarray           # (n_nodes, 2k) weighted spectral coordinates
    eigenvalues_cos: np.ndarray  # Eigenvalues from cos (right) helix
    eigenvalues_sin: np.ndarray  # Eigenvalues from sin (left) helix
    params: HelixParams = field(repr=False)


def build_sparse_laplacian(
    adj: csr_matrix,
    handedness: float,
    params: HelixParams,
) -> csr_matrix:
    """Build a helical Laplacian from a weighted adjacency matrix.

    Parameters
    ----------
    adj : csr_matrix
        Weighted adjacency matrix (n x n), symmetric.
    handedness : float
        +1.0 for right (cos) helix, -1.0 for left (sin) helix.
    params : HelixParams
        Tuning parameters.

    Returns
    -------
    csr_matrix
        The helical Laplacian L = D - A_helix.
    """
    n = adj.shape[0]
    if n == 0:
        return csr_matrix((0, 0))

    # Angular coordinates (log spacing)
    theta = params.c_log * np.log(np.arange(1, n + 1, dtype=np.float64))

    # Coupling strength: right helix uses phi, left helix uses phi^2
    coupling = PHI if handedness >= 0 else PHI * PHI
    phase_fn = np.cos if handedness >= 0 else np.sin

    # Extract non-zero entries from upper triangle
    adj_coo = adj.tocoo()
    rows_out = []
    cols_out = []
    vals_out = []
    degrees = np.zeros(n, dtype=np.float64)

    for u, v, w_orig in zip(adj_coo.row, adj_coo.col, adj_coo.data):
        if u >= v:
            continue  # Upper triangle only to avoid double-counting

        delta = theta[u] - theta[v]
        # Mobius twist for topologically distant pairs
        if abs(u - v) > n * params.twist_fraction:
            delta += np.pi

        w = phase_fn(params.omega * delta) * w_orig * coupling
        if w > 0:
            rows_out.extend([u, v])
            cols_out.extend([v, u])
            vals_out.extend([w, w])
            degrees[u] += w
            degrees[v] += w

    A_helix = csr_matrix(
        (vals_out, (rows_out, cols_out)), shape=(n, n)
    )
    L = diags(degrees, format="csr") - A_helix
    return L


def build_standard_laplacian(adj: csr_matrix, normalized: bool = False) -> csr_matrix:
    """Build a graph Laplacian from an adjacency matrix.

    Handles both unsigned and signed adjacency matrices correctly.
    For signed graphs (with negative edges), uses |w| for degree computation
    to ensure the Laplacian is positive semi-definite (signed Laplacian).

    Parameters
    ----------
    adj : csr_matrix
        Weighted adjacency matrix (n x n), symmetric. May contain negative weights.
    normalized : bool
        If True, build the symmetric normalized Laplacian:
        L_sym = I - D^{-1/2} A D^{-1/2}
        If False, build the combinatorial Laplacian L = D - A.
    """
    n = adj.shape[0]
    if n == 0:
        return csr_matrix((0, 0))

    # Symmetrize (take upper triangle, mirror)
    adj_coo = adj.tocoo()
    rows_out = []
    cols_out = []
    vals_out = []
    degrees = np.zeros(n, dtype=np.float64)

    for u, v, w in zip(adj_coo.row, adj_coo.col, adj_coo.data):
        if u >= v:
            continue
        rows_out.extend([u, v])
        cols_out.extend([v, u])
        vals_out.extend([w, w])
        # Use |w| for degree to ensure L is PSD (signed Laplacian theory)
        degrees[u] += abs(w)
        degrees[v] += abs(w)

    A = csr_matrix((vals_out, (rows_out, cols_out)), shape=(n, n))

    if normalized:
        # L_sym = I - D^{-1/2} A D^{-1/2}
        d_inv_sqrt = np.zeros(n, dtype=np.float64)
        nonzero = degrees > 1e-12
        d_inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
        D_inv_sqrt = diags(d_inv_sqrt, format="csr")
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
        return diags(np.ones(n), format="csr") - A_norm

    return diags(degrees, format="csr") - A


def _compute_eigenvectors(
    L: csr_matrix,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute k smallest eigenvectors of a sparse Laplacian.

    Returns (eigenvalues[k], eigenvectors[n, k]).
    Skips the trivial zero eigenvalue (index 0).
    """
    n = L.shape[0]
    if n == 0:
        return np.array([]), np.zeros((0, 0))

    # Number of eigenvectors to request (k useful + 1 trivial)
    num_request = min(k + 1, n)

    if n < 10:
        # Dense fallback for tiny matrices
        evals, evecs = np.linalg.eigh(L.toarray())
        idx = np.argsort(evals)
        start = min(1, n - 1)
        end = min(start + k, n)
        return evals[idx[start:end]], evecs[:, idx[start:end]]

    try:
        evals, evecs = eigsh(L, k=num_request, which="SM", tol=1e-8, maxiter=2000)
        idx = np.argsort(evals)
        # Skip index 0 (trivial zero eigenvalue)
        start = min(1, len(idx) - 1)
        end = min(start + k, len(idx))
        return evals[idx[start:end]], evecs[:, idx[start:end]]
    except Exception:
        # Dense fallback
        evals, evecs = np.linalg.eigh(L.toarray())
        idx = np.argsort(evals)
        start = min(1, n - 1)
        end = min(start + k, n)
        return evals[idx[start:end]], evecs[:, idx[start:end]]


def _golden_ratio_indices(n_available: int, k: int) -> list[int]:
    """Select k indices from [0, n_available) using golden ratio spacing.

    From VHGT's anti-periodic frequency initialization: golden ratio spacing
    avoids harmonic relationships between selected eigenvectors, providing
    multi-scale structural information without redundancy.

    The golden ratio conjugate (~0.618) produces the most uniformly distributed
    sequence on [0, 1] — each new point falls in the largest existing gap.
    """
    if k >= n_available:
        return list(range(n_available))

    phi_conj = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618, golden ratio conjugate
    indices = []
    for i in range(k):
        # Golden ratio spacing on [0, 1], mapped to [0, n_available)
        frac = (i * phi_conj) % 1.0
        idx = int(frac * n_available)
        # Avoid duplicates — shift to next available
        while idx in indices:
            idx = (idx + 1) % n_available
        indices.append(idx)

    return sorted(indices)


def _compute_eigenvectors_golden(
    L: csr_matrix,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute eigenvectors at golden-ratio-spaced spectral indices.

    Instead of the k smallest eigenvectors (consecutive), this selects
    eigenvectors spread across the spectrum using golden ratio spacing.
    This provides multi-scale structural information (from smooth Fiedler
    modes to high-frequency oscillatory modes) without harmonic redundancy.
    """
    n = L.shape[0]
    if n == 0:
        return np.array([]), np.zeros((0, 0))

    # Compute all eigenvectors (or as many as we can)
    if n < 50:
        evals, evecs = np.linalg.eigh(L.toarray())
    else:
        # Request more eigenvectors than we need, then select
        n_request = min(max(3 * k, 20), n)
        try:
            evals, evecs = eigsh(L, k=n_request, which="SM", tol=1e-8, maxiter=2000)
        except Exception:
            evals, evecs = np.linalg.eigh(L.toarray())

    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Skip trivial zero eigenvalue
    if len(evals) > 1:
        evals = evals[1:]
        evecs = evecs[:, 1:]

    n_available = len(evals)
    if n_available == 0:
        return np.array([]), np.zeros((n, 0))

    # Select k indices using golden ratio spacing
    selected = _golden_ratio_indices(n_available, k)

    return evals[selected], evecs[:, selected]


def compute_spectral_coords(
    adj: csr_matrix,
    params: Optional[HelixParams] = None,
    golden_selection: bool = False,
) -> SpectralCoords:
    """Compute dual-helix spectral coordinates for a graph.

    Builds both cos (right) and sin (left) helical Laplacians,
    extracts k eigenvectors from each, applies spectral attenuation,
    and concatenates into a (n_nodes, 2k) coordinate matrix.

    Parameters
    ----------
    adj : csr_matrix
        Weighted adjacency matrix (n x n), symmetric.
    params : HelixParams, optional
        Tuning parameters. Uses defaults if None.

    Returns
    -------
    SpectralCoords
        Weighted spectral coordinates and metadata.
    """
    if params is None:
        params = HelixParams()

    n = adj.shape[0]
    if n == 0:
        return SpectralCoords(
            coords=np.zeros((0, 2 * params.k)),
            eigenvalues_cos=np.array([]),
            eigenvalues_sin=np.array([]),
            params=params,
        )

    if params.use_helix:
        # Dual-helix mode: two phase-modulated Laplacians (cos + sin)
        L_cos = build_sparse_laplacian(adj, +1.0, params)
        L_sin = build_sparse_laplacian(adj, -1.0, params)
    else:
        # Standard mode: single graph Laplacian, used for both slots
        L_std = build_standard_laplacian(adj)
        L_cos = L_std
        L_sin = L_std

    # Extract eigenvectors
    eigfn = _compute_eigenvectors_golden if golden_selection else _compute_eigenvectors
    evals_cos, evecs_cos = eigfn(L_cos, params.k)
    if params.use_helix:
        evals_sin, evecs_sin = eigfn(L_sin, params.k)
    else:
        # In standard mode, only use one set of eigenvectors (no duplication)
        evals_sin, evecs_sin = evals_cos, evecs_cos

    # Spectral attenuation: w_m = exp(-alpha * lambda_m / lambda_max)
    def attenuate(evals: np.ndarray, evecs: np.ndarray) -> np.ndarray:
        if len(evals) == 0:
            return np.zeros((n, 0))
        lam_max = evals[-1] if evals[-1] > 1e-12 else 1.0
        weights = np.exp(-params.alpha * evals / lam_max)
        return evecs * weights[np.newaxis, :]

    coords_cos = attenuate(evals_cos, evecs_cos)
    coords_sin = attenuate(evals_sin, evecs_sin)

    # Concatenate: (n, k_cos) + (n, k_sin) → (n, 2k)
    coords = np.hstack([coords_cos, coords_sin])

    return SpectralCoords(
        coords=coords,
        eigenvalues_cos=evals_cos,
        eigenvalues_sin=evals_sin,
        params=params,
    )
