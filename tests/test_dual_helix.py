"""Tests for the Dual-Helix spectral engine."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from zynerji_qc.core.dual_helix import (
    PHI,
    HelixParams,
    SpectralCoords,
    build_sparse_laplacian,
    compute_spectral_coords,
    _compute_eigenvectors,
)


def _make_ring_adj(n: int) -> csr_matrix:
    """Create adjacency matrix for a ring graph of n nodes."""
    rows, cols, vals = [], [], []
    for i in range(n):
        j = (i + 1) % n
        rows.extend([i, j])
        cols.extend([j, i])
        vals.extend([1.0, 1.0])
    return csr_matrix((vals, (rows, cols)), shape=(n, n))


def _make_complete_adj(n: int) -> csr_matrix:
    """Create adjacency matrix for a complete graph."""
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            rows.extend([i, j])
            cols.extend([j, i])
            vals.extend([1.0, 1.0])
    return csr_matrix((vals, (rows, cols)), shape=(n, n))


class TestConstants:
    def test_phi_value(self):
        assert abs(PHI - 1.6180339887498949) < 1e-10

    def test_phi_golden_property(self):
        # PHI^2 = PHI + 1
        assert abs(PHI * PHI - PHI - 1.0) < 1e-10


class TestHelixParams:
    def test_defaults(self):
        p = HelixParams()
        assert p.omega == 0.3
        assert p.c_log == 1.0
        assert p.twist_fraction == 0.33
        assert p.k == 8
        assert p.alpha == 3.0

    def test_custom(self):
        p = HelixParams(omega=0.5, k=12, alpha=5.0)
        assert p.omega == 0.5
        assert p.k == 12
        assert p.alpha == 5.0


class TestBuildSparseLaplacian:
    def test_empty_graph(self):
        adj = csr_matrix((0, 0))
        L = build_sparse_laplacian(adj, +1.0, HelixParams())
        assert L.shape == (0, 0)

    def test_ring_shape(self):
        adj = _make_ring_adj(10)
        L = build_sparse_laplacian(adj, +1.0, HelixParams())
        assert L.shape == (10, 10)

    def test_laplacian_row_sums_zero(self):
        """Laplacian rows should sum to zero (or near-zero for filtered edges)."""
        adj = _make_ring_adj(8)
        L = build_sparse_laplacian(adj, +1.0, HelixParams())
        row_sums = np.abs(L.toarray().sum(axis=1))
        assert np.all(row_sums < 1e-10)

    def test_laplacian_symmetric(self):
        adj = _make_ring_adj(8)
        L = build_sparse_laplacian(adj, +1.0, HelixParams())
        L_dense = L.toarray()
        assert np.allclose(L_dense, L_dense.T, atol=1e-12)

    def test_laplacian_positive_semidefinite(self):
        adj = _make_ring_adj(8)
        L = build_sparse_laplacian(adj, +1.0, HelixParams())
        eigenvalues = np.linalg.eigvalsh(L.toarray())
        assert np.all(eigenvalues >= -1e-10)

    def test_cos_sin_different(self):
        """Right (cos) and left (sin) helices should produce different Laplacians."""
        adj = _make_ring_adj(10)
        params = HelixParams()
        L_cos = build_sparse_laplacian(adj, +1.0, params)
        L_sin = build_sparse_laplacian(adj, -1.0, params)
        assert not np.allclose(L_cos.toarray(), L_sin.toarray())

    def test_coupling_constants(self):
        """Right helix uses PHI, left uses PHI^2."""
        adj = _make_ring_adj(6)
        params = HelixParams()
        L_r = build_sparse_laplacian(adj, +1.0, params)
        L_l = build_sparse_laplacian(adj, -1.0, params)
        # They should have different magnitudes due to coupling difference
        assert not np.isclose(
            np.abs(L_r.toarray()).sum(),
            np.abs(L_l.toarray()).sum(),
        )


class TestComputeEigenvectors:
    def test_ring_eigenvectors(self):
        adj = _make_ring_adj(10)
        L = build_sparse_laplacian(adj, +1.0, HelixParams())
        evals, evecs = _compute_eigenvectors(L, k=4)
        assert len(evals) == 4
        assert evecs.shape == (10, 4)
        # Eigenvalues should be positive (skipping trivial zero)
        assert np.all(evals > -1e-8)

    def test_small_graph_dense_fallback(self):
        adj = _make_ring_adj(5)
        L = build_sparse_laplacian(adj, +1.0, HelixParams())
        evals, evecs = _compute_eigenvectors(L, k=3)
        assert len(evals) <= 3
        assert evecs.shape[0] == 5

    def test_k_larger_than_graph(self):
        """If k > n-1, should return as many eigenvectors as available."""
        adj = _make_ring_adj(4)
        L = build_sparse_laplacian(adj, +1.0, HelixParams())
        evals, evecs = _compute_eigenvectors(L, k=10)
        assert len(evals) <= 3  # At most n-1 non-trivial
        assert evecs.shape[0] == 4


class TestComputeSpectralCoords:
    def test_ring_coords(self):
        adj = _make_ring_adj(10)
        coords = compute_spectral_coords(adj)
        assert isinstance(coords, SpectralCoords)
        assert coords.coords.shape[0] == 10
        assert coords.coords.shape[1] > 0

    def test_complete_graph_coords(self):
        adj = _make_complete_adj(8)
        coords = compute_spectral_coords(adj, HelixParams(k=4))
        assert coords.coords.shape[0] == 8

    def test_empty_graph(self):
        adj = csr_matrix((0, 0))
        coords = compute_spectral_coords(adj)
        assert coords.coords.shape == (0, 16)

    def test_spectral_attenuation(self):
        """Higher alpha should attenuate higher eigenvectors more."""
        adj = _make_ring_adj(10)
        c_low = compute_spectral_coords(adj, HelixParams(alpha=1.0, k=4))
        c_high = compute_spectral_coords(adj, HelixParams(alpha=10.0, k=4))
        # High alpha should have smaller coordinate magnitudes overall
        norm_low = np.linalg.norm(c_low.coords)
        norm_high = np.linalg.norm(c_high.coords)
        assert norm_high < norm_low

    def test_different_params_different_coords(self):
        adj = _make_ring_adj(10)
        c1 = compute_spectral_coords(adj, HelixParams(omega=0.1))
        c2 = compute_spectral_coords(adj, HelixParams(omega=0.9))
        assert not np.allclose(c1.coords, c2.coords)

    def test_coords_finite(self):
        adj = _make_ring_adj(20)
        coords = compute_spectral_coords(adj)
        assert np.all(np.isfinite(coords.coords))
