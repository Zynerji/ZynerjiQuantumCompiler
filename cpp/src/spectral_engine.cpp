/**
 * Sparse Dual-Helix Laplacian construction and eigensolver.
 *
 * Phase 5 C++ hot path — only pursue if Python eigsh is a bottleneck.
 * Uses Eigen3 for sparse matrices and Spectra for eigensolving.
 */

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <algorithm>

namespace zynerji_qc {

static constexpr double PHI = 1.6180339887498949;  // Golden ratio

struct SpectralResult {
    Eigen::MatrixXd eigenvectors;  // (n, k) matrix
    Eigen::VectorXd eigenvalues;   // (k,) vector
};

/**
 * Build a sparse helical Laplacian from a weighted adjacency matrix.
 *
 * @param adj_rows    Row indices of non-zero entries (upper triangle)
 * @param adj_cols    Column indices
 * @param adj_vals    Edge weights
 * @param n           Number of nodes
 * @param handedness  +1.0 for cos (right) helix, -1.0 for sin (left)
 * @param omega       Phase frequency
 * @param c_log       Log spacing constant
 * @param twist_frac  Mobius twist threshold (fraction of n)
 * @return Sparse Laplacian matrix
 */
Eigen::SparseMatrix<double> build_sparse_laplacian(
    const std::vector<int>& adj_rows,
    const std::vector<int>& adj_cols,
    const std::vector<double>& adj_vals,
    int n,
    double handedness,
    double omega,
    double c_log,
    double twist_frac
) {
    // Angular coordinates (log spacing)
    Eigen::VectorXd theta(n);
    for (int i = 0; i < n; ++i) {
        theta(i) = c_log * std::log(static_cast<double>(i + 1));
    }

    double coupling = (handedness >= 0) ? PHI : PHI * PHI;

    // Build Laplacian entries
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(adj_rows.size() * 2);
    Eigen::VectorXd degrees = Eigen::VectorXd::Zero(n);

    for (size_t e = 0; e < adj_rows.size(); ++e) {
        int u = adj_rows[e];
        int v = adj_cols[e];
        double w_orig = adj_vals[e];

        if (u >= v) continue;  // Upper triangle only

        double delta = theta(u) - theta(v);
        if (std::abs(u - v) > static_cast<int>(n * twist_frac)) {
            delta += M_PI;  // Mobius twist
        }

        double w;
        if (handedness >= 0) {
            w = std::cos(omega * delta) * w_orig * coupling;
        } else {
            w = std::sin(omega * delta) * w_orig * coupling;
        }

        if (w > 0) {
            triplets.emplace_back(u, v, -w);  // Off-diagonal (negated for L = D - A)
            triplets.emplace_back(v, u, -w);
            degrees(u) += w;
            degrees(v) += w;
        }
    }

    // Add diagonal (degree)
    for (int i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, degrees(i));
    }

    Eigen::SparseMatrix<double> L(n, n);
    L.setFromTriplets(triplets.begin(), triplets.end());
    return L;
}

/**
 * Compute k smallest non-trivial eigenvectors of a sparse Laplacian.
 *
 * Falls back to dense for small matrices. For large matrices,
 * uses Spectra (ARPACK-style) iterative solver.
 *
 * @param L  Sparse Laplacian matrix (n x n)
 * @param k  Number of eigenvectors to compute (excluding trivial zero)
 * @return SpectralResult with eigenvalues and eigenvectors
 */
SpectralResult compute_eigenvectors(
    const Eigen::SparseMatrix<double>& L,
    int k
) {
    int n = L.rows();
    SpectralResult result;

    if (n == 0) {
        result.eigenvalues = Eigen::VectorXd();
        result.eigenvectors = Eigen::MatrixXd(0, 0);
        return result;
    }

    // Dense fallback for small matrices
    // TODO: For large matrices, integrate Spectra eigensolver
    Eigen::MatrixXd L_dense = Eigen::MatrixXd(L);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(L_dense);

    int num_eigs = std::min(k, n - 1);
    // Skip trivial zero eigenvalue (index 0)
    int start = std::min(1, n - 1);
    int end = std::min(start + num_eigs, n);

    result.eigenvalues = solver.eigenvalues().segment(start, end - start);
    result.eigenvectors = solver.eigenvectors().middleCols(start, end - start);

    return result;
}

}  // namespace zynerji_qc
